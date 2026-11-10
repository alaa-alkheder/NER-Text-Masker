import io

import docx

from masker import DetectionSettings, Document, Masker, MaskOptions
from masker.text_utils import chunk_text, detect_language, find_occurrences


def by_text(findings):
    return {(f.text, f.label, f.source) for f in findings}


def test_language_and_chunking():
    assert detect_language("التقى محمد مع Google في دمشق") == "ar"
    assert detect_language("Barack Obama met officials") == "en"
    text = " ".join(f"w{i}." for i in range(250))
    chunks = chunk_text(text, max_words=100)
    assert len(chunks) == 3 and all(len(text[s:e].split()) <= 100 for s, e in chunks)
    long_sentence = " ".join(["word"] * 250)
    assert [len(long_sentence[s:e].split()) for s, e in chunk_text(long_sentence, 100)] == [100, 100, 50]


def test_arabic_proclitic_occurrences():
    t = "ومحمد وبمحمد محمدي محمد"
    spans = find_occurrences(t, "محمد")
    assert [t[s:e] for s, e in spans] == ["محمد", "محمد", "محمد"]
    assert spans[0][0] == 1  # the 'و' stays outside the span


def test_routing_english(pipeline, fakes, samples):
    doc = Document.load(samples / "sample_en.docx")
    s = DetectionSettings(labels=["PERSON", "ORGANIZATION", "LOCATION", "NATIONALITY", "DATE", "EMAIL", "URL",
                                  "PHONE", "IBAN", "CREDIT_CARD", "IP_ADDRESS", "ID_NUMBER"])
    fs = pipeline.detect(doc.texts, s)
    got = by_text(fs)
    assert ("Barack Obama", "PERSON", "hf") in got
    assert ("United Nations", "ORGANIZATION", "hf") in got
    assert ("Iraqi", "NATIONALITY", "zero") in got  # routed to the zero-shot model, not dslim MISC
    assert ("12 March 2024", "DATE", "regex") in got  # regex outranks the model on overlap
    assert ("john.smith@acme.com", "EMAIL", "regex") in got
    assert ("GB82 WEST 1234 5698 7654 32", "IBAN", "regex") in got
    assert ("548211930", "ID_NUMBER", "regex") in got
    # English text never goes to the Arabic model
    assert fakes["ar_zero"].calls == []
    # zero-shot call got the user label + distractors, with the notebook threshold 0.7
    _, prompts, thr = fakes["en_zero"].calls[0]
    assert {"nationality", "date"} <= set(prompts) and "person" in prompts and thr == 0.7


def test_routing_arabic(pipeline, fakes, samples):
    doc = Document.load(samples / "sample_ar.docx")
    fs = pipeline.detect(doc.texts, DetectionSettings(labels=["PERSON", "ORGANIZATION", "LOCATION", "PHONE"]))
    got = by_text(fs)
    assert ("محمد الأحمد", "PERSON", "zero") in got
    assert ("جامعة دمشق", "ORGANIZATION", "zero") in got
    assert ("+963 933 123 456", "PHONE", "regex") in got
    assert fakes["en_ner"].calls == [] or all(not any("؀" <= c <= "ۿ" for t in call for c in t)
                                             for call in fakes["en_ner"].calls)
    _, prompts, thr = fakes["ar_zero"].calls[0]
    assert thr == 0.4 and {"person", "organization", "location"} <= set(prompts)


def test_camelbert_backend(pipeline, fakes, samples):
    doc = Document.load(samples / "sample_ar.docx")
    pipeline.detect(doc.texts, DetectionSettings(labels=["PERSON", "NATIONALITY"], arabic_backend="camelbert"))
    assert fakes["ar_ner"].calls and fakes["ar_zero"].calls


def test_only_selected_labels_are_returned(pipeline, samples):
    doc = Document.load(samples / "sample_en.docx")
    fs = pipeline.detect(doc.texts, DetectionSettings(labels=["LOCATION"]))
    assert {f.label for f in fs} == {"LOCATION"}


def test_propagation_and_manual_terms(pipeline):
    segs = ["Obama spoke.", "Later Obama left with Zorblax."]
    s = DetectionSettings(labels=["PERSON"], manual_terms=[("Zorblax", "PERSON")], propagate=True)
    fs = pipeline.detect(segs, s)
    assert ("Zorblax", "PERSON", "manual") in by_text(fs)
    assert sum(1 for f in fs if f.text == "Obama") == 2


def test_custom_labels_zero_shot(pipeline, samples):
    doc = Document.load(samples / "sample_mixed.txt")
    fs = pipeline.detect(doc.texts, DetectionSettings(labels=["EMAIL"], custom_labels=["disease", "medication"]))
    got = by_text(fs)
    assert ("diabetes", "DISEASE", "zero") in got and ("metformin", "MEDICATION", "zero") in got


def test_model_failure_is_reported_not_fatal(samples):
    from masker import MaskingPipeline, ModelProvider

    def boom(role):
        raise OSError("no internet")

    p = MaskingPipeline(ModelProvider(factory=boom))
    fs = p.detect(["mail a@b.co from Bob"], DetectionSettings(labels=["PERSON", "EMAIL"]))
    assert [f.text for f in fs] == ["a@b.co"] and p.warnings


def test_end_to_end_docx(pipeline, samples, tmp_path):
    doc = Document.load(samples / "sample_ar.docx")
    fs = pipeline.detect(doc.texts, DetectionSettings(labels=["PERSON", "ORGANIZATION", "LOCATION", "PHONE",
                                                                "EMAIL", "IBAN", "ID_NUMBER", "DATE"]))
    Masker("pseudonym", options=MaskOptions(tag_language="ar")).assign(fs, doc.texts)
    out = docx.Document(io.BytesIO(doc.render(fs)))
    body = "\n".join(p.text for p in out.paragraphs)
    table = "\n".join(c.text for r in out.tables[0].rows for c in r.cells)
    header = out.sections[0].header.paragraphs[0].text
    for secret in ["محمد الأحمد", "خالد النجار", "khaled.najjar@example.com", "0944123456", "SA03", "01020304050"]:
        assert secret not in body
    assert "أحمد العلي" not in table and "حلب" not in table
    assert "سامر الخطيب" not in header
    assert body.count("[شخص_1]") == 2  # same person -> same pseudonym in both paragraphs
