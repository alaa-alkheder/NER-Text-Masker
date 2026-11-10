"""Rules added from the second study ("NER model comparison.ipynb", Colab, Nov 2026).
Each test reproduces a failure seen in that notebook with fake models, then checks the fix."""
import pytest

from masker import DetectionSettings, Document, Masker, MaskOptions
from masker.pipeline import MaskingPipeline, ModelProvider
from masker.strategies import partial_mask
from masker.text_utils import lacks_case_signal, name_parts, split_by_case, strip_title

from conftest import FakeHF, FakeZero, _scan


def by_text(findings):
    return {(f.text, f.label, f.source) for f in findings}


class GazHF(FakeHF):
    def __init__(self, gaz):
        super().__init__()
        self.gaz = gaz

    def predict(self, texts):
        self.calls.append(list(texts))
        return [_scan(t, self.gaz) for t in texts]


class GazZero(FakeZero):
    def __init__(self, gaz):
        super().__init__()
        self.gaz = gaz

    def predict(self, texts, prompts, threshold=0.5):
        self.calls.append((list(texts), list(prompts), threshold))
        return [[e for e in _scan(t, self.gaz) if e["label"] in prompts] for t in texts]


def make_pipeline(hf_gaz=None, zero_gaz=None):
    fakes = {"en_ner": GazHF(hf_gaz or {}), "ar_ner": GazHF({}),
             "en_zero": GazZero(zero_gaz or {}), "ar_zero": GazZero(zero_gaz or {})}
    return MaskingPipeline(ModelProvider(factory=lambda role: fakes[role])), fakes


# ----------------------------------------------------------------------------- lower-case text
LOWER = "lina khoury works for amazon in dubai."


def test_case_signal_helpers():
    assert lacks_case_signal(LOWER) and lacks_case_signal("THE UN MET IN GENEVA.")
    assert not lacks_case_signal("Barack Obama met officials.") and not lacks_case_signal("ok.")
    text = "Barack Obama met officials. " + LOWER + " Then Obama left."
    assert [(text[s:e], c) for s, e, c in split_by_case(text, 0, len(text))] == [
        ("Barack Obama met officials.", False), (LOWER, True), ("Then Obama left.", False)]


def test_lowercase_sentence_goes_to_zero_shot():
    # dslim (cased) finds nothing in an all-lowercase sentence; NuNER_Zero finds all three
    pipe, fakes = make_pipeline(hf_gaz={"Barack Obama": "PER"},
                                zero_gaz={"lina khoury": "person", "amazon": "organization", "dubai": "location"})
    segs = ["Barack Obama visited London. " + LOWER]
    fs = pipe.detect(segs, DetectionSettings(labels=["PERSON", "ORGANIZATION", "LOCATION"]))
    got = by_text(fs)
    assert {("lina khoury", "PERSON", "zero"), ("amazon", "ORGANIZATION", "zero"),
            ("dubai", "LOCATION", "zero")} <= got
    assert ("Barack Obama", "PERSON", "hf") in got
    assert all(LOWER not in t for call in fakes["en_ner"].calls for t in call)
    assert pipe.stats["chunks_en_nocase"] == 1


def test_lowercase_fallback_can_be_turned_off():
    pipe, fakes = make_pipeline(zero_gaz={"lina khoury": "person"})
    fs = pipe.detect([LOWER], DetectionSettings(labels=["PERSON"], caseless_fallback=False))
    assert fs == [] and fakes["en_ner"].calls == [[LOWER]]


# ----------------------------------------------------------------------------- name parts
SAMPLE = ["Ahmed Al-Khatib met Sara Haddad at Microsoft in Seattle.",
          "Later, Ahmed e-mailed sara.haddad@example.com and called +963 944 123 456.",
          "Sara replied from Damascus on 2024-10-31."]


def test_name_parts_helper():
    from masker.config import NAME_PARTICLES, TITLES_AR, TITLES_EN
    titles = TITLES_EN | TITLES_AR
    assert name_parts("Ahmed Al-Khatib", NAME_PARTICLES, titles) == ["Ahmed", "Al-Khatib"]
    assert name_parts("Khalid bin Saeed", NAME_PARTICLES, titles) == ["Khalid", "Saeed"]
    assert name_parts("عبد الله الأحمد", NAME_PARTICLES, titles) == ["عبد الله", "الأحمد"]  # never 'الله' alone
    assert name_parts("Omar", NAME_PARTICLES, titles) == []


def test_first_name_alone_is_masked_after_full_name():
    # the zero-shot model found the full names but not 'Ahmed' / 'Sara' alone (section 9)
    pipe, _ = make_pipeline(hf_gaz={"Ahmed Al-Khatib": "PER", "Sara Haddad": "PER", "Microsoft": "ORG",
                                    "Seattle": "LOC", "Damascus": "LOC"})
    s = DetectionSettings(labels=["PERSON", "ORGANIZATION", "LOCATION", "EMAIL", "PHONE"])
    fs = pipe.detect(SAMPLE, s)
    got = by_text(fs)
    assert ("Ahmed", "PERSON", "propagated") in got and ("Sara", "PERSON", "propagated") in got
    assert ("sara.haddad@example.com", "EMAIL", "regex") in got  # the e-mail outranks the name part

    Masker("pseudonym").assign(fs, SAMPLE)
    reps = {f.text: f.replacement for f in fs}
    assert reps["Ahmed"] == reps["Ahmed Al-Khatib"] == "[PERSON_1]"
    assert reps["Sara"] == reps["Sara Haddad"] == "[PERSON_2]"

    off = pipe.detect(SAMPLE, DetectionSettings(labels=s.labels, propagate_name_parts=False))
    assert ("Ahmed", "PERSON", "propagated") not in by_text(off)


def test_common_words_are_not_propagated():
    pipe, _ = make_pipeline(hf_gaz={"Will Smith": "PER", "Jordan Peterson": "PER", "Jordan": "LOC"})
    segs = ["Will Smith met Jordan Peterson.", "The talks will continue. Will they agree?", "He flew to Jordan."]
    fs = pipe.detect(segs, DetectionSettings(labels=["PERSON", "LOCATION"]))
    got = by_text(fs)
    assert not any(t == "Will" for t, _, _ in got)  # 'will' appears in lower case -> common word
    assert ("Jordan", "LOCATION", "hf") in got and ("Jordan", "PERSON", "propagated") not in got


# ----------------------------------------------------------------------------- titles / labels
def test_titles_stay_outside_person_spans():
    pipe, _ = make_pipeline(hf_gaz={"Dr. Sarah Johnson": "PER"}, zero_gaz={"الدكتور خالد النجار": "person"})
    fs = pipe.detect(["Dr. Sarah Johnson presented.", "قدّم الدكتور خالد النجار التقرير."],
                     DetectionSettings(labels=["PERSON"]))
    texts = {f.text for f in fs}
    assert {"Sarah Johnson", "خالد النجار"} <= texts and "Dr. Sarah Johnson" not in texts
    seg = "Prof. Dr. Ali Hassan"
    assert seg[slice(*strip_title(seg, 0, len(seg), frozenset({"prof", "dr"})))] == "Ali Hassan"
    keep = pipe.detect(["Dr. Sarah Johnson presented."], DetectionSettings(labels=["PERSON"], strip_titles=False))
    assert keep[0].text == "Dr. Sarah Johnson"


def test_hospitals_and_universities_become_organizations():
    pipe, _ = make_pipeline(hf_gaz={"Mayo Clinic": "LOC", "Boston": "LOC"}, zero_gaz={"جامعة حلب": "location"})
    fs = pipe.detect(["He moved from Boston to the Mayo Clinic.", "تعمل في جامعة حلب."],
                     DetectionSettings(labels=["ORGANIZATION", "LOCATION"]))
    got = {(f.text, f.label) for f in fs}
    assert {("Mayo Clinic", "ORGANIZATION"), ("جامعة حلب", "ORGANIZATION"), ("Boston", "LOCATION")} <= got
    # ORGANIZATION not selected -> keep the span as LOCATION so it is still masked
    only_loc = pipe.detect(["He moved to the Mayo Clinic."], DetectionSettings(labels=["LOCATION"]))
    assert [(f.text, f.label) for f in only_loc] == [("Mayo Clinic", "LOCATION")]


# ----------------------------------------------------------------------------- partial masking
def test_partial_masking_does_not_leak_number_structure():
    assert partial_mask("+963 944 123 456", "PHONE") == "+*** *** *** *56"
    assert partial_mask("4111 1111 1111 1111", "CREDIT_CARD") == "**** **** **** **11"
    assert partial_mask("sara.haddad@example.com", "EMAIL") == "s**********@e******.com"
    assert partial_mask("Barack Obama", "PERSON") == "B***** O****"
    f = Masker("partial")
    from masker.entities import Finding
    phone = Finding(0, 0, 16, "PHONE", "+963 944 123 456", source="regex")
    assert f.replacement(phone) == "+*** *** *** *56"


# ----------------------------------------------------------------------------- long inputs
class StridePipe:
    def __init__(self):
        self.kwargs = []

    def __call__(self, texts, batch_size=16, stride=None):
        self.kwargs.append({"batch_size": batch_size, "stride": stride})
        return [[] for _ in texts]


def test_hf_wrapper_passes_stride():
    from masker.detectors.model_detectors import HFTokenDetector
    p = StridePipe()
    HFTokenDetector("x", pipe=p, device="cpu").predict(["a b c"])
    assert p.kwargs[-1]["stride"] == 64


def test_real_pipeline_reads_past_token_512():
    """Without stride the transformers pipeline silently drops everything after token 512
    (second study, section 6). With the wrapper's stride the end of the text is read too."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import os
    import tempfile

    from masker.detectors.model_detectors import HFTokenDetector

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "obama", "met", "in", "baghdad", "."]
    with tempfile.TemporaryDirectory() as tmp:
        vocab = os.path.join(tmp, "vocab.txt")
        with open(vocab, "w") as fh:
            fh.write("\n".join(words))
        tok = transformers.BertTokenizerFast(vocab_file=vocab, do_lower_case=True, model_max_length=512)
        labels = ["O", "B-PER", "I-PER"]
        cfg = transformers.BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=32, num_labels=3,
                                      max_position_embeddings=512,
                                      id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
        torch.manual_seed(0)
        model = transformers.BertForTokenClassification(cfg).eval()
        with torch.no_grad():  # every token predicted as B-PER, so entities exist everywhere
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([0.0, 5.0, 0.0]))
        pipe = transformers.pipeline("token-classification", model=model, tokenizer=tok,
                                     aggregation_strategy="max", device="cpu")
        text = "obama met in baghdad . " * 300  # ~1500 tokens
        cut = tok(text, return_offsets_mapping=True)["offset_mapping"][510][1]
        ents = HFTokenDetector("tiny", pipe=pipe, device="cpu").predict([text])[0]
        assert ents and max(e["end"] for e in ents) > len(text) - 30 > cut
