"""Release 1.2.0: Arabic spelling variants in the name search, and prepositions / stop words
at the edges of model spans."""
from masker import DetectionSettings, Masker
from masker.config import STOPWORDS
from masker.text_utils import find_occurrences, fold_for_search, normalize_key, strip_stopwords

from test_study_fixes import by_text, make_pipeline


def texts(text, needle):
    return [text[s:e] for s, e in find_occurrences(text, needle)]


def test_search_ignores_hamza_alef_maqsura_ta_marbuta_and_diacritics():
    text = "قابل أَحْمَد واحمد وإبراهيم في مكه، ثم زار مصطفي ومسوول المكتب. احمدي ليس هو."
    assert texts(text, "أحمد") == ["أَحْمَد", "احمد"]  # proclitic و stays outside, احمدي is another word
    assert texts(text, "ابراهيم") == ["إبراهيم"]
    assert texts(text, "مكة") == ["مكه"]
    assert texts(text, "مصطفى") == ["مصطفي"]
    assert texts(text, "مسؤول") == ["مسوول"]  # hamza seat written without the hamza


def test_folded_offsets_point_into_the_original_text():
    folded, index = fold_for_search("أَحْمَد")
    assert folded == "احمد" and index == [0, 2, 4, 6]
    text = "زار أَحْمَدُ دمشق"
    (s, e), = find_occurrences(text, "احمد")
    assert text[s:e] == "أَحْمَدُ"  # the diacritic of the last letter is kept inside the span


def test_latin_search_is_unchanged():
    assert texts("Will met will. Willow", "Will") == ["Will"]


def test_replacement_key_matches_the_search():
    assert normalize_key("مسؤول") == normalize_key("مسوول")
    assert normalize_key("رئيس") == normalize_key("رييس")


def test_variant_spellings_get_the_same_pseudonym():
    pipe, _ = make_pipeline(zero_gaz={"أحمد الخطيب": "person"})
    segs = ["التقى أحمد الخطيب بالمدير.", "ثم غادر احمد الخطيب المبنى."]
    fs = pipe.detect(segs, DetectionSettings(labels=["PERSON"], propagate_name_parts=False))
    assert {f.text for f in fs} == {"أحمد الخطيب", "احمد الخطيب"}
    Masker("pseudonym").assign(fs, segs)
    assert len({f.replacement for f in fs}) == 1  # one pseudonym for both spellings


def test_stopwords_leave_the_edges_of_a_span():
    def strip(span):
        s, e = strip_stopwords(span, 0, len(span), STOPWORDS)
        return span[s:e]
    assert strip("في دمشق") == "دمشق"
    assert strip("من الرياض إلى") == "الرياض"
    assert strip("the United Nations") == "United Nations"
    assert strip("Bank of America") == "Bank of America"  # inner words stay
    assert strip("بدمشق") == "بدمشق"  # attached letters are never cut (بشار، وليد، لينا)
    assert strip("في") == ""


def test_pipeline_removes_prepositions_from_model_output():
    pipe, _ = make_pipeline(zero_gaz={"في دمشق": "location", "من": "location", "the United Nations": "organization"})
    segs = ["يعمل في دمشق منذ سنوات.", "He joined the United Nations."]
    s = DetectionSettings(labels=["LOCATION", "ORGANIZATION"], language="ar", propagate=False)
    got = {(f.text, f.label) for f in pipe.detect(segs[:1], s)}
    assert got == {("دمشق", "LOCATION")}
    pipe_en, _ = make_pipeline(hf_gaz={"the United Nations": "ORG"})
    en = pipe_en.detect(segs[1:], DetectionSettings(labels=["ORGANIZATION"], propagate=False))
    assert [f.text for f in en] == ["United Nations"]
    keep = pipe.detect(segs[:1], DetectionSettings(labels=["LOCATION"], language="ar", propagate=False,
                                                   strip_stopwords=False))
    assert "في دمشق" in {f.text for f in keep}
