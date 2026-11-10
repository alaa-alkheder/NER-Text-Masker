import pytest

from masker.entities import Finding
from masker.strategies import METHODS, Masker, MaskOptions


def F(text, label="PERSON", seg=0, start=0):
    return Finding(seg, start, start + len(text), label, text)


def test_tag_auto_language():
    m = Masker("tag")
    assert m.replacement(F("Barack Obama"), "Barack Obama met") == "[PERSON]"
    assert m.replacement(F("محمد الأحمد"), "التقى محمد الأحمد") == "[شخص]"
    assert Masker("tag", options=MaskOptions(tag_language="en")).replacement(F("محمد"), "محمد") == "[PERSON]"
    assert Masker("tag", options=MaskOptions(tag_style="angle")).replacement(F("Bob"), "Bob") == "<PERSON>"


def test_pseudonym_is_consistent_and_normalised():
    m = Masker("pseudonym", options=MaskOptions(tag_language="en"))
    a = m.replacement(F("أحمد"))
    b = m.replacement(F("Omar"))
    c = m.replacement(F("احمد"))  # hamza variant -> same entity
    d = m.replacement(F("Google", "ORGANIZATION"))
    assert (a, b, c, d) == ("[PERSON_1]", "[PERSON_2]", "[PERSON_1]", "[ORGANIZATION_1]")


def test_char_mask_and_partial():
    assert Masker("char_mask").replacement(F("Bob Lee")) == "●●● ●●●"
    assert Masker("char_mask", options=MaskOptions(mask_char="*")).replacement(F("Al")) == "**"
    assert Masker("partial").replacement(F("Barack Obama")) == "B***** O****"
    assert Masker("partial").replacement(F("خالد النجار")) == "خ*** ا*****"


def test_hash_is_salted_and_consistent():
    m1 = Masker("hash", options=MaskOptions(hash_salt="s1"))
    m2 = Masker("hash", options=MaskOptions(hash_salt="s2"))
    h = m1.replacement(F("Obama"))
    assert h.startswith("PERSON_") and len(h) == len("PERSON_") + 8
    assert h == m1.replacement(F("obama"))
    assert h != m2.replacement(F("Obama"))


def test_fake_keeps_language_and_consistency():
    m = Masker("fake")
    ar = m.replacement(F("محمد الأحمد"))
    en = m.replacement(F("Barack Obama"))
    assert any("؀" <= ch <= "ۿ" for ch in ar)
    assert en.isascii() and en != "Barack Obama"
    assert m.replacement(F("Barack Obama")) == en
    # custom label without generator falls back to a numbered pseudonym
    assert m.replacement(F("diabetes", "DISEASE"), "x") == "[DISEASE_1]"
    assert m.replacement(F("test@x.com", "EMAIL")).endswith("@example.com")


def test_remove_and_per_label_override():
    m = Masker("tag", per_label={"EMAIL": "remove", "PERSON": "partial"})
    assert m.replacement(F("a@b.co", "EMAIL")) == ""
    assert m.replacement(F("Bob")) == "B**"
    assert m.replacement(F("Google", "ORGANIZATION")) == "[ORGANIZATION]"


def test_assign_numbers_in_reading_order():
    fs = [F("Zed", seg=2), F("Amy", seg=0), F("Zed", seg=1)]
    Masker("pseudonym", options=MaskOptions(tag_language="en")).assign(fs)
    assert [f.replacement for f in fs] == ["[PERSON_2]", "[PERSON_1]", "[PERSON_2]"]


def test_unknown_method():
    with pytest.raises(ValueError):
        Masker("rot13")
    assert set(METHODS) == {"tag", "pseudonym", "char_mask", "partial", "hash", "fake", "remove"}


def test_partial_names_share_the_full_name_pseudonym():
    fs = [F("Barack Obama", seg=0), F("Obama", seg=1), F("Ahmed Hassan", seg=2), F("Ahmed Ali", seg=3),
          F("Ahmed", seg=4), F("خالد النجار", seg=5), F("النجار", seg=6)]
    Masker("pseudonym", options=MaskOptions(tag_language="en")).assign(fs)
    r = [f.replacement for f in fs]
    assert r[0] == r[1] == "[PERSON_1]"
    assert r[4] not in (r[2], r[3])  # 'Ahmed' is ambiguous -> own number
    assert r[5] == r[6]
    off = Masker("pseudonym", options=MaskOptions(tag_language="en", link_partial_names=False)).assign(
        [F("Barack Obama"), F("Obama", seg=1)])
    assert off[0].replacement != off[1].replacement
