import io

import docx

from masker.document import Document, masked_text
from masker.entities import Finding


def _find(doc, seg_text_part, needle, label="PERSON", repl="[X]"):
    seg = next(s for s in doc.segments if seg_text_part in s.text)
    i = seg.text.index(needle)
    return Finding(seg.id, i, i + len(needle), label, needle, replacement=repl)


def test_reads_body_tables_headers_hyperlinks(samples):
    d = Document.load(samples / "sample_en.docx")
    locs = {s.location for s in d.segments if s.text.strip()}
    assert {"body", "table", "header", "footer"} <= locs
    assert any("https://portal.acme.com/login" in s.text for s in d.segments)  # inside <w:hyperlink>


def test_replacement_across_runs_keeps_formatting(samples):
    d = Document.load(samples / "sample_en.docx")
    f = _find(d, "met officials", "Barack Obama", repl="[PERSON]")
    out = docx.Document(io.BytesIO(d.render([f])))
    p = next(p for p in out.paragraphs if "met officials" in p.text)
    assert p.text.startswith("On 12 March 2024, [PERSON] met officials")
    runs = [(r.text, r.bold, r.italic) for r in p.runs]
    assert ("[PERSON]", True, None) in runs  # replacement lives in the first (bold) run
    assert ("", None, True) in runs  # second run emptied, formatting untouched


def test_header_footer_table_hyperlink_and_metadata(samples):
    d = Document.load(samples / "sample_en.docx")
    fs = [_find(d, "Confidential", "Sarah Johnson"), _find(d, "Emily Clark", "Emily Clark"),
          _find(d, "Portal", "https://portal.acme.com/login", "URL", "[URL]")]
    out = docx.Document(io.BytesIO(d.render(fs)))
    assert out.sections[0].header.paragraphs[0].text == "Confidential - prepared by [X]"
    assert out.tables[0].rows[1].cells[0].text == "[X]"
    assert "Portal: [URL]" in "\n".join(p.text for p in out.paragraphs) or \
        any("[URL]" in t.text for t in out.element.body.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
    assert out.core_properties.author == ""
    # original document object is not mutated -> can be rendered again with other choices
    assert "Sarah Johnson" in d.segments[-1].text or any("Sarah Johnson" in s.text for s in d.segments)
    again = docx.Document(io.BytesIO(d.render([])))
    assert "Sarah Johnson" in again.sections[0].header.paragraphs[0].text


def test_disabled_findings_are_not_applied(samples):
    d = Document.load(samples / "sample_en.docx")
    f = _find(d, "met officials", "Barack Obama")
    f.enabled = False
    out = docx.Document(io.BytesIO(d.render([f])))
    assert any("Barack Obama" in p.text for p in out.paragraphs)


def test_txt_roundtrip_and_encoding():
    raw = "سطر أول\r\ncontact a@b.co now\r\n".encode("cp1256", errors="ignore") if False else \
        "سطر أول\r\ncontact a@b.co now\r\n".encode("utf-8")
    d = Document.load(raw, "x.txt")
    seg = d.segments[1]
    f = Finding(1, seg.text.index("a@b.co"), seg.text.index("a@b.co") + 6, "EMAIL", "a@b.co", replacement="")
    assert d.render([f]).decode("utf-8") == "سطر أول\r\ncontact now\r\n"


def test_cp1256_text_file():
    d = Document.load("مرحبا يا أحمد".encode("cp1256"), "old.txt")
    assert d.segments[0].text == "مرحبا يا أحمد"


def test_masked_text_multiple_spans():
    t = "Bob met Ann in Rome"
    fs = [Finding(0, 0, 3, "PERSON", "Bob", replacement="[P1]"), Finding(0, 8, 11, "PERSON", "Ann", replacement="[P2]"),
          Finding(0, 15, 19, "LOCATION", "Rome", replacement="[L]")]
    assert masked_text(t, fs) == "[P1] met [P2] in [L]"
