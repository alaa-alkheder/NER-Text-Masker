"""Text outside the main body: comments, footnotes (generic part) and text boxes."""
import io
import zipfile

import docx
from lxml import etree

from masker.document import Document
from masker.entities import Finding

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _mask(doc, needle, repl="[X]"):
    fs = []
    for s in doc.segments:
        i = s.text.find(needle)
        if i >= 0:
            fs.append(Finding(s.id, i, i + len(needle), "PERSON", needle, replacement=repl))
    return fs


def _with_footnote(base: bytes) -> bytes:
    """Add word/footnotes.xml + reference to a python-docx document (python-docx cannot)."""
    zin = zipfile.ZipFile(io.BytesIO(base))
    out = io.BytesIO()
    zout = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    foot = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:footnotes xmlns:w="{W}">'
            f'<w:footnote w:id="1"><w:p><w:r><w:t>Source: interview with Laila Haddad.</w:t></w:r></w:p>'
            f'</w:footnote></w:footnotes>')
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "[Content_Types].xml":
            data = data.replace(b"</Types>", b'<Override PartName="/word/footnotes.xml" ContentType="application/'
                                b'vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/></Types>')
        if item.filename == "word/_rels/document.xml.rels":
            data = data.replace(b"</Relationships>", b'<Relationship Id="rIdFoot" Type="http://schemas.openxmlformats.'
                                b'org/officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/></Relationships>')
        zout.writestr(item, data)
    zout.writestr("word/footnotes.xml", foot)
    zout.close()
    return out.getvalue()


def test_comments_and_footnotes_are_masked():
    d = docx.Document()
    p = d.add_paragraph("Report reviewed by the board.")
    d.add_comment(p.runs, text="Ask Laila Haddad before sending", author="Laila Haddad")
    buf = io.BytesIO()
    d.save(buf)
    data = _with_footnote(buf.getvalue())

    doc = Document.load(data, "x.docx")
    locs = {s.location for s in doc.segments if "Laila Haddad" in s.text}
    assert locs == {"comments", "footnotes"}
    out = doc.render(_mask(doc, "Laila Haddad"))
    z = zipfile.ZipFile(io.BytesIO(out))
    for part in ("word/footnotes.xml", "word/comments.xml"):
        assert "Laila Haddad" not in z.read(part).decode("utf-8")  # text and comment author both gone
        texts = "".join(etree.fromstring(z.read(part)).itertext())
        assert "Laila Haddad" not in texts and "[X]" in texts
    docx.Document(io.BytesIO(out))  # still a valid package


def test_text_box_paragraph_is_its_own_segment():
    d = docx.Document()
    p = d.add_paragraph("Outer text ")
    box = etree.fromstring(
        f'<w:r xmlns:w="{W}" xmlns:v="urn:schemas-microsoft-com:vml"><w:pict><v:shape><v:textbox><w:txbxContent>'
        f'<w:p><w:r><w:t>Secret Name inside box</w:t></w:r></w:p></w:txbxContent></v:textbox></v:shape></w:pict></w:r>')
    p._p.append(box)
    p.add_run("continues.")
    buf = io.BytesIO()
    d.save(buf)
    doc = Document.load(buf.getvalue(), "box.docx")
    texts = [s.text for s in doc.segments]
    assert "Outer text continues." in texts and "Secret Name inside box" in texts
    out = doc.render(_mask(doc, "Secret Name"))
    all_text = "".join(t.text or "" for t in docx.Document(io.BytesIO(out)).element.body.iter(f"{{{W}}}t"))
    assert "Secret Name" not in all_text and "[X] inside box" in all_text
