"""Reading .docx / .txt files and writing the masked copy.

For Word files the text of each paragraph is rebuilt from its <w:t> nodes (runs,
hyperlinks, tracked insertions, text boxes...). Every character keeps a pointer to the node
it came from, so a replacement is written back *into the original runs*: bold, colours,
fonts, tables, headers, footers, footnotes and comments stay exactly as they were.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from .entities import Finding

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
_W = "{%s}" % W_NS
W_P, W_T, W_TAB, W_BR, W_CR = _W + "p", _W + "t", _W + "tab", _W + "br", _W + "cr"
_SKIP = {_W + "delText", _W + "instrText", _W + "delInstrText"}

# parts of a .docx that hold user text
_TEXT_PARTS = re.compile(r"^/word/(document|header\d*|footer\d*|footnotes|endnotes|comments)\.xml$")


@dataclass
class Segment:
    id: int
    text: str
    location: str = "body"
    pieces: list = field(default_factory=list, repr=False)  # [(start, end, <w:t> node)]


def _part_label(partname: str) -> str:
    name = partname.rsplit("/", 1)[-1].split(".")[0]
    return re.sub(r"\d+$", "", name).replace("document", "body")


class Document:
    """Uniform view over .docx and .txt: a list of text segments + a writer."""

    def __init__(self, kind: str, segments: list[Segment], filename: str, *, docx=None, roots=None,
                 raw_parts=None, newline: str = "\n"):
        self.kind = kind
        self.segments = segments
        self.filename = filename
        self._docx = docx
        self._roots = roots or []
        self._raw_parts = raw_parts or []  # generic parts whose blob we re-serialise
        self._newline = newline

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, source, filename: str | None = None) -> "Document":
        if isinstance(source, (str, Path)):
            filename = filename or Path(source).name
            data = Path(source).read_bytes()
        else:
            data = source if isinstance(source, bytes) else source.read()
        filename = filename or "document.txt"
        ext = Path(filename).suffix.lower()
        if ext == ".docx":
            return cls._load_docx(data, filename)
        if ext in (".txt", ".md", ".csv", ""):
            return cls._load_txt(data, filename)
        raise ValueError(f"نوع الملف غير مدعوم: {ext} (المدعوم: .docx و .txt)")

    @classmethod
    def _load_txt(cls, data: bytes, filename: str) -> "Document":
        for enc in ("utf-8-sig", "cp1256", "latin-1"):  # cp1256 = Arabic Windows
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.replace("\r\n", "\n").split("\n")
        segs = [Segment(i, line, "line") for i, line in enumerate(lines)]
        return cls("txt", segs, filename, newline=newline)

    @classmethod
    def _load_docx(cls, data: bytes, filename: str) -> "Document":
        import docx
        from lxml import etree

        d = docx.Document(io.BytesIO(data))
        roots, raw_parts = [], []
        for part in d.part.package.iter_parts():
            pname = str(part.partname)
            if not _TEXT_PARTS.match(pname):
                continue
            if hasattr(part, "_element"):
                roots.append((pname, part._element))
            else:  # e.g. footnotes in python-docx: generic Part holding a blob
                root = etree.fromstring(part.blob)
                roots.append((pname, root))
                raw_parts.append((part, root))
        # main document first, then headers/footers/notes
        roots.sort(key=lambda r: (r[0] != "/word/document.xml", r[0]))
        segs: list[Segment] = []
        for pname, root in roots:
            loc = _part_label(pname)
            for p in root.iter(W_P):
                text, pieces = cls._paragraph_text(p)
                segs.append(Segment(len(segs), text, "table" if loc == "body" and _in_table(p) else loc, pieces))
        return cls("docx", segs, filename, docx=d, roots=roots, raw_parts=raw_parts)

    @staticmethod
    def _paragraph_text(p) -> tuple[str, list]:
        """Concatenate <w:t> text of this paragraph only (nested paragraphs of text boxes
        are separate segments). Tabs/breaks become virtual '\\t' / '\\n' characters."""
        parts: list[str] = []
        pieces: list = []
        pos = 0

        def walk(el):
            nonlocal pos
            for child in el:
                tag = child.tag
                if not isinstance(tag, str) or tag in _SKIP:
                    continue
                if tag == W_P:
                    continue
                if tag == W_T:
                    t = child.text or ""
                    pieces.append((pos, pos + len(t), child))
                    parts.append(t)
                    pos += len(t)
                elif tag == W_TAB:
                    parts.append("\t")
                    pos += 1
                elif tag in (W_BR, W_CR):
                    parts.append("\n")
                    pos += 1
                else:
                    walk(child)

        walk(p)
        return "".join(parts), pieces

    # ------------------------------------------------------------------ info
    @property
    def texts(self) -> list[str]:
        return [s.text for s in self.segments]

    def stats(self) -> dict:
        non_empty = [s for s in self.segments if s.text.strip()]
        return {"segments": len(non_empty), "words": sum(len(s.text.split()) for s in non_empty),
                "chars": sum(len(s.text) for s in non_empty)}

    # ------------------------------------------------------------------ writing
    def render(self, findings: list[Finding], scrub_metadata: bool = True) -> bytes:
        """Return the masked file as bytes (uses ``finding.replacement`` of enabled findings)."""
        by_seg: dict[int, list[Finding]] = {}
        for f in findings:
            if f.enabled:
                by_seg.setdefault(f.segment, []).append(f)

        if self.kind == "txt":
            lines = [masked_text(s.text, by_seg.get(s.id, [])) for s in self.segments]
            return self._newline.join(lines).encode("utf-8")

        from lxml import etree

        # we mutate the loaded tree in place -> work on a fresh copy each time
        clone = Document._load_docx(self._to_bytes(), self.filename)
        for seg in clone.segments:
            spans = _spans_for_render(seg.text, by_seg.get(seg.id, []))
            for start, end, repl in sorted(spans, key=lambda x: x[0], reverse=True):
                _replace_in_pieces(seg.pieces, start, end, repl)
        if scrub_metadata:  # authors of comments / tracked changes are names too
            for _, root in clone._roots:
                for el in root.iter():
                    for attr in (_W + "author", _W + "initials"):
                        if el.get(attr):
                            el.set(attr, "")
        for part, root in clone._raw_parts:
            part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        if scrub_metadata:
            cp = clone._docx.core_properties
            for attr in ("author", "last_modified_by", "comments", "keywords", "subject", "title"):
                try:
                    setattr(cp, attr, "")
                except Exception:  # pragma: no cover
                    pass
        return clone._to_bytes()

    def _to_bytes(self) -> bytes:
        buf = io.BytesIO()
        self._docx.save(buf)
        return buf.getvalue()


def _in_table(p) -> bool:
    el = p.getparent()
    while el is not None:
        if el.tag == _W + "tc":
            return True
        el = el.getparent()
    return False


def _spans_for_render(text: str, findings: list[Finding]) -> list[tuple[int, int, str]]:
    out = []
    for f in findings:
        repl = f.replacement or ""
        s, e = f.start, f.end
        if repl == "":  # 'remove' -> also swallow one neighbouring space
            if s > 0 and text[s - 1] == " " and (e == len(text) or text[e] in " .,،;:!?؟)"):
                s -= 1
            elif e < len(text) and text[e] == " " and s == 0:
                e += 1
        out.append((s, e, repl))
    return out


def masked_text(text: str, findings: list[Finding]) -> str:
    spans = _spans_for_render(text, [f for f in findings if f.enabled])
    for s, e, repl in sorted(spans, key=lambda x: x[0], reverse=True):
        text = text[:s] + repl + text[e:]
    return text


def _replace_in_pieces(pieces: list, start: int, end: int, repl: str) -> None:
    first = True
    for ps, pe, node in pieces:
        if pe <= start or ps >= end or ps == pe:
            continue
        t = node.text or ""
        ls, le = max(start, ps) - ps, min(end, pe) - ps
        if first:
            node.text = t[:ls] + repl + t[le:]
            first = False
        else:
            node.text = t[:ls] + t[le:]
        node.set("{%s}space" % XML_NS, "preserve")
