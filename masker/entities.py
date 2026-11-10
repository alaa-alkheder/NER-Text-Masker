"""Entity (finding) data structure and overlap resolution."""
from __future__ import annotations

from dataclasses import dataclass, asdict

# Higher wins when two spans overlap. Patterns are exact, manual terms are the user's
# explicit wish, supervised models beat zero-shot ones (notebook: dslim > NuNER on PER/ORG/LOC).
SOURCE_PRIORITY = {"manual": 5, "regex": 4, "hf": 3, "zero": 2, "propagated": 1}


@dataclass
class Finding:
    segment: int  # index of the paragraph/line in the document
    start: int  # char offsets inside the segment text
    end: int
    label: str
    text: str
    score: float = 1.0
    source: str = "zero"  # manual | regex | hf | zero | propagated
    model: str = ""
    enabled: bool = True
    replacement: str | None = None  # filled by Masker.assign()

    def overlaps(self, other: "Finding") -> bool:
        return self.segment == other.segment and self.start < other.end and other.start < self.end

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_overlaps(findings: list[Finding]) -> list[Finding]:
    """Keep a non-overlapping subset: priority of the source, then longer span, then score."""
    ranked = sorted(
        findings,
        key=lambda f: (SOURCE_PRIORITY.get(f.source, 0), f.end - f.start, f.score),
        reverse=True,
    )
    kept: list[Finding] = []
    by_seg: dict[int, list[Finding]] = {}
    for f in ranked:
        if f.end <= f.start:
            continue
        if any(f.overlaps(k) for k in by_seg.get(f.segment, [])):
            continue
        kept.append(f)
        by_seg.setdefault(f.segment, []).append(f)
    return sorted(kept, key=lambda f: (f.segment, f.start))


def merge_adjacent(entities: list[dict], text: str) -> list[dict]:
    """Merge neighbouring fragments with the same label ('New' + 'York' -> 'New York').
    Needed for token-level GLiNER models such as NuNER_Zero (notebook section 4)."""
    if not entities:
        return []
    ents = sorted((dict(e) for e in entities), key=lambda e: e["start"])
    merged = [ents[0]]
    for e in ents[1:]:
        cur = merged[-1]
        gap = text[cur["end"]:e["start"]]
        if e["label"] == cur["label"] and len(gap) <= 1 and gap.strip() == "":
            cur["end"] = e["end"]
            cur["text"] = text[cur["start"]:cur["end"]]
            cur["score"] = min(cur["score"], e["score"])
        else:
            merged.append(e)
    return merged
