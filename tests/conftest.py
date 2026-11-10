"""Test fixtures: gazetteer-based fake models so the pipeline can be tested without
downloading the real NER checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from masker.pipeline import MaskingPipeline, ModelProvider  # noqa: E402

EN_GAZ = {"Barack Obama": "PER", "Obama": "PER", "Ahmed Hassan": "PER", "Emily Clark": "PER",
          "Omar Khalid": "PER", "Sarah Johnson": "PER", "United Nations": "ORG", "Microsoft": "ORG",
          "Google": "ORG", "Siemens": "ORG", "Acme Corporation": "ORG", "Baghdad": "LOC", "Amman": "LOC",
          "London": "LOC", "Berlin": "LOC", "Iraqi": "MISC"}
ZERO_GAZ = {"Iraqi": "nationality", "Monday": "date", "12 March 2024": "date", "senior engineer": "occupation",
            "محمد الأحمد": "person", "خالد النجار": "person", "ليلى حسن": "person", "أحمد العلي": "person",
            "سارة يوسف": "person", "سامر الخطيب": "person", "الأمم المتحدة": "organization",
            "سيرياتيل": "organization", "جامعة دمشق": "organization", "بنك سورية الدولي": "organization",
            "منظمة الصحة العالمية": "organization", "وزارة الخارجية": "organization", "دمشق": "location",
            "حلب": "location", "بيروت": "location", "الرياض": "location", "سعودي": "nationality",
            "diabetes": "disease", "metformin": "medication"}


def _scan(text: str, gaz: dict[str, str]) -> list[dict]:
    out = []
    for term, lab in gaz.items():
        i = text.find(term)
        while i != -1:
            out.append({"start": i, "end": i + len(term), "label": lab, "score": 0.9, "text": term})
            i = text.find(term, i + 1)
    # keep longest non-overlapping
    out.sort(key=lambda e: e["end"] - e["start"], reverse=True)
    kept = []
    for e in out:
        if all(e["end"] <= k["start"] or e["start"] >= k["end"] for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e["start"])


class FakeHF:
    kind = "hf"

    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(list(texts))
        return [_scan(t, EN_GAZ) for t in texts]


class FakeZero:
    kind = "zero"

    def __init__(self):
        self.calls = []

    def predict(self, texts, prompts, threshold=0.5):
        self.calls.append((list(texts), list(prompts), threshold))
        return [[e for e in _scan(t, ZERO_GAZ) if e["label"] in prompts] for t in texts]


@pytest.fixture
def fakes():
    return {"en_ner": FakeHF(), "ar_ner": FakeHF(), "en_zero": FakeZero(), "ar_zero": FakeZero()}


@pytest.fixture
def pipeline(fakes):
    return MaskingPipeline(ModelProvider(factory=lambda role: fakes[role]))


@pytest.fixture
def samples():
    return ROOT / "samples"
