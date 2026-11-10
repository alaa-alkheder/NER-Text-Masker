"""Detection pipeline: routes every selected label to one detector per language.

    English chunk:  PERSON / ORGANIZATION / LOCATION  -> dslim/bert-base-NER  (supervised)
                    any other label + custom labels   -> numind/NuNER_Zero    (zero-shot)
    English sentence with no case signal (all lower / all caps): every model label
                                                      -> numind/NuNER_Zero (dslim needs capitals)
    Arabic chunk:   every model label                 -> urchade/gliner_multi-v2.1
                    (or CAMeLBERT for PER/ORG/LOC when arabic_backend="camelbert")
    Always:         EMAIL / PHONE / URL / IBAN / ...  -> regex (+ numeric dates for DATE)
                    manual terms typed by the user

Model spans are refined (titles moved out of PERSON spans, "Mayo Clinic"/"جامعة حلب" moved from
LOCATION to ORGANIZATION), overlapping spans are resolved and (optionally) every detected entity
string - and every single word of a detected person name - is searched in the whole document
so that later mentions the model missed are masked too.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .config import (DISTRACTOR_PROMPTS, MAX_CHUNK_WORDS, MIN_NAME_PART_LEN, MODELS, NAME_PARTICLES, ORG_KEYWORDS_AR,
                     STOPWORDS,
                     ORG_KEYWORDS_EN, TITLES_AR, TITLES_EN, USE_DISTRACTORS, DetectionSettings, LabelSpec)
from .detectors import GLiNERDetector, HFTokenDetector, RegexDetector
from .entities import Finding, resolve_overlaps
from .text_utils import (chunk_text, detect_language, find_occurrences, has_org_keyword, name_parts, split_by_case,
                         strip_stopwords, strip_title, trim_span)

TITLES = TITLES_EN | TITLES_AR
LANES = {"en": "English", "en_nocase": "English بلا أحرف كبيرة", "ar": "عربي"}

log = logging.getLogger(__name__)
Progress = Optional[Callable[[float, str], None]]


class ModelProvider:
    """Creates detectors on first use and keeps them in memory."""

    def __init__(self, device: str = "auto", factory: Callable[[str], object] | None = None):
        self.device = device
        self._factory = factory
        self._cache: dict[str, object] = {}

    def get(self, role: str):
        if role not in self._cache:
            self._cache[role] = (self._factory or self._build)(role)
        return self._cache[role]

    def _build(self, role: str):
        cfg = MODELS[role]
        if cfg["kind"] == "hf":
            return HFTokenDetector(cfg["id"], aggregation=cfg.get("aggregation", "first"), device=self.device)
        return GLiNERDetector(cfg["id"], merge=cfg.get("merge_adjacent", False),
                              lowercase_labels=cfg.get("lowercase_labels", False), device=self.device)


class MaskingPipeline:
    def __init__(self, provider: ModelProvider | None = None):
        self.provider = provider or ModelProvider()
        self.regex = RegexDetector()
        self.warnings: list[str] = []
        self.stats: dict[str, int] = {}

    # ------------------------------------------------------------------ public
    def detect(self, segments: list[str], settings: DetectionSettings, progress: Progress = None) -> list[Finding]:
        self.warnings, self.stats = [], {}
        report = progress or (lambda *_: None)
        specs = settings.specs()
        findings: list[Finding] = []

        # 1) regex labels (+ numeric/month-name dates when DATE is selected)
        regex_labels = [k for k, s in specs.items() if s.kind == "regex"]
        if "DATE" in specs:
            regex_labels.append("DATE")
        if regex_labels:
            for i, text in enumerate(segments):
                findings += self.regex.detect(text, regex_labels, i)

        # 2) manual terms
        for term, label in settings.manual_terms:
            term = term.strip()
            for i, text in enumerate(segments):
                for s, e in find_occurrences(text, term):
                    findings.append(Finding(i, s, e, label, text[s:e], 1.0, "manual", "manual"))

        # 3) model labels, per language
        ner_specs = {k: s for k, s in specs.items() if s.kind == "ner"}
        if ner_specs:
            jobs: dict[str, list[tuple[int, int, int]]] = {lane: [] for lane in LANES}
            for i, text in enumerate(segments):
                if not text.strip():
                    continue
                for cs, ce in chunk_text(text, MAX_CHUNK_WORDS):
                    lang = settings.language if settings.language in ("ar", "en") else detect_language(text[cs:ce])
                    if lang == "en" and settings.caseless_fallback:
                        for s, e, caseless in split_by_case(text, cs, ce):
                            jobs["en_nocase" if caseless else "en"].append((i, s, e))
                    else:
                        jobs[lang].append((i, cs, ce))
            self.stats = {"chunks_en": len(jobs["en"]), "chunks_en_nocase": len(jobs["en_nocase"]),
                          "chunks_ar": len(jobs["ar"])}
            plan = self._plan(ner_specs, settings)
            steps = [(lane, role, labels) for lane in LANES for role, labels in plan[lane] if jobs[lane]]
            for n, (lang, role, labels) in enumerate(steps):
                report(n / max(len(steps), 1), f"{MODELS[role]['id']} ({LANES[lang]})")
                try:
                    findings += self._run(role, labels, jobs[lang], segments, settings, specs)
                except Exception as exc:  # model missing / no internet / OOM ...
                    msg = f"تعذّر تشغيل النموذج {MODELS[role]['id']}: {type(exc).__name__}: {exc}"
                    log.exception(msg)
                    self.warnings.append(msg)

        report(0.95, "دمج النتائج")
        resolved = resolve_overlaps(findings)

        # 4) propagate model-detected strings to all their occurrences
        if settings.propagate:
            seen: dict[str, Finding] = {}
            for f in resolved:
                if f.source in ("hf", "zero") and len(f.text.strip()) >= 2:
                    seen.setdefault(f.text, f)
            if settings.propagate_name_parts:
                for part, f0 in self._name_parts(resolved, segments).items():
                    seen.setdefault(part, f0)
            extra = []
            for i, text in enumerate(segments):
                for t, f0 in seen.items():
                    for s, e in find_occurrences(text, t):  # 'أحمد' also finds 'احمد'
                        extra.append(Finding(i, s, e, f0.label, text[s:e], f0.score, "propagated", f0.model))
            resolved = resolve_overlaps(resolved + extra)
            self.stats["propagated"] = sum(1 for f in resolved if f.source == "propagated")
        report(1.0, "تم")
        return resolved

    @staticmethod
    def _name_parts(resolved: list[Finding], segments: list[str]) -> dict[str, Finding]:
        """Single words of detected person names ('Ahmed' from 'Ahmed Al-Khatib'). In the second
        study NuNER_Zero found the full name but missed 'Ahmed' and 'Sara' alone in later lines.
        A Latin word that also appears in lower case in the document ('Will' / 'will') is a common
        word, and a word already found with another label ('Jordan' / LOCATION) is left alone."""
        other_labels = {f.text for f in resolved if f.label != "PERSON"}
        lower_words: set[str] | None = None
        out: dict[str, Finding] = {}
        for f in resolved:
            if f.source not in ("hf", "zero") or f.label != "PERSON":
                continue
            for part in name_parts(f.text, NAME_PARTICLES, TITLES, MIN_NAME_PART_LEN):
                if part in out or part in other_labels:
                    continue
                if part[:1].isascii() and part[:1].isupper():
                    if lower_words is None:
                        lower_words = {w.strip(".,;:!?\"'()") for t in segments for w in t.split() if w.islower()}
                    if part.lower() in lower_words:
                        continue
                out[part] = f
        return out

    # ------------------------------------------------------------------ routing
    @staticmethod
    def _plan(ner_specs: dict[str, LabelSpec], settings: DetectionSettings) -> dict[str, list[tuple[str, list[str]]]]:
        hf_core = [k for k, s in ner_specs.items() if s.en_backend == "hf" and not s.custom]
        others = [k for k in ner_specs if k not in hf_core]
        plan: dict[str, list[tuple[str, list[str]]]] = {lane: [] for lane in LANES}
        # no capital letters to rely on -> the zero-shot model takes every label
        plan["en_nocase"].append(("en_zero", list(ner_specs)))
        if hf_core:
            plan["en"].append(("en_ner", hf_core))
        if others:
            plan["en"].append(("en_zero", others))
        if settings.arabic_backend == "camelbert":
            if hf_core:
                plan["ar"].append(("ar_ner", hf_core))
            if others:
                plan["ar"].append(("ar_zero", others))
        else:
            plan["ar"].append(("ar_zero", list(ner_specs)))
        return plan

    def _run(self, role: str, labels: list[str], jobs: list[tuple[int, int, int]], segments: list[str],
             settings: DetectionSettings, specs: dict[str, LabelSpec]) -> list[Finding]:
        detector = self.provider.get(role)
        texts = [segments[i][cs:ce] for i, cs, ce in jobs]
        lang = "ar" if role.startswith("ar") else "en"
        out: list[Finding] = []

        if detector.kind == "hf":
            tag2label = {t: k for k in labels for t in specs[k].hf_tags}
            preds = detector.predict(texts)
            source = "hf"
            threshold = settings.min_hf_score
        else:
            prompt2label: dict[str, str] = {}
            for k in labels:
                for p in (specs[k].prompts_ar if lang == "ar" else specs[k].prompts_en):
                    prompt2label.setdefault(p.lower(), k)
            prompts = list(prompt2label)
            if USE_DISTRACTORS:
                prompts += [p for p in DISTRACTOR_PROMPTS if p not in prompt2label]
                # labels routed to the supervised model act as distractors here
                for k, s in specs.items():
                    if s.kind == "ner" and k not in labels:
                        prompts += [p for p in s.prompts_en if p not in prompts]
            tag2label = prompt2label
            threshold = settings.thresholds.get(role, MODELS[role].get("threshold", 0.5))
            preds = detector.predict(texts, prompts, threshold=threshold)
            source = "zero"
            threshold = 0.0  # already applied by the model

        for (seg, cs, _), ents in zip(jobs, preds):
            for e in ents:
                raw = str(e["label"])
                label = tag2label.get(raw.upper()) if detector.kind == "hf" else tag2label.get(raw.lower())
                if label is None or e["score"] < threshold:
                    continue
                s, t = trim_span(segments[seg], cs + e["start"], cs + e["end"])
                if settings.strip_stopwords:
                    s, t = strip_stopwords(segments[seg], s, t, STOPWORDS)
                if t <= s:
                    continue
                if label == "PERSON" and settings.strip_titles:
                    s, t = strip_title(segments[seg], s, t, TITLES)
                if (label == "LOCATION" and settings.fix_org_labels and "ORGANIZATION" in specs
                        and has_org_keyword(segments[seg][s:t], ORG_KEYWORDS_EN, ORG_KEYWORDS_AR)):
                    label = "ORGANIZATION"
                out.append(Finding(seg, s, t, label, segments[seg][s:t], round(float(e["score"]), 4),
                                   source, MODELS[role]["id"]))
        return out
