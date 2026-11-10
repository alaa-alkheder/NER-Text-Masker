"""Wrappers around the NER models chosen in the notebook.

* ``HFTokenDetector``  - supervised token-classification models (dslim/bert-base-NER,
  CAMeLBERT-NER) through the ``transformers`` pipeline with word-level aggregation.
* ``GLiNERDetector``   - zero-shot GLiNER models (numind/NuNER_Zero, urchade/gliner_multi-v2.1)
  that take the label names as the prompt.

Both return, for every input text, a list of dicts ``{start, end, label, score, text}``
with character offsets. Heavy libraries are imported lazily so the rest of the project
(regex masking, tests, UI) works without torch installed.
"""
from __future__ import annotations

import logging

from ..config import HF_STRIDE
from ..entities import merge_adjacent

log = logging.getLogger(__name__)


def models_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import gliner  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the environment
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def resolve_device(device: str = "auto") -> str:
    """'auto' -> cuda if present, else cpu. Apple 'mps' only when asked explicitly
    (some DeBERTa ops used by GLiNER are unreliable on MPS)."""
    if device and device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


class HFTokenDetector:
    kind = "hf"

    def __init__(self, model_id: str, aggregation: str = "max", device: str = "auto",
                 batch_size: int = 16, pipe=None, stride: int = HF_STRIDE):
        self.model_id = model_id
        self.aggregation = aggregation
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.stride = stride
        self._stride_ok = stride > 0
        self._pipe = pipe  # injectable for tests

    def load(self):
        if self._pipe is None:
            from transformers import pipeline
            log.info("loading %s on %s", self.model_id, self.device)
            self._pipe = pipeline("token-classification", model=self.model_id,
                                  aggregation_strategy=self.aggregation, device=self.device)
            # A checkpoint without model_max_length gets a huge default: the pipeline would never
            # split a long input and BERT would see only its first 512 tokens.
            tok = getattr(self._pipe, "tokenizer", None)
            cfg = getattr(getattr(self._pipe, "model", None), "config", None)
            if tok is not None and cfg is not None and getattr(tok, "model_max_length", 0) > 100_000:
                tok.model_max_length = getattr(cfg, "max_position_embeddings", 512)
        return self._pipe

    def _call(self, pipe, texts: list[str]):
        """Run with overlapping windows (stride) so text past token 512 is read, not dropped.
        Pipelines or tokenizers that do not support it fall back to a plain call."""
        if self._stride_ok:
            try:
                return pipe(texts, batch_size=self.batch_size, stride=self.stride)
            except (TypeError, ValueError) as exc:
                log.info("stride not supported by %s (%s); calling without it", self.model_id, exc)
                self._stride_ok = False
        return pipe(texts, batch_size=self.batch_size)

    def predict(self, texts: list[str]) -> list[list[dict]]:
        if not texts:
            return []
        pipe = self.load()
        raw = self._call(pipe, texts)
        if texts and raw and isinstance(raw[0], dict):  # single text -> flat list
            raw = [raw]
        out = []
        for text, ents in zip(texts, raw):
            cur = []
            for e in ents:
                tag = e.get("entity_group") or e.get("entity", "")
                tag = tag.split("-", 1)[-1] if tag[:2] in ("B-", "I-") else tag
                s, t = int(e["start"]), int(e["end"])
                cur.append({"start": s, "end": t, "label": tag.upper(),
                            "score": float(e["score"]), "text": text[s:t]})
            out.append(cur)
        return out


class GLiNERDetector:
    kind = "zero"

    def __init__(self, model_id: str, merge: bool = False, lowercase_labels: bool = False,
                 device: str = "auto", batch_size: int = 8, model=None):
        self.model_id = model_id
        self.merge = merge
        self.lowercase_labels = lowercase_labels
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = model  # injectable for tests

    def load(self):
        if self._model is None:
            from gliner import GLiNER
            log.info("loading %s on %s", self.model_id, self.device)
            m = GLiNER.from_pretrained(self.model_id)
            m = m.to(self.device)
            m.eval()
            self._model = m
        return self._model

    def predict(self, texts: list[str], prompts: list[str], threshold: float = 0.5) -> list[list[dict]]:
        if not texts or not prompts:
            return [[] for _ in texts]
        model = self.load()
        labels = [p.lower() for p in prompts] if self.lowercase_labels else list(prompts)
        labels = list(dict.fromkeys(labels))
        raw: list[list[dict]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            if hasattr(model, "inference"):  # gliner >= 0.2.x
                res = model.inference(chunk, labels, threshold=threshold, batch_size=self.batch_size)
            elif hasattr(model, "batch_predict_entities"):
                res = model.batch_predict_entities(chunk, labels, threshold=threshold)
            else:
                res = [model.predict_entities(t, labels, threshold=threshold) for t in chunk]
            raw.extend(res)
        out = []
        for text, ents in zip(texts, raw):
            ents = [{"start": int(e["start"]), "end": int(e["end"]), "label": str(e["label"]).lower(),
                     "score": float(e.get("score", 1.0)), "text": text[int(e["start"]):int(e["end"])]}
                    for e in ents]
            out.append(merge_adjacent(ents, text) if self.merge else ents)
        return out
