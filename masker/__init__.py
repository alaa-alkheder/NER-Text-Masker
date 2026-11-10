"""NER-based text masking for Arabic and English Word/text files."""
from __future__ import annotations

from .config import CATALOG, DEFAULT_LABELS, MODELS, DetectionSettings, resolve_labels
from .document import Document, masked_text
from .entities import Finding
from .pipeline import MaskingPipeline, ModelProvider
from .strategies import METHODS, Masker, MaskOptions

__all__ = ["CATALOG", "DEFAULT_LABELS", "MODELS", "DetectionSettings", "resolve_labels", "Document",
           "masked_text", "Finding", "MaskingPipeline", "ModelProvider", "METHODS", "Masker", "MaskOptions",
           "mask_file"]
__version__ = "1.2.0"


def mask_file(path: str, output: str, labels: list[str], method: str = "tag", custom_labels=None,
              language: str = "auto", options: MaskOptions | None = None, pipeline: MaskingPipeline | None = None,
              **settings_kw) -> list[Finding]:
    """One-call API: detect -> mask -> write. Returns the findings."""
    doc = Document.load(path)
    settings = DetectionSettings(labels=labels, custom_labels=list(custom_labels or []), language=language,
                                 **settings_kw)
    pipe = pipeline or MaskingPipeline(ModelProvider(settings.device))
    findings = pipe.detect(doc.texts, settings)
    Masker(method, options=options).assign(findings, doc.texts)
    with open(output, "wb") as fh:
        fh.write(doc.render(findings))
    return findings
