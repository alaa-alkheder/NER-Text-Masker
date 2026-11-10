"""Command-line version of the masker (same engine as the Streamlit app).

Examples
    python cli.py samples/sample_ar.docx -l PERSON ORGANIZATION LOCATION PHONE EMAIL -m pseudonym
    python cli.py report.docx -l PERSON DATE --custom "disease, medication" -m fake -o out.docx
    python cli.py notes.txt -l EMAIL PHONE IBAN -m char_mask --regex-only
    python cli.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from masker import CATALOG, DEFAULT_LABELS, METHODS, DetectionSettings, Document, Masker, MaskOptions
from masker.pipeline import MaskingPipeline, ModelProvider


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mask named entities in .docx / .txt files (Arabic + English).")
    ap.add_argument("input", nargs="?", help="input .docx or .txt")
    ap.add_argument("-o", "--output", help="output path (default: <name>_masked.<ext>)")
    ap.add_argument("-l", "--labels", nargs="+", default=DEFAULT_LABELS, help="labels to mask (see --list)")
    ap.add_argument("--custom", default="", help="comma-separated custom zero-shot labels, e.g. 'disease, مرض'")
    ap.add_argument("-m", "--method", default="tag", choices=list(METHODS))
    ap.add_argument("--lang", default="auto", choices=["auto", "ar", "en"])
    ap.add_argument("--arabic-backend", default="gliner", choices=["gliner", "camelbert"])
    ap.add_argument("--th-en", type=float, default=None, help="NuNER_Zero threshold (default 0.7)")
    ap.add_argument("--th-ar", type=float, default=None, help="gliner_multi threshold (default 0.4)")
    ap.add_argument("--tag-language", default="auto", choices=["auto", "ar", "en"])
    ap.add_argument("--mask-char", default="●")
    ap.add_argument("--salt", default=None, help="salt for the hash method")
    ap.add_argument("--no-propagate", action="store_true", help="do not mask other occurrences of found entities")
    ap.add_argument("--no-name-parts", action="store_true",
                    help="do not mask a first/last name on its own after the full name was found")
    ap.add_argument("--no-caseless-fallback", action="store_true",
                    help="keep all-lowercase English on dslim instead of the zero-shot model")
    ap.add_argument("--keep-titles", action="store_true", help="mask titles (Dr., الدكتور) together with the name")
    ap.add_argument("--no-org-fix", action="store_true",
                    help="do not relabel 'Mayo Clinic' / 'جامعة حلب' from LOCATION to ORGANIZATION")
    ap.add_argument("--keep-stopwords", action="store_true",
                    help="keep prepositions/articles at the edges of model spans ('في دمشق', 'the United Nations')")
    ap.add_argument("--regex-only", action="store_true", help="skip the NER models (patterns only)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--report", help="write a JSON summary (no original values) to this path")
    ap.add_argument("--list", action="store_true", help="list available labels and methods")
    a = ap.parse_args(argv)

    if a.list:
        print("Labels:")
        for k, s in CATALOG.items():
            print(f"  {k:<14} {s.kind:<6} {s.ar} - {s.description}")
        print("\nMethods:")
        for k, m in METHODS.items():
            print(f"  {k:<10} {m['example']:<30} {m['ar']}")
        return 0
    if not a.input:
        ap.error("input file is required")

    src = Path(a.input)
    labels = [l.upper() for l in a.labels]
    if a.regex_only:
        labels = [l for l in labels if CATALOG.get(l) and CATALOG[l].kind == "regex"] + (["DATE"] if "DATE" in labels else [])
    unknown = [l for l in labels if l not in CATALOG]
    if unknown:
        ap.error(f"unknown labels {unknown}; use --custom for free-text labels")

    settings = DetectionSettings(labels=labels, custom_labels=[] if a.regex_only else
                                 [c.strip() for c in a.custom.split(",") if c.strip()],
                                 language=a.lang, arabic_backend=a.arabic_backend, propagate=not a.no_propagate,
                                 device=a.device, propagate_name_parts=not a.no_name_parts,
                                 caseless_fallback=not a.no_caseless_fallback, strip_titles=not a.keep_titles,
                                 fix_org_labels=not a.no_org_fix, strip_stopwords=not a.keep_stopwords)
    if a.th_en is not None:
        settings.thresholds["en_zero"] = a.th_en
    if a.th_ar is not None:
        settings.thresholds["ar_zero"] = a.th_ar

    doc = Document.load(src)
    pipe = MaskingPipeline(ModelProvider(a.device))
    if a.regex_only:
        # DATE is a model label; with --regex-only keep only its regex part
        from masker.detectors import RegexDetector
        from masker.entities import resolve_overlaps
        rx = RegexDetector()
        findings = resolve_overlaps([f for i, t in enumerate(doc.texts) for f in rx.detect(t, labels, i)])
    else:
        findings = pipe.detect(doc.texts, settings,
                               progress=lambda p, m: print(f"  [{p:4.0%}] {m}", file=sys.stderr))
    for w in pipe.warnings:
        print("WARNING:", w, file=sys.stderr)

    opts = MaskOptions(tag_language=a.tag_language, mask_char=a.mask_char)
    if a.salt:
        opts.hash_salt = a.salt
    Masker(a.method, options=opts).assign(findings, doc.texts)
    out = Path(a.output) if a.output else src.with_name(f"{src.stem}_masked{src.suffix}")
    out.write_bytes(doc.render(findings))

    counts = Counter(f.label for f in findings)
    print(f"masked {len(findings)} spans -> {out}")
    for k, v in counts.most_common():
        print(f"  {k:<14} {v}")
    if a.report:
        Path(a.report).write_text(json.dumps({"input": src.name, "output": out.name, "method": a.method,
                                              "labels_masked": dict(counts), "total_masked": len(findings)},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
