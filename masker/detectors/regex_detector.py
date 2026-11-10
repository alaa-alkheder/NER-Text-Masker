"""Deterministic detectors for structured PII (e-mail, phone, IBAN, ...).

Patterns are exact and cheap, so they run on the full paragraph (no chunking) and take
priority over model predictions when spans overlap. ``\\d`` also matches Arabic-Indic
digits (٠-٩), so '+٩٦٣ ٩٣٣ ١٢٣ ٤٥٦' is found too.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

from ..entities import Finding, resolve_overlaps


def _digits(s: str) -> str:
    return "".join(str(int(c)) for c in s if c.isdecimal())


def _luhn(s: str) -> bool:
    d = _digits(s)
    if not 13 <= len(d) <= 19 or len(set(d)) == 1:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _iban(s: str) -> bool:
    c = s.replace(" ", "").upper()
    if not 15 <= len(c) <= 34:
        return False
    moved = c[4:] + c[:4]
    num = "".join(str(int(ch, 36)) for ch in moved)
    return int(num) % 97 == 1


def _phone(s: str) -> bool:
    return 7 <= len(_digits(s)) <= 15


def _ipv4(s: str) -> bool:
    parts = s.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 for p in parts)


_EN_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
             r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
_AR_MONTH = ("(?:يناير|كانون الثاني|فبراير|شباط|مارس|آذار|اذار|أبريل|ابريل|إبريل|نيسان|مايو|أيار|ايار|"
             "يونيو|يونيه|حزيران|يوليو|يوليه|تموز|أغسطس|اغسطس|آب|سبتمبر|أيلول|ايلول|أكتوبر|اكتوبر|"
             "تشرين الأول|تشرين الاول|نوفمبر|تشرين الثاني|ديسمبر|كانون الأول|كانون الاول|محرم|صفر|"
             "ربيع الأول|ربيع الاول|ربيع الآخر|ربيع الثاني|جمادى الأولى|جمادى الاولى|جمادى الآخرة|"
             "جمادى الاخرة|رجب|شعبان|رمضان|شوال|ذو القعدة|ذي القعدة|ذو الحجة|ذي الحجة)")
_ORD = r"(?:st|nd|rd|th)?"

PATTERNS: dict[str, list[tuple[re.Pattern, Callable[[str], bool] | None]]] = {
    "EMAIL": [(re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])"), None)],
    "URL": [(re.compile(r"(?:https?://|www\.)[^\s<>\"'،؛]+", re.I), None)],
    "IP_ADDRESS": [(re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])(?!\.[0-9])"), _ipv4)],
    "CREDIT_CARD": [(re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"), _luhn)],
    "IBAN": [(re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"), _iban)],
    "PHONE": [
        # international: +963 933 123 456, 00971-4-1234567, +1 (212) 555-0100
        (re.compile(r"(?<![\w+])(?:\+|00|٠٠)[1-9١-٩۱-۹]\d{0,2}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?(?:\d{1,4}[\s.-]?){1,4}\d{2,4}(?!\w)"), _phone),
        # local with trunk 0: 0933 123 456, 011-2345678, (011) 234 5678
        (re.compile(r"(?<![\w+])(?:\([0٠۰]\d{1,3}\)|[0٠۰]\d{1,3})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)"), _phone),
    ],
    "DATE": [
        (re.compile(r"(?<!\d)(?:\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})(?!\d)"), None),
        (re.compile(rf"\b\d{{1,2}}{_ORD}\s+(?:of\s+)?{_EN_MONTH}(?:,?\s+\d{{4}})?\b"), None),
        (re.compile(rf"\b{_EN_MONTH}\s+\d{{1,2}}{_ORD}(?:,?\s+\d{{4}})?\b"), None),
        (re.compile(rf"\b{_EN_MONTH}\s+\d{{4}}\b"), None),
        (re.compile(rf"(?<!\w)(?:\d{{1,2}}\s+(?:من\s+)?)?{_AR_MONTH}(?:\s+(?:عام|سنة|لعام|لسنة)?\s*\d{{4}})?(?:\s*(?:م|هـ|ه)(?!\w))?"), None),
    ],
    "ID_NUMBER": [(re.compile(r"(?<![\d+])\d{9,18}(?!\d)"), None)],
}

_URL_TRAIL = ".,;:!?)]}>'\"،؛"

# Words just before a number that tell which kind it is. A 10-11 digit number can be a phone
# or an ID; the label whose keyword appears in the preceding 30 characters wins the overlap.
CONTEXT = {
    "ID_NUMBER": ("هوية", "الهوية", "وطني", "الوطني", "جواز", "قومي", "سجل مدني", "passport", "national id",
                  "id no", "id number", "identity", "ssn"),
    "PHONE": ("هاتف", "الهاتف", "جوال", "موبايل", "خلوي", "واتساب", "phone", "tel", "mobile", "call", "whatsapp",
              "fax", "فاكس"),
    "CREDIT_CARD": ("card", "بطاقة", "visa", "mastercard"),
    "IBAN": ("iban", "حساب", "account"),
}


class RegexDetector:
    name = "regex"

    def labels(self) -> list[str]:
        return list(PATTERNS)

    def detect(self, text: str, labels: Iterable[str], segment: int = 0) -> list[Finding]:
        out: list[Finding] = []
        for label in labels:
            for pattern, validate in PATTERNS.get(label, []):
                for m in pattern.finditer(text):
                    s, e = m.start(), m.end()
                    if label == "URL":
                        while e > s and text[e - 1] in _URL_TRAIL:
                            e -= 1
                    frag = text[s:e]
                    if label == "DATE" and not any(ch.isdigit() for ch in frag):
                        continue  # an Arabic month name alone ('رمضان', 'صفر') is not a date
                    if validate is not None and not validate(frag):
                        continue
                    before = text[max(0, s - 30):s].lower()
                    dists = [len(before) - before.rfind(k) - len(k) for k in CONTEXT.get(label, ()) if k in before]
                    score = round(1.0 - 0.001 * min(dists), 3) if dists else 0.95  # nearest keyword wins
                    out.append(Finding(segment, s, e, label, frag, score, "regex", "regex"))
        return resolve_overlaps(out)  # several patterns may hit the same text -> keep the longest
