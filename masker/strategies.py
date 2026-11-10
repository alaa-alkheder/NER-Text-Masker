
from __future__ import annotations

import hashlib
import random
import secrets
from dataclasses import dataclass, field

from .config import CATALOG, TAG_STYLES
from .entities import Finding
from .text_utils import is_arabic, normalize_key

METHODS: dict[str, dict[str, str]] = {
    "tag": {"ar": "استبدال بنوع الـ label", "example": "[PERSON]",
            "help": "يُخفي القيمة ويُبقي نوعها ظاهراً."},
    "pseudonym": {"ar": "اسم مستعار مرقّم", "example": "[PERSON_1]",
                  "help": "نفس الكيان يأخذ نفس الرقم في كل الملف، فيبقى النص مفهوماً."},
    "char_mask": {"ar": "إخفاء بالرموز", "example": "●●●●●●",
                  "help": "يستبدل كل حرف برمز مع الحفاظ على الطول."},
    "partial": {"ar": "إخفاء جزئي", "example": "M***** A***",
                "help": "يُبقي أول حرف من كل كلمة ويخفي الباقي. الأرقام تُبقي آخر رقمين فقط، "
                        "والبريد أول حرف والامتداد (.com)."},
    "hash": {"ar": "بصمة Hash (SHA-256)", "example": "PERSON_3fa9c2e1",
             "help": "بصمة مملّحة أحادية الاتجاه؛ متّسقة داخل الملف ولا يمكن عكسها."},
    "fake": {"ar": "بيانات وهمية", "example": "Ahmad Salem → Khaled Nasser",
             "help": "يستبدل الكيان بقيمة وهمية واقعية من نفس النوع وبنفس اللغة."},
    "remove": {"ar": "حذف كامل", "example": "(يُحذف)",
               "help": "يحذف النص تماماً من المستند."},
}


@dataclass
class MaskOptions:
    tag_language: str = "auto"  # auto | en | ar  (auto -> Arabic tag inside Arabic text)
    tag_style: str = "square"
    mask_char: str = "●"
    keep_first: int = 1  # partial: letters kept at the start of each word
    hash_salt: str = field(default_factory=lambda: secrets.token_hex(16))
    hash_length: int = 8
    seed: int = 13  # fake data reproducibility
    link_partial_names: bool = True  # "Obama" gets the same pseudonym as "Barack Obama"
    keep_last_digits: int = 2  # partial: digits left visible at the end of a number


# ----------------------------------------------------------------------------- partial
_NUMERIC_LABELS = {"PHONE", "CREDIT_CARD", "IBAN", "ID_NUMBER", "IP_ADDRESS"}


def partial_mask(text: str, label: str, keep_first: int = 1, keep_last_digits: int = 2) -> str:
    """Partial masking that does not leak structure. Words keep their first letter(s); numbers
    keep only their last digits, because 'first letter of every word' on '+963 944 123 456'
    left the first digit of every group visible (second study, section 9)."""
    if label == "EMAIL" and "@" in text:
        local, _, domain = text.partition("@")
        host, dot, tld = domain.rpartition(".")
        if not dot:
            host, tld = domain, ""
        hide = lambda s: s[:1] + "*" * max(len(s) - 1, 1)
        return f"{hide(local)}@{hide(host)}{dot}{tld}"
    if label in _NUMERIC_LABELS:
        alnum = [i for i, ch in enumerate(text) if ch.isalnum()]
        visible = set(alnum[-keep_last_digits:]) if keep_last_digits > 0 else set()
        return "".join(ch if (not ch.isalnum() or i in visible) else "*" for i, ch in enumerate(text))
    k = max(0, keep_first)
    return " ".join(w[:k] + "*" * max(len(w) - k, 1) if len(w) > k else "*" * len(w) for w in text.split())


# ----------------------------------------------------------------------------- fake data
_AR_FIRST = ["أحمد", "خالد", "سامر", "يوسف", "عمر", "كريم", "رامي", "باسل", "طارق", "مازن", "هادي", "زياد",
             "ليلى", "سارة", "ريم", "هبة", "دانة", "نور", "لينا", "رنا", "سلمى", "مريم", "جود", "تالا"]
_AR_LAST = ["الحسن", "النجار", "العلي", "الخطيب", "الشامي", "الحلبي", "السيد", "المصري", "الأحمد", "الزين",
            "العمر", "الحداد", "الصباغ", "القاسم", "الرفاعي", "الجابر"]
_AR_ORG = ["شركة النور للتجارة", "مؤسسة الأفق", "مجموعة الريادة", "جامعة المستقبل", "شركة الواحة للتقنية",
           "مركز الإبداع", "مصرف الأمان", "مؤسسة الغد الخيرية", "شركة الجسر للخدمات", "مستشفى الرحمة"]
_AR_LOC = ["مدينة الواحة", "بلدة السنديان", "حي الياسمين", "منطقة الضياء", "مدينة الزيتون", "قرية النخيل",
           "حي الورود", "مدينة الربيع"]
_AR_NAT = ["مواطن أ", "مواطن ب", "مقيم أ", "مقيم ب"]
_AR_JOB = ["موظف", "مستشار", "فني", "مدير قسم", "باحث", "مهندس", "محاسب", "منسق"]
_AR_DATE = ["١ كانون الثاني ٢٠٠٠", "١٥ آذار ٢٠٠١", "٣ أيار ١٩٩٩", "٢٠ تموز ٢٠٠٢"]

_EN_ORG = ["Acme Holdings", "Northwind Group", "Blue River Labs", "Summit Partners", "Globex Services",
           "Pinecrest Institute", "Silverline Bank", "Oakridge Hospital"]
_EN_LOC = ["Springfield", "Riverton", "Lakeside", "Maple Grove", "Fairview", "Brookhaven", "Cedar Falls"]
_EN_NAT = ["Freedonian", "Ruritanian", "Latverian", "Genovian"]
_EN_JOB = ["consultant", "analyst", "technician", "coordinator", "engineer", "manager", "officer"]


class _Faker:
    """Uses the `faker` package for English/Arabic names when installed, pools otherwise."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self._en = None
        try:
            from faker import Faker
            self._en = Faker("en_US")
            self._en.seed_instance(seed)
        except Exception:  # pragma: no cover
            pass

    def value(self, label: str, arabic: bool) -> str | None:
        r = self.rng
        if arabic:
            pools = {"PERSON": lambda: f"{r.choice(_AR_FIRST)} {r.choice(_AR_LAST)}",
                     "ORGANIZATION": lambda: r.choice(_AR_ORG), "LOCATION": lambda: r.choice(_AR_LOC),
                     "NATIONALITY": lambda: r.choice(_AR_NAT), "OCCUPATION": lambda: r.choice(_AR_JOB),
                     "DATE": lambda: r.choice(_AR_DATE)}
        else:
            f = self._en
            pools = {"PERSON": (lambda: f.name()) if f else (lambda: r.choice(["John Carter", "Emily Stone"])),
                     "ORGANIZATION": lambda: r.choice(_EN_ORG), "LOCATION": lambda: r.choice(_EN_LOC),
                     "NATIONALITY": lambda: r.choice(_EN_NAT), "OCCUPATION": lambda: r.choice(_EN_JOB),
                     "DATE": (lambda: f.date(pattern="%d %B %Y")) if f else (lambda: "1 January 2000")}
        pools.update({
            "EMAIL": lambda: f"user{r.randint(1000, 9999)}@example.com",
            "PHONE": lambda: f"+000 {r.randint(100, 999)} {r.randint(100, 999)} {r.randint(100, 999)}",
            "URL": lambda: f"https://example.com/{r.randint(1000, 9999)}",
            "IP_ADDRESS": lambda: f"192.0.2.{r.randint(1, 254)}",  # RFC 5737 documentation range
            "CREDIT_CARD": lambda: "0000 0000 0000 " + f"{r.randint(0, 9999):04d}",
            "IBAN": lambda: "XX00 0000 0000 " + f"{r.randint(0, 9999):04d}",
            "ID_NUMBER": lambda: "".join(str(r.randint(0, 9)) for _ in range(10)),
        })
        gen = pools.get(label)
        return gen() if gen else None


# ----------------------------------------------------------------------------- masker
class Masker:
    def __init__(self, method: str = "tag", per_label: dict[str, str] | None = None,
                 options: MaskOptions | None = None):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
        self.method = method
        self.per_label = {k: v for k, v in (per_label or {}).items() if v in METHODS}
        self.opt = options or MaskOptions()
        self._numbers: dict[str, dict[str, int]] = {}
        self._memo: dict[tuple[str, str, str], str] = {}
        self._fake = _Faker(self.opt.seed)
        self._alias: dict[tuple[str, str], str] = {}

    def method_for(self, label: str) -> str:
        return self.per_label.get(label, self.method)

    def _tag_name(self, label: str, arabic_ctx: bool) -> str:
        lang = self.opt.tag_language
        if lang == "auto":
            lang = "ar" if arabic_ctx else "en"
        spec = CATALOG.get(label)
        if lang == "ar":
            return spec.ar if spec else label.replace("_", " ")
        return label

    def replacement(self, f: Finding, context: str = "") -> str:
        method = self.method_for(f.label)
        nk = normalize_key(f.text)
        key = (method, f.label, self._alias.get((f.label, nk), nk))
        if method in ("pseudonym", "hash", "fake") and key in self._memo:
            return self._memo[key]
        arabic_ctx = is_arabic(context or f.text)
        style = TAG_STYLES.get(self.opt.tag_style, "[{label}]")

        if method == "tag":
            out = style.format(label=self._tag_name(f.label, arabic_ctx))
        elif method == "pseudonym":
            nums = self._numbers.setdefault(f.label, {})
            n = nums.setdefault(key[2], len(nums) + 1)
            out = style.format(label=f"{self._tag_name(f.label, arabic_ctx)}_{n}")
        elif method == "char_mask":
            out = "".join(ch if ch.isspace() else self.opt.mask_char for ch in f.text)
        elif method == "partial":
            out = partial_mask(f.text, f.label, max(0, self.opt.keep_first), self.opt.keep_last_digits)
        elif method == "hash":
            digest = hashlib.sha256(f"{self.opt.hash_salt}|{f.label}|{key[2]}".encode()).hexdigest()
            out = f"{f.label}_{digest[:self.opt.hash_length]}"
        elif method == "fake":
            out = None
            for _ in range(5):
                out = self._fake.value(f.label, is_arabic(f.text) or (arabic_ctx and not f.text.isascii()))
                if out is None or normalize_key(out) != key[2]:
                    break
            if out is None:  # custom labels have no generator -> numbered pseudonym
                nums = self._numbers.setdefault(f.label, {})
                n = nums.setdefault(key[2], len(nums) + 1)
                out = style.format(label=f"{self._tag_name(f.label, arabic_ctx)}_{n}")
        else:  # remove
            out = ""
        if method in ("pseudonym", "hash", "fake"):
            self._memo[key] = out
        return out

    def assign(self, findings: list[Finding], segments: list[str] | None = None) -> list[Finding]:
        """Set ``replacement`` on every enabled finding, in reading order (so pseudonym
        numbers follow the order of appearance)."""
        if self.opt.link_partial_names:
            self._alias = link_partial_names(findings)
        for f in sorted(findings, key=lambda x: (x.segment, x.start)):
            if f.enabled:
                ctx = segments[f.segment] if segments else f.text
                f.replacement = self.replacement(f, ctx)
            else:
                f.replacement = None
        return findings


def _contains(long: list[str], short: list[str]) -> bool:
    n = len(short)
    return any(long[i:i + n] == short for i in range(len(long) - n + 1))


def link_partial_names(findings: list[Finding]) -> dict[tuple[str, str], str]:
    """Map a shorter mention to the full name that contains it, per label:
    'Obama' -> 'barack obama', 'النجار' -> 'خالد النجار'. Ambiguous mentions ('Ahmed' inside both
    'Ahmed Hassan' and 'Ahmed Ali') are left alone."""
    by_label: dict[str, set[str]] = {}
    for f in findings:
        if f.enabled:
            by_label.setdefault(f.label, set()).add(normalize_key(f.text))
    alias: dict[tuple[str, str], str] = {}
    for label, keys in by_label.items():
        toks = {k: k.split() for k in keys}
        for k in keys:
            if len(k) < 3 or not toks[k]:
                continue
            parents = [p for p in keys if len(toks[p]) > len(toks[k]) and _contains(toks[p], toks[k])]
            if not parents:
                continue
            top = max(parents, key=lambda p: len(toks[p]))
            if all(_contains(toks[top], toks[q]) for q in parents):
                alias[(label, k)] = top
    return alias
