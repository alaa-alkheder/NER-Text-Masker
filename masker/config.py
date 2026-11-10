"""
Project configuration: label catalog, model choice and tuned defaults.

Model choice comes from the Colab study "NER: dslim/bert-base-NER vs numind/NuNER_Zero"
(GMB dataset for English, Wojood for Arabic, entity-level F1 with seqeval, paired bootstrap):

  English PER/ORG/LOC (test, 1000 sentences, after tuning on dev)
    dslim/bert-base-NER ........ F1 0.698  (agg=max)      ~123 sent/s
    numind/NuNER_Zero .......... F1 0.581  (thr=0.7)      ~65 sent/s
    dslim - NuNER = +0.117, 95% CI [+0.096, +0.138]  -> significant
    Ensemble (dslim-first + NuNER) = -0.026 vs dslim alone -> significant loss, NOT used.

  English labels dslim cannot produce (zero-shot, 8 GMB types)
    date (TIM) ......... NuNER F1 0.772  vs dslim 0.000
    nationality (GPE) .. NuNER F1 0.841  vs dslim 0.667
    -> NuNER_Zero handles every label other than PER/ORG/LOC, and every custom label.

  Arabic (Wojood dev) - urchade/gliner_multi-v2.1, zero-shot
    english labels + distractors, threshold 0.4 -> dev F1 0.742 (best in the sweep)
    -> gliner_multi-v2.1 handles Arabic. CAMeLBERT-NER is available as an optional
       backend for PERSON/ORGANIZATION/LOCATION.

So the engine *routes* each label to exactly one model per language instead of ensembling.

Second study: "NER model comparison.ipynb" (Colab, Tesla T4, Nov 2026) - same two models run
through the anonymization pipeline itself, 18 hand-labelled English sentences with hard cases
(53 entities), plus long-document, Arabic and end-to-end .txt/.docx tests:

  English PER/ORG/LOC          dslim F1 0.913, masking recall 0.943 | NuNER F1 0.916, masking recall 1.000
  Speed (s / 1000 words)       dslim CPU 2.89, GPU 0.18             | NuNER CPU 15.18, GPU 0.52
  Arabic (3 sentences)         gliner_multi F1 0.889, masking 1.000 | CAMeLBERT F1 0.889, one partial leak

  What it showed, and the rule added for each (all on by default, see DetectionSettings):
  * dslim missed every entity of an all-lowercase sentence ("lina khoury works for amazon in
    dubai") while NuNER found all three           -> caseless_fallback
  * NuNER found "Ahmed Al-Khatib" but missed a later "Ahmed" on its own, in every method
                                                    -> propagate_name_parts
  * NuNER put titles inside PER spans ("Dr. Sarah Johnson"), dslim did not
                                                    -> strip_titles
  * dslim labelled "St. Mary's Hospital", "Mayo Clinic", "Al-Noor Hospital" as LOC; NuNER
    labelled "جامعة حلب" as LOC                     -> fix_org_labels
  * without stride, the transformers pipeline silently read only the first 512 tokens
                                                    -> HF_STRIDE (safety net behind MAX_CHUNK_WORDS)
  * partial masking of "+963 944 123 456" left the first digit of every group
                                                    -> partial keeps only the last digits of numbers
  The small set does not overturn the larger GMB result, so PER/ORG/LOC stay on dslim
  (5.3x faster on CPU); the rules above cover the leaks the second study exposed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- models
MODELS: dict[str, dict] = {
    # supervised English NER (CoNLL-2003): PER / ORG / LOC / MISC
    "en_ner": {
        "id": "dslim/bert-base-NER",
        "kind": "hf",
        "aggregation": "max",  # best on dev in the notebook (section 6.1)
        "size": "~430 MB",
    },
    # zero-shot English (GLiNER architecture) - any label, labels must be lowercase
    "en_zero": {
        "id": "numind/NuNER_Zero",
        "kind": "gliner",
        "threshold": 0.7,  # best on dev (section 6.4)
        "merge_adjacent": True,  # token-level model -> merge "New" + "York"
        "lowercase_labels": True,
        "size": "~1.8 GB",
    },
    # zero-shot multilingual - used for Arabic
    "ar_zero": {
        "id": "urchade/gliner_multi-v2.1",
        "kind": "gliner",
        "threshold": 0.4,  # best on Arabic dev (section 14)
        "merge_adjacent": False,  # span-level model
        "lowercase_labels": False,
        "size": "~1.2 GB",
    },
    # optional supervised Arabic NER (ANERcorp): PERS / ORG / LOC / MISC
    "ar_ner": {
        "id": "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
        "kind": "hf",
        "aggregation": "first",
        "size": "~440 MB",
    },
}

# Distractor labels are sent to GLiNER even when the user did not pick them, so the model
# does not squeeze e.g. "Iraqi" into "location". Their predictions are dropped afterwards.
# (Arabic: "english+distractors" was the best prompt set in the notebook.)
USE_DISTRACTORS = True
DISTRACTOR_PROMPTS = ("person", "organization", "location", "nationality", "date")

# GLiNER / BERT inputs are limited (384 words / 512 sub-words); long paragraphs are
# split into sentence chunks of at most this many words.
MAX_CHUNK_WORDS = 120
# Safety net for supervised models: a chunk can still exceed 512 sub-words (URLs, long
# numbers). The pipeline then reads it in overlapping windows instead of silently
# dropping everything after token 512 (second study, section 6).
HF_STRIDE = 64

# Titles kept outside PERSON spans: "Dr. [PERSON_1]" instead of "[PERSON_1]" swallowing the title.
TITLES_EN = frozenset("""
mr mrs ms miss mx dr prof professor sir dame lord lady madam judge justice president
gen general col colonel capt captain lt sgt rev fr sheikh eng engr hon mister doctor
""".split())
TITLES_AR = frozenset("""
د د. دكتور الدكتور الدكتورة دكتورة السيد السيدة سيد سيدة الأستاذ الأستاذة أستاذ أ أ. م م.
المهندس المهندسة مهندس الشيخ شيخ الحاج الحاجة القاضي القاضية الرئيس العميد اللواء العقيد النقيب
""".split())

# A LOCATION whose name contains one of these words is an organization.
# English: any word ("University of Damascus", "Mayo Clinic"). Arabic: the first word ("جامعة حلب").
ORG_KEYWORDS_EN = frozenset("""
hospital hospitals clinic clinics university college school academy institute bank ministry
department authority agency council committee commission company corporation corp inc ltd llc
plc gmbh group holdings foundation association society federation party club airlines airways
laboratories labs organization organisation
""".split())
ORG_KEYWORDS_AR = frozenset("""
مستشفى المستشفى مشفى جامعة الجامعة كلية الكلية مدرسة المدرسة معهد المعهد أكاديمية بنك البنك مصرف
المصرف وزارة الوزارة هيئة الهيئة مؤسسة المؤسسة شركة الشركة مجموعة المجموعة جمعية الجمعية منظمة
المنظمة مجلس المجلس لجنة اللجنة حزب الحزب نادي النادي مركز المركز مختبر المختبر
""".split())

# Words that are never part of a name. A model sometimes takes the preposition or article
# before a name ('في دمشق', 'the United Nations'); these are removed from both ends of a span.
STOPWORDS_AR = frozenset("""
في من إلى الى على عن مع منذ حتى بين عند لدى نحو خلال ضد عبر قبل بعد فوق تحت حول دون لكل
و أو او ثم لكن بل أن إن ان كان كانت هذا هذه ذلك تلك هؤلاء الذي التي الذين هو هي هم قد لقد كل
""".split())
STOPWORDS_EN = frozenset("""
the a an of in on at to from by for with into onto about over under between during via
and or but nor his her their its our your my this that these those is was are were
""".split())
STOPWORDS = STOPWORDS_AR | STOPWORDS_EN

# Name parts that are never propagated on their own ("bin", "عبد", "Al"...).
NAME_PARTICLES = frozenset("""
bin bint ibn al el abu abd de da di du del della der den van von la le dos das the and jr sr ii iii
بن بنت ابن أبو ابو أبي آل ال عبد عبدال أم
""".split())
MIN_NAME_PART_LEN = 3

TAG_STYLES = {
    "square": "[{label}]",
    "angle": "<{label}>",
    "curly": "{{{label}}}",
}


# --------------------------------------------------------------------------- labels
@dataclass(frozen=True)
class LabelSpec:
    key: str  # canonical id, e.g. PERSON
    ar: str  # Arabic display name / Arabic tag
    en: str  # English display name
    kind: str  # "ner" (model) or "regex"
    description: str = ""
    hf_tags: tuple[str, ...] = ()  # tags of supervised models that map to this label
    prompts_en: tuple[str, ...] = ()  # GLiNER prompts for English text
    prompts_ar: tuple[str, ...] = ()  # GLiNER prompts for Arabic text
    en_backend: str = "zero"  # "hf" -> dslim, "zero" -> NuNER_Zero
    color: str = "#8a8984"
    custom: bool = False


CATALOG: dict[str, LabelSpec] = {
    s.key: s
    for s in [
        LabelSpec("PERSON", "شخص", "Person", "ner", "أسماء الأشخاص",
                  hf_tags=("PER", "PERS", "PERSON"), prompts_en=("person",),
                  prompts_ar=("person",), en_backend="hf", color="#2a78d6"),
        LabelSpec("ORGANIZATION", "منظمة", "Organization", "ner", "شركات، وزارات، جامعات، أحزاب",
                  hf_tags=("ORG", "ORGANIZATION"), prompts_en=("organization",),
                  prompts_ar=("organization",), en_backend="hf", color="#eb6834"),
        # v2_specific_loc (country / city / location) was the best NuNER prompt set
        LabelSpec("LOCATION", "موقع", "Location", "ner", "دول، مدن، مناطق، عناوين",
                  hf_tags=("LOC", "GEO", "LOCATION", "GPE"), prompts_en=("country", "city", "location"),
                  prompts_ar=("location",), en_backend="hf", color="#1baf7a"),
        LabelSpec("NATIONALITY", "جنسية", "Nationality", "ner", "جنسيات ومجموعات (عراقي، British)",
                  hf_tags=("NORP",), prompts_en=("nationality",), prompts_ar=("nationality",),
                  color="#eda100"),
        LabelSpec("DATE", "تاريخ", "Date", "ner", "تواريخ وأوقات (نموذج + أنماط رقمية)",
                  hf_tags=("DATE", "TIM", "TIME"), prompts_en=("date",), prompts_ar=("date",),
                  color="#e87ba4"),
        LabelSpec("OCCUPATION", "مهنة", "Occupation", "ner", "المسميات الوظيفية",
                  hf_tags=("OCC",), prompts_en=("occupation",), prompts_ar=("occupation",),
                  color="#7a5af5"),
        # ---- deterministic patterns
        LabelSpec("EMAIL", "بريد إلكتروني", "Email", "regex", "name@example.com", color="#0e7490"),
        LabelSpec("PHONE", "هاتف", "Phone", "regex", "أرقام الهواتف المحلية والدولية", color="#0e7490"),
        LabelSpec("URL", "رابط", "URL", "regex", "روابط الويب", color="#0e7490"),
        LabelSpec("IP_ADDRESS", "عنوان IP", "IP address", "regex", "IPv4", color="#0e7490"),
        LabelSpec("CREDIT_CARD", "بطاقة ائتمان", "Credit card", "regex", "مع تحقق Luhn", color="#0e7490"),
        LabelSpec("IBAN", "IBAN", "IBAN", "regex", "مع تحقق mod-97", color="#0e7490"),
        LabelSpec("ID_NUMBER", "رقم هوية", "ID number", "regex", "أرقام طويلة 9-18 خانة", color="#0e7490"),
    ]
}

NER_LABELS = [k for k, s in CATALOG.items() if s.kind == "ner"]
REGEX_LABELS = [k for k, s in CATALOG.items() if s.kind == "regex"]
DEFAULT_LABELS = ["PERSON", "ORGANIZATION", "LOCATION", "EMAIL", "PHONE"]


def custom_label(text: str) -> LabelSpec:
    """A user-typed label ('disease', 'مرض', 'medication'...) -> zero-shot only."""
    name = " ".join(text.strip().split())
    key = name.upper().replace(" ", "_")
    prompt = name.lower()
    return LabelSpec(key=key, ar=name, en=name, kind="ner", description="label مخصص (zero-shot)",
                     prompts_en=(prompt,), prompts_ar=(prompt,), en_backend="zero",
                     color="#52514e", custom=True)


def resolve_labels(keys: list[str], custom: list[str] | None = None) -> dict[str, LabelSpec]:
    out: dict[str, LabelSpec] = {}
    for k in keys:
        if k in CATALOG:
            out[k] = CATALOG[k]
    for c in custom or []:
        if c.strip():
            spec = custom_label(c)
            out.setdefault(spec.key, spec)
    return out


@dataclass
class DetectionSettings:
    labels: list[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    custom_labels: list[str] = field(default_factory=list)
    language: str = "auto"  # auto | ar | en
    arabic_backend: str = "gliner"  # gliner | camelbert
    thresholds: dict[str, float] = field(
        default_factory=lambda: {"en_zero": MODELS["en_zero"]["threshold"],
                                 "ar_zero": MODELS["ar_zero"]["threshold"]})
    min_hf_score: float = 0.0
    propagate: bool = True  # mask every other occurrence of a detected entity
    manual_terms: list[tuple[str, str]] = field(default_factory=list)  # (term, label)
    device: str = "auto"
    # rules added from the second study (see module docstring)
    caseless_fallback: bool = True  # all-lowercase / all-caps English -> zero-shot model for PER/ORG/LOC
    propagate_name_parts: bool = True  # "Ahmed" alone is masked once "Ahmed Al-Khatib" was found
    strip_titles: bool = True  # "Dr." / "الدكتور" stay outside PERSON spans
    fix_org_labels: bool = True  # "Mayo Clinic" / "جامعة حلب" labelled LOCATION -> ORGANIZATION
    strip_stopwords: bool = True  # "في دمشق" -> "دمشق": prepositions/articles stay outside spans

    def specs(self) -> dict[str, LabelSpec]:
        specs = resolve_labels(self.labels, self.custom_labels)
        for _, lab in self.manual_terms:
            if lab not in specs:
                specs[lab] = CATALOG.get(lab) or custom_label(lab)
        return specs
