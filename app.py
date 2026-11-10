"""
Streamlit interface - "مُقنِّع النصوص" (NER Text Masker)

    streamlit run app.py

Four steps:  ١ الملف والـ labels  →  ٢ طريقة التشفير  →  ٣ المراجعة  →  ٤ الإخراج
"""
from __future__ import annotations

import dataclasses
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from masker import CATALOG, DEFAULT_LABELS, METHODS, MODELS, DetectionSettings, Document, Masker, MaskOptions
from masker.config import NER_LABELS, REGEX_LABELS, TAG_STYLES, custom_label
from masker.detectors import models_available
from masker.document import masked_text
from masker.pipeline import MaskingPipeline, ModelProvider
from masker.text_utils import detect_language

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
STEPS = ["الملف والـ labels", "طريقة التشفير", "المراجعة", "الإخراج"]
SOURCE_AR = {"hf": "نموذج مُدرَّب", "zero": "نموذج zero-shot", "regex": "نمط Regex", "manual": "كلمة يدوية",
             "propagated": "تكرار"}

st.set_page_config(page_title="مُقنِّع النصوص | NER Text Masker", page_icon=":material/shield_lock:", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select { font-family: 'IBM Plex Sans Arabic', system-ui, sans-serif; }
.stApp, section[data-testid="stSidebar"] { direction: rtl; }
[data-testid="stMarkdownContainer"] :is(p, li, h1, h2, h3, h4, h5) { text-align: right; }
[data-testid="stSlider"], [data-testid="stDataFrame"], [data-testid="stDataEditor"], code, pre, .ltr { direction: ltr; }
[data-testid="stSlider"] label, [data-testid="stSlider"] [data-testid="stWidgetLabel"] { direction: rtl; }
.stepper { display: flex; gap: .5rem; margin: .25rem 0 1.25rem; flex-wrap: wrap; }
.stepper .s { flex: 1; min-width: 140px; padding: .55rem .8rem; border-radius: 10px; border: 1px solid rgba(128,128,128,.25);
              font-size: .92rem; opacity: .6; }
.stepper .s b { display: inline-block; width: 1.6rem; height: 1.6rem; line-height: 1.6rem; text-align: center;
                border-radius: 50%; background: rgba(128,128,128,.18); margin-left: .45rem; }
.stepper .s.done { opacity: .85; }
.stepper .s.active { opacity: 1; border-color: #2a78d6; box-shadow: 0 0 0 1px #2a78d6 inset; }
.stepper .s.active b { background: #2a78d6; color: #fff; }
.seg { padding: .45rem .7rem; margin-bottom: .35rem; border-radius: 8px; background: rgba(128,128,128,.07);
       line-height: 2.05; font-size: .98rem; }
.seg small { opacity: .55; font-size: .72rem; margin-inline-end: .4rem; }
mark.ent { padding: .05rem .3rem; border-radius: 5px; color: inherit; }
mark.ent sub { font-size: .62rem; opacity: .8; margin-inline-start: .25rem; font-weight: 600; }
.chip { display: inline-block; padding: .15rem .6rem; margin: .15rem; border-radius: 999px; font-size: .82rem;
        border: 1px solid rgba(128,128,128,.3); }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================ state helpers
def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


ss("step", 1)
ss("editor_version", 0)

# Streamlit forgets the value of a widget that is not drawn in the current step. Re-assigning
# the keys turns them into plain session state, so choices survive moving between steps.
_PERSIST = {"custom_labels", "manual_terms", "method", "tag_language", "tag_style", "mask_char", "keep_first",
            "hash_salt", "fake_seed", "per_label_on"}
for _k in list(st.session_state.keys()):
    if _k in _PERSIST or _k.startswith(("lbl_", "method_")):
        st.session_state[_k] = st.session_state[_k]
for _k in list(CATALOG):
    ss(f"lbl_{_k}", _k in DEFAULT_LABELS)
ss("custom_labels", "")
ss("manual_terms", "")
ss("method", "tag")
ss("tag_language", "auto")
ss("tag_style", "square")
ss("mask_char", "●")
ss("keep_first", 1)
ss("hash_salt", hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:32])
ss("fake_seed", 13)
ss("per_label_on", False)


def go(step: int):
    # freeze edits of the review table before leaving it
    if st.session_state.get("step") == 3 and "review_final" in st.session_state:
        st.session_state["review_base"] = st.session_state["review_final"]
        st.session_state["editor_version"] += 1
    st.session_state["step"] = step


def spec_of(key: str):
    return CATALOG.get(key) or custom_label(key.replace("_", " "))


def color_of(key: str) -> str:
    return spec_of(key).color


@st.cache_resource(show_spinner=False)
def get_provider(device: str) -> ModelProvider:
    """One provider per device; the loaded models stay in memory across reruns and users."""
    return ModelProvider(device)


MODELS_OK, MODELS_ERR = models_available()


# ============================================================================ sidebar
with st.sidebar:
    st.subheader("إعدادات النماذج")
    language = st.radio("لغة النص", ["auto", "ar", "en"], horizontal=True,
                        format_func={"auto": "تلقائي", "ar": "عربي", "en": "English"}.get,
                        help="«تلقائي» يكتشف اللغة لكل جملة، فيعمل مع الملفات المختلطة.")
    arabic_backend = st.selectbox(
        "نموذج العربي لـ شخص/منظمة/موقع", ["gliner", "camelbert"],
        format_func={"gliner": "gliner_multi (zero-shot)", "camelbert": "CAMeLBERT (supervised)"}.get,
        help="الافتراضي gliner_multi-v2.1 لأنه يدعم أي label بالعربي. CAMeLBERT مدرّب على ANERcorp لـ PER/ORG/LOC فقط.")
    th_en = st.slider("عتبة NuNER_Zero (إنجليزي)", 0.1, 0.95, float(MODELS["en_zero"]["threshold"]), 0.05,
                      help="أعلى = أخطاء أقل (Precision) · أقل = كشف أكثر (Recall). 0.7 هي الأفضل في النوتبوك.")
    th_ar = st.slider("عتبة gliner_multi (عربي)", 0.1, 0.95, float(MODELS["ar_zero"]["threshold"]), 0.05,
                      help="0.4 مع labels إنجليزية + distractors كانت الأفضل على بيانات Wojood.")
    propagate = st.toggle("إخفاء كل تكرارات الكيان في الملف", value=True,
                          help="إذا اكتُشف «خالد النجار» مرة، يُخفى في كل مكان يظهر فيه حتى لو فاته النموذج.")
    with st.expander("قواعد من نوتبوك المقارنة الثاني", expanded=False):
        name_parts_on = st.toggle("إخفاء الاسم الأول أو العائلة منفرداً", value=True, disabled=not propagate,
                                  help="بعد اكتشاف «Ahmed Al-Khatib» يُخفى «Ahmed» وحده أيضاً. في النوتبوك فات "
                                       "NuNER_Zero «Ahmed» و«Sara» منفردين فظهرا في الناتج.")
        caseless_on = st.toggle("النص الإنجليزي بلا أحرف كبيرة ← نموذج zero-shot", value=True,
                                help="dslim يعتمد على الحرف الكبير: فاته كل كيانات جملة "
                                     "«lina khoury works for amazon in dubai.» بينما وجدها NuNER_Zero كلها.")
        titles_on = st.toggle("إبقاء الألقاب خارج اسم الشخص", value=True,
                              help="«Dr. [PERSON_1]» بدل إخفاء اللقب مع الاسم، مثل الدكتور والسيد.")
        org_fix_on = st.toggle("تصحيح المستشفيات والجامعات إلى منظمة", value=True,
                               help="dslim وسم «Mayo Clinic» و«St. Mary's Hospital» كمواقع، وNuNER وسم «جامعة حلب» كموقع.")
        stopwords_on = st.toggle("حذف حروف الجر وكلمات مثل the من أطراف الكيان", value=True,
                                 help="«في دمشق» تصبح «دمشق»، و«the United Nations» تصبح «United Nations». "
                                      "الحروف الملتصقة بالاسم لا تُقص، لأن أسماء كثيرة تبدأ بـ و ف ب ل ك.")
    device = st.selectbox("المعالج", ["auto", "cpu", "cuda", "mps"],
                          help="auto = GPU (CUDA) إن وُجد وإلا CPU. mps لأجهزة Mac (تجريبي).")
    st.divider()
    with st.expander("لماذا هذه النماذج؟ (من نتائج النوتبوك)"):
        st.markdown(
            """
| المهمة | النموذج | النتيجة |
|---|---|---|
| إنجليزي PER/ORG/LOC | **dslim/bert-base-NER** | F1 = 0.698 |
| | NuNER_Zero | F1 = 0.581 |
| الفرق (bootstrap) | | +0.117، دال إحصائياً |
| إنجليزي date / nationality | **NuNER_Zero** | 0.772 / 0.841 |
| عربي (Wojood) | **gliner_multi-v2.1** | dev F1 = 0.742 |

- لا نستخدم Ensemble: خفّض F1 بمقدار 0.026 (دال إحصائياً).
- كل label يُوجَّه لنموذج واحد حسب اللغة، والـ labels المخصصة لنماذج zero-shot.
- الأنماط الثابتة (بريد، هاتف، IBAN…) بالـ Regex لأنها أدق من أي نموذج.

**النوتبوك الثاني** (مقارنة داخل مسار التجهيل نفسه، Tesla T4):

| | dslim | NuNER_Zero |
|---|---|---|
| F1 (53 كياناً) | 0.913 | 0.916 |
| Masking recall | 0.943 | 1.000 |
| ث/1000 كلمة CPU | 2.89 | 15.18 |
| ث/1000 كلمة GPU | 0.18 | 0.52 |

- dslim أسرع 5.3 مرات على CPU، لذلك بقي لـ PER/ORG/LOC، وأُضيفت قواعد تغطي ما سرّبه النموذجان.
- العربي: gliner_multi أخفى كل الكيانات (F1 0.889)، وCAMeLBERT سرّب كياناً جزئياً.
"""
        )
    if not MODELS_OK:
        st.warning("مكتبات النماذج (torch / transformers / gliner) غير مثبّتة، لذلك يعمل الكشف "
                   "بالأنماط الثابتة فقط.\n\n`pip install -r requirements.txt`")

# ============================================================================ header
st.title("مُقنِّع النصوص")
st.caption("إخفاء المعلومات الحساسة في ملفات Word والنصوص العربية والإنجليزية باستخدام نماذج NER — الإخفاء أحادي الاتجاه ولا يمكن عكسه.")

step = st.session_state["step"]
st.markdown(
    '<div class="stepper">' + "".join(
        f'<div class="s {"active" if i == step else "done" if i < step else ""}"><b>{i}</b>{name}</div>'
        for i, name in enumerate(STEPS, start=1)) + "</div>",
    unsafe_allow_html=True,
)


# ============================================================================ rendering helpers
def highlight(text: str, findings, masked: bool = False) -> str:
    """HTML of one segment with entities (or their replacements) highlighted."""
    fs = sorted([f for f in findings if f.enabled], key=lambda f: f.start)
    out, pos = [], 0
    for f in fs:
        if f.start < pos:
            continue
        out.append(html.escape(text[pos:f.start]))
        col = color_of(f.label)
        shown = f.replacement if masked else f.text
        if masked and shown == "":
            shown = "∅"
        out.append(f'<mark class="ent" style="background:{col}26;border-bottom:2px solid {col}">'
                   f'{html.escape(shown)}{"" if masked else f"<sub>{html.escape(f.label)}</sub>"}</mark>')
        pos = f.end
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def dir_style(text: str) -> str:
    return "direction:rtl;text-align:right" if detect_language(text) == "ar" else "direction:ltr;text-align:left"


def segment_block(seg, findings, masked=False) -> str:
    return (f'<div class="seg" style="{dir_style(seg.text)}"><small>{html.escape(seg.location)} #{seg.id}</small>'
            f'{highlight(seg.text, findings, masked)}</div>')


def load_document(data: bytes, name: str) -> Document:
    key = hashlib.sha1(data).hexdigest()
    if st.session_state.get("doc_key") != key:
        st.session_state["doc"] = Document.load(data, name)
        st.session_state["doc_key"] = key
        st.session_state["doc_name"] = name
        for k in ("findings", "det_signature", "review_base", "review_final"):
            st.session_state.pop(k, None)
    return st.session_state["doc"]


def current_settings() -> DetectionSettings:
    labels = [k for k in NER_LABELS + REGEX_LABELS if st.session_state.get(f"lbl_{k}")]
    custom = [c.strip() for c in (st.session_state.get("custom_labels") or "").replace("،", ",").split(",") if c.strip()]
    manual = []
    for line in (st.session_state.get("manual_terms") or "").splitlines():
        if not line.strip():
            continue
        term, _, lab = line.partition("|")
        lab = lab.strip().upper().replace(" ", "_") or "CUSTOM_TERM"
        manual.append((term.strip(), lab))
    return DetectionSettings(labels=labels, custom_labels=custom, language=language, arabic_backend=arabic_backend,
                             thresholds={"en_zero": th_en, "ar_zero": th_ar}, propagate=propagate,
                             manual_terms=manual, device=device, propagate_name_parts=name_parts_on,
                             caseless_fallback=caseless_on, strip_titles=titles_on, fix_org_labels=org_fix_on,
                             strip_stopwords=stopwords_on)


def build_review(findings) -> pd.DataFrame:
    rows: dict[tuple[str, str], dict] = {}
    for f in findings:
        k = (f.text, f.label)
        r = rows.setdefault(k, {"إخفاء": True, "النص": f.text, "label": f.label, "label الأصلي": f.label,
                                "المصدر": SOURCE_AR.get(f.source, f.source), "الثقة": f.score, "التكرار": 0})
        r["التكرار"] += 1
        r["الثقة"] = max(r["الثقة"], f.score)
        if f.source != "propagated" and r["المصدر"] == SOURCE_AR["propagated"]:
            r["المصدر"] = SOURCE_AR.get(f.source, f.source)
    df = pd.DataFrame(rows.values())
    if len(df):
        df = df.sort_values(["label", "التكرار"], ascending=[True, False]).reset_index(drop=True)
    return df


def reviewed_findings():
    """Findings copied with the user's review decisions applied (enabled flag / new label)."""
    findings = st.session_state.get("findings", [])
    review = st.session_state.get("review_final")
    if review is None:
        review = st.session_state.get("review_base")
    decisions = {}
    if review is not None and len(review):
        for _, r in review.iterrows():
            decisions[(r["النص"], r["label الأصلي"])] = (bool(r["إخفاء"]), r["label"])
    out = []
    for f in findings:
        enabled, label = decisions.get((f.text, f.label), (True, f.label))
        out.append(dataclasses.replace(f, enabled=enabled, label=label or f.label, replacement=None))
    return out


def make_masker() -> Masker:
    opts = MaskOptions(tag_language=st.session_state.get("tag_language", "auto"),
                       tag_style=st.session_state.get("tag_style", "square"),
                       mask_char=st.session_state.get("mask_char", "●") or "●",
                       keep_first=int(st.session_state.get("keep_first", 1)),
                       hash_salt=st.session_state.get("hash_salt") or "salt",
                       seed=int(st.session_state.get("fake_seed", 13)))
    per_label = {}
    if st.session_state.get("per_label_on"):
        for lab in st.session_state.get("labels_found", []):
            m = st.session_state.get(f"method_{lab}")
            if m:
                per_label[lab] = m
    return Masker(st.session_state.get("method", "tag"), per_label=per_label, options=opts)


# ============================================================================ step 1
if step == 1:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("١) اختر الملف")
        up = st.file_uploader("ملف Word (.docx) أو نص (.txt)", type=["docx", "txt"])
        samples = sorted(p.name for p in SAMPLES.glob("sample_*")) if SAMPLES.exists() else []
        sample = st.selectbox("أو جرّب ملفاً تجريبياً", ["—"] + samples) if samples else "—"
        data, name = None, None
        if up is not None:
            data, name = up.getvalue(), up.name
        elif sample != "—":
            data, name = (SAMPLES / sample).read_bytes(), sample
        if data:
            try:
                doc = load_document(data, name)
            except Exception as exc:
                st.error(f"تعذّرت قراءة الملف: {exc}")
                st.stop()
            stats = doc.stats()
            ar_share = sum(1 for s in doc.segments if s.text.strip() and any("؀" <= c <= "ۿ" for c in s.text))
            c1, c2, c3 = st.columns(3)
            c1.metric("فقرات / أسطر", stats["segments"])
            c2.metric("كلمات", stats["words"])
            c3.metric("فقرات عربية", f"{ar_share}/{stats['segments']}")
            with st.expander("معاينة المحتوى"):
                non_empty = [x for x in doc.segments if x.text.strip()][:40]
                st.markdown("".join(segment_block(x, []) for x in non_empty), unsafe_allow_html=True)
        elif "doc" in st.session_state:
            st.info(f"الملف الحالي: {st.session_state.get('doc_name')}")

    with right:
        st.subheader("٢) اختر الـ labels المراد إخفاؤها")
        st.markdown("**كيانات يكشفها نموذج NER**")
        cols = st.columns(2)
        for i, k in enumerate(NER_LABELS):
            s = CATALOG[k]
            cols[i % 2].checkbox(f"{s.ar} · {k}", key=f"lbl_{k}", help=s.description, disabled=not MODELS_OK)
        st.markdown("**أنماط ثابتة (Regex)**")
        cols = st.columns(2)
        for i, k in enumerate(REGEX_LABELS):
            s = CATALOG[k]
            cols[i % 2].checkbox(f"{s.ar} · {k}", key=f"lbl_{k}", help=s.description)
        st.text_input("labels مخصّصة (zero-shot) — افصل بفاصلة", key="custom_labels", disabled=not MODELS_OK,
                      placeholder="disease, medication, رقم الملف الطبي",
                      help="أي نوع تكتبه يبحث عنه النموذج مباشرة دون تدريب (NuNER_Zero / gliner_multi).")
        st.text_area("كلمات تُخفى دائماً (سطر لكل كلمة، واختيارياً | LABEL)", key="manual_terms", height=90,
                     placeholder="مشروع النسر | PROJECT\nAcme Holdings")

    st.divider()
    settings = current_settings()
    n_sel = len(settings.labels) + len(settings.custom_labels) + len(settings.manual_terms)
    ready = "doc" in st.session_state and n_sel > 0
    if st.button("التالي: كشف الكيانات", type="primary", disabled=not ready, icon=":material/arrow_back:"):
        doc = st.session_state["doc"]
        sig = json.dumps([st.session_state["doc_key"], dataclasses.asdict(settings)], sort_keys=True, default=str)
        if st.session_state.get("det_signature") != sig:
            pipe = MaskingPipeline(get_provider(device))
            bar = st.progress(0.0, text="تحميل النماذج والكشف… (التشغيل الأول ينزّل النماذج)")
            findings = pipe.detect(doc.texts, settings, progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
            bar.empty()
            st.session_state.update(findings=findings, det_signature=sig, det_warnings=pipe.warnings,
                                    review_base=build_review(findings), labels_found=sorted({f.label for f in findings}))
            st.session_state.pop("review_final", None)
            st.session_state["editor_version"] += 1
        go(2)
        st.rerun()
    if not ready:
        st.caption("اختر ملفاً وlabel واحداً على الأقل للمتابعة.")

# ============================================================================ step 2
elif step == 2:
    findings = st.session_state.get("findings", [])
    for w in st.session_state.get("det_warnings", []):
        st.warning(w)
    counts = pd.Series([f.label for f in findings]).value_counts() if findings else pd.Series(dtype=int)
    st.markdown(f"تم اكتشاف **{len(findings)}** موضعاً في **{len(set((f.text, f.label) for f in findings))}** كياناً مختلفاً: " +
                " ".join(f'<span class="chip" style="border-color:{color_of(k)}">{html.escape(spec_of(k).ar)} · {k}: {v}</span>'
                         for k, v in counts.items()), unsafe_allow_html=True)
    if not findings:
        st.info("لم يُكتشف أي كيان. ارجع وغيّر الـ labels أو خفّض العتبة من الشريط الجانبي.")

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.subheader("اختر طريقة التشفير")
        methods = list(METHODS)
        st.radio("الطريقة", methods, key="method",
                 format_func=lambda m: f"{METHODS[m]['ar']}  —  {METHODS[m]['example']}",
                 captions=[METHODS[m]["help"] for m in methods], label_visibility="collapsed")
        method = st.session_state["method"]
        with st.container(border=True):
            st.markdown("**خيارات**")
            if method in ("tag", "pseudonym", "fake"):
                c1, c2 = st.columns(2)
                c1.selectbox("لغة الوسم", ["auto", "ar", "en"], key="tag_language",
                             format_func={"auto": "حسب لغة النص", "ar": "عربي [شخص]", "en": "English [PERSON]"}.get)
                c2.selectbox("شكل الوسم", list(TAG_STYLES), key="tag_style",
                             format_func=lambda s: TAG_STYLES[s].format(label="PERSON"))
            if method == "char_mask":
                st.selectbox("رمز الإخفاء", ["●", "*", "█", "X", "#", "•"], key="mask_char")
            if method == "partial":
                st.number_input("عدد الأحرف الظاهرة من بداية كل كلمة", 0, 3, key="keep_first")
                st.caption("الأرقام (هاتف، بطاقة، IBAN…) تُبقي آخر رقمين فقط، والبريد أول حرف والامتداد.")
            if method == "hash":
                st.text_input("الملح (salt)", key="hash_salt",
                              help="احتفظ بنفس الملح لتحصل على نفس البصمة لنفس الكيان في ملفات مختلفة. "
                                   "بدون الملح لا يمكن تخمين الأصل من البصمة.")
            if method == "fake":
                st.number_input("البذرة (seed) لتكرار نفس البيانات الوهمية", 0, 10_000, key="fake_seed")
            if method == "remove":
                st.caption("سيُحذف النص مع مسافة واحدة مجاورة حتى لا تبقى فراغات مزدوجة.")
        st.toggle("طريقة مختلفة لكل label", key="per_label_on")
        if st.session_state.get("per_label_on"):
            for lab in st.session_state.get("labels_found", []):
                ss(f"method_{lab}", method)
                st.selectbox(f"{spec_of(lab).ar} · {lab}", methods, key=f"method_{lab}",
                             format_func=lambda m: METHODS[m]["ar"])

    with right:
        st.subheader("مثال مباشر على كياناتك")
        doc = st.session_state["doc"]
        demo = make_masker().assign([dataclasses.replace(f) for f in findings], doc.texts)
        seen, rows = set(), []
        for f in sorted(demo, key=lambda x: (x.segment, x.start)):
            if (f.text, f.label) in seen:
                continue
            seen.add((f.text, f.label))
            rows.append({"الأصل": f.text, "label": f.label, "بعد التشفير": f.replacement or "∅"})
            if len(rows) == 12:
                break
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    c1, c2 = st.columns([1, 5])
    if c1.button("السابق", icon=":material/arrow_forward:"):
        go(1)
        st.rerun()
    if c2.button("التالي: المراجعة", type="primary", icon=":material/arrow_back:"):
        go(3)
        st.rerun()

# ============================================================================ step 3
elif step == 3:
    doc = st.session_state["doc"]
    st.subheader("راجع الكيانات قبل الإخراج")
    st.caption("ألغِ علامة «إخفاء» لأي خطأ من النموذج، أو غيّر الـ label. القرار يُطبَّق على كل تكرارات نفس النص.")
    base = st.session_state.get("review_base")
    if base is None or not len(base):
        st.info("لا توجد كيانات للمراجعة.")
    else:
        label_options = sorted(set(base["label"]) | set(st.session_state.get("labels_found", [])))
        edited = st.data_editor(
            base, key=f"review_editor_{st.session_state['editor_version']}", hide_index=True, height=360,
            disabled=["النص", "label الأصلي", "المصدر", "الثقة", "التكرار"],
            column_order=["إخفاء", "النص", "label", "المصدر", "الثقة", "التكرار"],
            column_config={
                "إخفاء": st.column_config.CheckboxColumn(width="small"),
                "label": st.column_config.SelectboxColumn(options=label_options, required=True),
                "الثقة": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                "النص": st.column_config.TextColumn(width="large"),
            },
        )
        st.session_state["review_final"] = edited
        n_on = int(edited["إخفاء"].sum())
        st.caption(f"سيتم إخفاء {n_on} من {len(edited)} كياناً مختلفاً.")

    fs = reviewed_findings()
    make_masker().assign(fs, doc.texts)
    by_seg: dict[int, list] = {}
    for f in fs:
        by_seg.setdefault(f.segment, []).append(f)
    shown = [s for s in doc.segments if s.id in by_seg][:40]
    st.subheader("معاينة قبل / بعد")
    c1, c2 = st.columns(2, gap="medium")
    c1.markdown("**النص الأصلي**")
    c2.markdown("**بعد التشفير**")
    for seg in shown:
        a, b = st.columns(2, gap="medium")
        a.markdown(segment_block(seg, by_seg[seg.id]), unsafe_allow_html=True)
        b.markdown(segment_block(seg, by_seg[seg.id], masked=True), unsafe_allow_html=True)
    if len([s for s in doc.segments if s.id in by_seg]) > len(shown):
        st.caption("تُعرض أول 40 فقرة فيها كيانات فقط.")

    c1, c2 = st.columns([1, 5])
    if c1.button("السابق", icon=":material/arrow_forward:"):
        go(2)
        st.rerun()
    if c2.button("التالي: الإخراج", type="primary", icon=":material/arrow_back:"):
        go(4)
        st.rerun()

# ============================================================================ step 4
elif step == 4:
    doc: Document = st.session_state["doc"]
    fs = reviewed_findings()
    masker = make_masker()
    masker.assign(fs, doc.texts)
    active = [f for f in fs if f.enabled]

    st.subheader("الملف المشفَّر جاهز")
    scrub = st.checkbox("مسح البيانات الوصفية (المؤلف، آخر من عدّل، العنوان، كتّاب التعليقات)", value=True,
                        disabled=doc.kind != "docx")
    out_bytes = doc.render(fs, scrub_metadata=scrub)
    stem, ext = Path(st.session_state.get("doc_name", "document.docx")).stem, Path(doc.filename).suffix
    out_name = f"{stem}_masked{ext}"
    mime = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document" if ext == ".docx"
            else "text/plain")

    c1, c2, c3 = st.columns(3)
    c1.metric("مواضع مُخفاة", len(active))
    c2.metric("كيانات مختلفة", len({(f.text, f.label) for f in active}))
    c3.metric("الطريقة", METHODS[masker.method]["ar"])

    st.download_button(f"تنزيل {out_name}", data=out_bytes, file_name=out_name, mime=mime, type="primary",
                       icon=":material/download:")

    counts = pd.Series([f.label for f in active]).value_counts()
    summary = {
        "file": st.session_state.get("doc_name"),
        "output": out_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": masker.method,
        "per_label_methods": masker.per_label,
        "labels_masked": {k: int(v) for k, v in counts.items()},
        "total_masked": len(active),
        "models": sorted({f.model for f in active if f.model}),
        "note": "This report contains no original values.",
    }
    with st.expander("تقرير الإخفاء (بدون القيم الأصلية)"):
        st.json(summary)
        st.download_button("تنزيل التقرير JSON", json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name=f"{stem}_masking_report.json", mime="application/json")
    with st.expander("ملف تدقيق يحتوي القيم الأصلية (حسّاس)"):
        st.warning("هذا الملف يربط كل قيمة أصلية بما استُبدلت به؛ من يملكه يستطيع عكس الإخفاء. لا تشاركه مع الملف المشفّر.")
        if st.checkbox("أفهم ذلك، أظهر زر التنزيل"):
            audit = pd.DataFrame([{"segment": f.segment, "location": doc.segments[f.segment].location, "label": f.label,
                                   "original": f.text, "replacement": f.replacement, "source": f.source,
                                   "score": f.score} for f in active])
            st.download_button("تنزيل ملف التدقيق CSV", audit.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{stem}_audit_SENSITIVE.csv", mime="text/csv")

    with st.expander("معاينة النص المشفّر", expanded=True):
        by_seg: dict[int, list] = {}
        for f in fs:
            by_seg.setdefault(f.segment, []).append(f)
        preview = [f'<div class="seg" style="{dir_style(s.text)}">{html.escape(masked_text(s.text, by_seg.get(s.id, [])))}</div>'
                   for s in doc.segments if s.text.strip()][:60]
        st.markdown("".join(preview), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 5])
    if c1.button("السابق", icon=":material/arrow_forward:"):
        go(3)
        st.rerun()
    if c2.button("ملف جديد", icon=":material/restart_alt:"):
        for k in ("doc", "doc_key", "doc_name", "findings", "det_signature", "review_base", "review_final"):
            st.session_state.pop(k, None)
        go(1)
        st.rerun()
