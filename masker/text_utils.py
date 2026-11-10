"""Language detection, sentence chunking and Arabic normalisation helpers."""
from __future__ import annotations

import re

_AR = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LAT = re.compile(r"[A-Za-z]")
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # + tatweel


def detect_language(text: str) -> str:
    """'ar' if Arabic letters are at least 30% of the letters, else 'en'."""
    ar = len(_AR.findall(text))
    lat = len(_LAT.findall(text))
    if ar + lat == 0:
        return "en"
    return "ar" if ar >= 0.3 * (ar + lat) else "en"


def is_arabic(text: str) -> bool:
    return bool(_AR.search(text))


def normalize_key(text: str) -> str:
    """Key used to give the same replacement to spelling variants of one entity."""
    t = " ".join(text.split()).casefold()
    t = _DIACRITICS.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return t


# sentence boundary: after . ! ? ؟ ; ؛ followed by space, or any newline
_SENT_END = re.compile(r"(?<=[.!?؟;؛])\s+|\n+")
_WORD = re.compile(r"\S+")


def chunk_text(text: str, max_words: int = 120) -> list[tuple[int, int]]:
    """Split text into (start, end) character windows of whole sentences, each with at
    most ``max_words`` words (long sentences are cut on word boundaries)."""
    sentences: list[tuple[int, int]] = []
    pos = 0
    for m in _SENT_END.finditer(text):
        sentences.append((pos, m.start()))
        pos = m.end()
    sentences.append((pos, len(text)))

    pieces: list[tuple[int, int, int]] = []  # (start, end, n_words)
    for s, e in sentences:
        words = [(m.start() + s, m.end() + s) for m in _WORD.finditer(text[s:e])]
        if not words:
            continue
        for i in range(0, len(words), max_words):
            part = words[i:i + max_words]
            pieces.append((part[0][0], part[-1][1], len(part)))

    chunks: list[tuple[int, int]] = []
    cur_s = cur_e = None
    cur_n = 0
    for s, e, n in pieces:
        if cur_s is not None and cur_n + n <= max_words:
            cur_e, cur_n = e, cur_n + n
        else:
            if cur_s is not None:
                chunks.append((cur_s, cur_e))
            cur_s, cur_e, cur_n = s, e, n
    if cur_s is not None:
        chunks.append((cur_s, cur_e))
    return chunks


_TRIM_CHARS = " \t\n\r‏‎\"'“”‘’«»()[]{}<>,،;؛:"


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Drop surrounding whitespace/quotes/brackets and a lone trailing period."""
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    if end - start > 1 and text[end - 1] == "." and "." not in text[start:end - 1]:
        end -= 1
    return start, end


_LATIN_UPPER = re.compile(r"[A-Z]")
_LATIN_LOWER = re.compile(r"[a-z]")


def lacks_case_signal(text: str, min_letters: int = 6) -> bool:
    """True for English text written all in lower case (or all in capitals). A cased model
    such as dslim/bert-base-NER relies on capital letters to find names; in the second study it
    missed every entity of 'lina khoury works for amazon in dubai.'"""
    upper, lower = len(_LATIN_UPPER.findall(text)), len(_LATIN_LOWER.findall(text))
    if upper + lower < min_letters:
        return False
    return upper == 0 or lower == 0


def split_by_case(text: str, start: int, end: int) -> list[tuple[int, int, bool]]:
    """Split text[start:end] into runs of whole sentences, flagging the runs that have no case
    signal: [(s, e, caseless), ...]. Consecutive sentences with the same flag stay together."""
    sentences: list[tuple[int, int]] = []
    pos = start
    for m in _SENT_END.finditer(text, start, end):
        sentences.append((pos, m.start()))
        pos = m.end()
    sentences.append((pos, end))
    runs: list[list] = []
    for s, e in sentences:
        if not text[s:e].strip():
            continue
        flag = lacks_case_signal(text[s:e])
        if runs and runs[-1][2] == flag:
            runs[-1][1] = e
        else:
            runs.append([s, e, flag])
    return [(s, e, f) for s, e, f in runs]


def _token_key(token: str) -> str:
    return token.strip(".,:;،؛").casefold()


def strip_title(text: str, start: int, end: int, titles: frozenset[str]) -> tuple[int, int]:
    """Move ``start`` past leading titles: 'Dr. Sarah Johnson' -> 'Sarah Johnson',
    'الدكتور خالد النجار' -> 'خالد النجار', 'Prof. Dr. X' -> 'X'. Keeps at least one word."""
    while True:
        m = _WORD.match(text, start, end) if start < end else None
        if not m:
            return start, end
        rest = text[m.end():end]
        if _token_key(m.group()) not in titles or not rest.strip():
            return start, end
        start = m.end() + (len(rest) - len(rest.lstrip()))


def has_org_keyword(span: str, keywords_en: frozenset[str], keywords_ar: frozenset[str]) -> bool:
    """'Mayo Clinic', 'University of Damascus' (any English word) or 'جامعة حلب' (first Arabic word)."""
    tokens = [_token_key(t) for t in span.split()]
    if not tokens:
        return False
    if is_arabic(span):
        return tokens[0] in keywords_ar
    return any(t in keywords_en for t in tokens)


def name_parts(name: str, particles: frozenset[str], titles: frozenset[str], min_len: int = 3) -> list[str]:
    """Single words of a multi-word person name that may appear on their own later:
    'Ahmed Al-Khatib' -> ['Ahmed', 'Al-Khatib']; 'Khalid bin Saeed' -> ['Khalid', 'Saeed']."""
    words = [w.strip(".,:;،؛\"'()") for w in name.split()]
    words = [w for w in words if w]
    if len(words) < 2:
        return []
    out, i = [], 0
    while i < len(words):
        w = words[i]
        key = w.casefold()
        # compound names stay whole: 'عبد الله', 'أبو بكر', 'Abdul Rahman' (never 'الله' alone)
        if key in _COMPOUND_PREFIXES and i + 1 < len(words):
            part = f"{w} {words[i + 1]}"
            if part != " ".join(words):
                out.append(part)
            i += 2
            continue
        i += 1
        if len(w) < min_len or key in particles or key in titles or not any(c.isalpha() for c in w):
            continue
        out.append(w)
    return list(dict.fromkeys(out))


_COMPOUND_PREFIXES = frozenset({"عبد", "عبدال", "أبو", "ابو", "أبي", "أم", "abd", "abdul", "abdel", "abu"})


_PROCLITICS = "وفبلك"

# Letters that are written in several ways for the same name: أحمد / احمد, إبراهيم / ابراهيم,
# مصطفى / مصطفي, مكة / مكه. Each maps to ONE character, so offsets stay easy to map back.
_SEARCH_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})


def _is_mark(ch: str) -> bool:
    return bool(_DIACRITICS.match(ch))


def fold_for_search(text: str) -> tuple[str, list[int]]:
    """Text used to look for other mentions of a name: Arabic letter variants unified and
    diacritics/tatweel removed. Returns the folded text and, for each folded character, its
    index in the original text. Latin text is left as it is (case still matters)."""
    out, index = [], []
    for i, ch in enumerate(text):
        if _is_mark(ch):
            continue
        out.append(ch.translate(_SEARCH_FOLD))
        index.append(i)
    return "".join(out), index


def find_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    """Whole-word occurrences of ``needle`` in ``text``. Arabic spelling variants match each
    other ('أحمد' finds 'احمد' and 'أَحْمَد'). For Arabic, a leading proclitic (و ف ب ل ك, up to
    two) is allowed and left outside the span: 'ومحمد' -> 'و' + [محمد]."""
    out: list[tuple[int, int]] = []
    folded_needle, _ = fold_for_search(needle.strip())
    if not folded_needle:
        return out
    folded, index = fold_for_search(text)
    arabic = is_arabic(needle)
    i = folded.find(folded_needle)
    while i != -1:
        j = i + len(folded_needle)
        right_ok = j == len(folded) or not folded[j].isalnum()
        left_ok = i == 0 or not folded[i - 1].isalnum()
        if not left_ok and arabic:
            k = i
            for _ in range(2):
                if k > 0 and folded[k - 1] in _PROCLITICS:
                    k -= 1
                    if k == 0 or not folded[k - 1].isalnum():
                        left_ok = True
                        break
                else:
                    break
        if left_ok and right_ok:
            start, end = index[i], index[j - 1] + 1
            while end < len(text) and _is_mark(text[end]):  # keep the diacritics of the last letter
                end += 1
            out.append((start, end))
        i = folded.find(folded_needle, i + 1)
    return out


def strip_stopwords(text: str, start: int, end: int, stopwords: frozenset[str]) -> tuple[int, int]:
    """Drop whole stop words at both ends of a span: 'في دمشق' -> 'دمشق', 'the United Nations'
    -> 'United Nations'. Words inside the span stay ('Bank of America'). Letters attached to a
    word are never cut ('بدمشق' stays whole), because many names start with و ف ب ل ك.
    A span made only of stop words comes back empty (start == end)."""
    words = [(m.start(), m.end()) for m in _WORD.finditer(text, start, end)]
    key = lambda w: _token_key(text[w[0]:w[1]])
    while words and key(words[0]) in stopwords:
        words.pop(0)
    while words and key(words[-1]) in stopwords:
        words.pop()
    if not words:
        return start, start
    return words[0][0], words[-1][1]
