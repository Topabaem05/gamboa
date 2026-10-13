"""Vessel entity matcher for the DACON vessel-identification competition (v2).

Two layers:

1. ``AnswerBook`` - hard-coded answers memorised from the official ``train.csv``
   (``call_sign`` and ``vessel_name`` each map one-to-one onto the target).
2. ``Ranker`` - candidate retrieval over ``vessel_catalog.csv`` and a
   RandomForest pairwise scorer for rows the answer book has never seen.

Korean (Hangul) names are transliterations of the catalog's Latin names, so
both sides are mapped into a shared *phonetic class* alphabet (e.g. L/R -> R,
B/V -> B, soft C/J/Z/S -> S, post-vocalic R dropped) before comparison.

Every query row is processed on its own; nothing looks at other evaluation
rows or at the evaluation distribution.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import multiprocessing as mp
import os
import re
import unicodedata

import numpy as np
import pandas as pd

VERSION = "hardcoded-lookup+phonetic-hgb-v5"

FEATURE_NAMES = (
    "latin_exact", "latin_lev", "latin_osa", "latin_bigram_jacc", "latin_trigram_jacc",
    "latin_prefix_ratio", "latin_length_ratio",
    "cons_exact", "cons_lev", "cons_bigram_jacc", "cons_trigram_jacc", "cons_length_ratio",
    "vow_exact", "vow_lev", "vow_osa", "vow_bigram_jacc",
    "token_lev_mean", "token_lev_min", "token_lev_max", "token_count_diff", "token_overlap",
    "latin_token_exact", "latin_token_osa_max",
    "first_token_exact", "last_token_exact", "last_token_osa", "last_token_cons_lev",
    "digits_match", "query_has_digits",
    "gross_gap", "intl_gap", "best_gap", "gt_exact", "query_gt_missing", "cat_gt_missing",
    "flag_prob", "flag_missing", "type_prob",
    "query_is_hangul", "src_name", "src_tonnage", "src_exact",
    # listwise (relative to the other candidates of the same query row)
    "n_candidates",
    "best_gap_minus_min", "is_min_gap",
    "cons_lev_minus_max", "is_max_cons_lev",
    "latin_osa_minus_max", "is_max_latin_osa",
    "vow_lev_minus_max", "is_max_vow_lev",
    "flag_prob_minus_max", "is_max_flag_prob",
    "type_prob_minus_max", "is_max_type_prob",
    "latin_token_exact_minus_max", "is_max_latin_token_exact",
    "token_lev_max_minus_max", "is_max_token_lev_max",
    "digits_match_minus_max",
    "name_score_minus_max", "is_max_name_score",
)
_BASE_FEATURES = FEATURE_NAMES[:FEATURE_NAMES.index("n_candidates")]
_LISTWISE = (("best_gap", "min"), ("cons_lev", "max"), ("latin_osa", "max"), ("vow_lev", "max"),
             ("flag_prob", "max"), ("type_prob", "max"), ("latin_token_exact", "max"),
             ("token_lev_max", "max"), ("digits_match", "max"), ("last_token_osa", "max"))


def pair_matrix(ranker: "Ranker", row: dict, index: "CatalogIndex",
                cands: list[tuple[int, int]]) -> np.ndarray:
    """Base pairwise features for every candidate plus listwise context features."""
    base = np.asarray([ranker.features(row, index, i, b) for i, b in cands], dtype=np.float32)
    n = len(cands)
    cols = {name: base[:, j] for j, name in enumerate(_BASE_FEATURES)}
    extra = [np.full(n, float(n), dtype=np.float32)]
    for name, how in _LISTWISE:
        col = cols[name]
        best = col.min() if how == "min" else col.max()
        extra.append(col - best)
        if name != "digits_match":
            extra.append((col == best).astype(np.float32))
    name_score = np.maximum.reduce([cols["cons_lev"], cols["latin_osa"], cols["token_lev_max"]])
    extra.append(name_score - name_score.max())
    extra.append((name_score == name_score.max()).astype(np.float32))
    return np.column_stack([base] + extra).astype(np.float32)

# --------------------------------------------------------------------------- #
# Hangul decomposition
# --------------------------------------------------------------------------- #

# Phonetic classes of Hangul jamo (initial, medial, final)
_INIT_CLS = ["K", "K", "N", "D", "D", "R", "M", "B", "B", "S", "S", "",
             "S", "S", "C", "K", "T", "P", "H"]
_MED_VOW = ["A", "A", "A", "A", "U", "E", "U", "E", "O", "A", "A",
            "E", "O", "U", "O", "E", "I", "U", "", "I", "I"]
_FIN_CLS = ["", "K", "K", "K", "N", "N", "N", "T", "R", "K", "M", "P", "R",
            "R", "P", "R", "M", "P", "P", "T", "T", "N", "T", "T", "K", "T",
            "P", "T"]
# Revised romanisation (for a Latin-looking key)
_INIT_ROM = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "",
             "j", "jj", "ch", "k", "t", "p", "h"]
_MED_ROM = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
            "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
_FIN_ROM = ["", "k", "k", "k", "n", "n", "n", "t", "l", "lk", "lm", "lp", "ls",
            "lt", "lp", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t",
            "p", "t"]


def is_hangul(value: object) -> bool:
    return isinstance(value, str) and any("가" <= ch <= "힣" for ch in value)


def _syllables(text: str):
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            code -= 0xAC00
            yield True, code // 588, (code % 588) // 28, code % 28, ch
        else:
            yield False, 0, 0, 0, ch


def romanize(text: str) -> str:
    out = []
    for is_syl, i, m, f, ch in _syllables(text):
        out.append(_INIT_ROM[i] + _MED_ROM[m] + _FIN_ROM[f] if is_syl else ch)
    return "".join(out)


_ROMAN = {"II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
          "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17,
          "XVIII": 18, "XIX": 19, "XX": 20}


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).upper()
    value = re.sub(r"^\s*M\s*[/.]?\s*[VT]\s*[/.]?\s+", "", value)   # "M/V Seren" -> "Seren"
    value = re.sub(r"제\s*(\d+)", r"NO\1", value)        # 제7하나 -> NO7하나
    if len(re.findall(r"[가-힣]", value)) >= 3:
        value = re.sub(r"호\s*$", "", value)             # trailing 호 (ship suffix)
    # "No. 2 Asian Pioneer" / "Ocean Pioneer No. 2" / 제2 -> move the hull number to the end
    value = re.sub(r"\b(X{0,1}(?:IX|IV|V?I{0,3}))\s*$",
                   lambda m: str(_ROMAN[m.group(1)]) if m.group(1) in _ROMAN else m.group(0), value)
    value = re.sub(r"\b(투|쓰리)\s*$", lambda m: {"투": "2", "쓰리": "3"}[m.group(1)], value)
    numbers = re.findall(r"NO\s*\.?\s*(\d+)", value)
    if numbers:
        value = re.sub(r"NO\s*\.?\s*\d+", " ", value)
        value = value.strip() + "".join(f" {n}" for n in numbers)
    return value.strip()


def tokens(value: object) -> list[str]:
    value = _clean(value)
    return [t for t in re.split(r"[^0-9A-Z가-힣]+", value) if t]


def latin_key(value: object) -> str:
    """Latin-only comparison key (Hangul romanised), alphanumerics only."""
    return "".join(ch for ch in romanize(_clean(value)).upper() if ch.isalnum())


# --------------------------------------------------------------------------- #
# Phonetic class strings
# --------------------------------------------------------------------------- #

_VOWELS = set("AEIOU")


def _english_phones(word: str) -> list[str]:
    """Map an upper-case Latin word to phone classes (vowels lower-case)."""
    if _INITIALISM.match(word):                               # KS, CNC, JBU, MSC ... spelled out
        return [p for ch in word for p in _LETTER_PHONES[ch]]
    w = word
    w = re.sub(r"^H(?=ON|OUR)", "", w)                          # Honour, Hour
    w = re.sub(r"([BDFGHJ-NP-TVWXZ])\1", r"\1", w)          # double consonants (not CC)
    w = re.sub(r"(?<=[AEIOU])GH", "", w)                       # Highway, Eight
    w = re.sub(r"^GH", "K", w)                                  # Ghazal
    w = w.replace("CHEM", "KEM").replace("CHOR", "KOR").replace("CHR", "KR").replace("CHL", "KL")
    w = w.replace("CHT", "Č").replace("TCH", "Č").replace("CH", "Č").replace("SH", "S").replace("PH", "P")
    w = re.sub(r"^TH", "T", w).replace("TH", "S")              # Thor -> 토르, Zenith -> 제니스
    w = w.replace("CK", "K").replace("QU", "K").replace("WH", "W")
    w = w.replace("ZH", "S").replace("BH", "B").replace("DH", "D").replace("KH", "K")
    w = w.replace("DG", "S").replace("TION", "SON").replace("SION", "SON")
    w = re.sub(r"^X", "S", w).replace("X", "KS")               # Xiu -> 슈, Galaxy -> 갤럭시
    w = w.replace("WR", "R")
    w = re.sub(r"^TS", "S", w)                                  # Tsurugi -> 쮸루기
    w = w.replace("NGY", "NY")                                  # Kwangyang -> 광양
    w = re.sub(r"NG(?![AEIOUYLR])", "N", w)
    w = re.sub(r"NH$", "N", w)                                  # Hochiminh -> 호치민
    w = re.sub(r"(?<=[AEIOU])H$", "", w)                        # Sarah -> 사라
    w = re.sub(r"R(?![AEIOUYW])", "", w)                       # Harbour -> 하버, Lombard -> 롬바드
    silent_e = len(w) > 3 and w.endswith("E") and w[-2] not in "AEIOU"
    out: list[str] = []
    n = len(w)
    for i, ch in enumerate(w):
        nxt = w[i + 1] if i + 1 < n else ""
        if ch.isdigit():
            out.append(ch)
        elif ch in "AEIOU":
            out.append(ch.lower())
        elif ch == "Y":
            out.append("i" if (not nxt or nxt not in "AEIOU") else "")
        elif ch == "W":
            out.append("")
        elif ch == "Č":
            out.append("C")
        elif ch == "C":
            out.append("S" if (nxt and nxt in "EIY") else "K")
        elif ch == "G":
            out.append("S" if (nxt and nxt in "EIY") else "K")
        elif ch in "JZS":
            out.append("S")
        elif ch in "BV":
            out.append("B")
        elif ch in "PF":
            out.append("P")
        elif ch == "Q":
            out.append("K")
        elif ch in "LR":
            out.append("R")
        else:
            out.append(ch)
    out = [p for p in out if p]
    if silent_e and out and out[-1] == "e":
        out.pop()
    return out


def _hangul_phones(word: str) -> list[str]:
    out: list[str] = []
    for is_syl, i, m, f, ch in _syllables(word):
        if not is_syl:
            if ch.isalnum():
                out.extend(_english_phones(ch.upper()) if ch.isalpha() else [ch])
            continue
        ini, vow, fin = _INIT_CLS[i], _MED_VOW[m], _FIN_CLS[f]
        if i == 5 and m == 18 and f == 0:
            continue
        if ini:
            out.append(ini)
        if vow:
            out.append(vow.lower())
        if fin:
            out.append(fin)
    return out


_LETTER_NAMES = {
    "에이치": "H", "더블유": "W", "에이": "A", "에프": "F", "아이": "I", "제이": "J",
    "케이": "K", "에스": "S", "브이": "V", "엑스": "X", "와이": "Y", "제트": "Z",
    "지드": "Z", "비": "B", "씨": "C", "디": "D", "이": "E", "지": "G", "엘": "L",
    "엠": "M", "엔": "N", "앤": "N", "오": "O", "피": "P", "큐": "Q", "알": "R",
    "티": "T", "유": "U", "쥐": "G", "시": "C",
}
# How Korean spells out Latin letter names, as phones (consonant classes upper, vowels lower)
_LETTER_PHONES = {
    "A": ["a"], "B": ["B", "i"], "C": ["S", "i"], "D": ["D", "i"], "E": ["i"], "F": ["e", "P"],
    "G": ["S", "i"], "H": ["a", "i", "C"], "I": ["a", "i"], "J": ["S", "a", "i"],
    "K": ["K", "a", "i"], "L": ["e", "R"], "M": ["e", "M"], "N": ["e", "N"], "O": ["o"],
    "P": ["P", "i"], "Q": ["K", "u"], "R": ["a", "R"], "S": ["e", "S"], "T": ["T", "i"],
    "U": ["u"], "V": ["B", "u", "i"], "W": ["D", "e", "B", "u", "R", "u"], "X": ["e", "K", "S"],
    "Y": ["a", "i"], "Z": ["S", "e", "T"],
}
_INITIALISM = re.compile(r"^[B-DF-HJ-NP-TV-Z]{2,4}$")
_LETTER_KEYS = sorted(_LETTER_NAMES, key=len, reverse=True)


def split_letter_names(token: str) -> tuple[str, str]:
    """Greedy parse of spelled-out Latin letters at the start of a Hangul token.

    ``케이에스`` -> ("KS", ""), ``엠에스살비아`` -> ("MS", "살비아"),
    ``에이스`` -> ("", "에이스") (fewer than two letters: not an initialism).
    """
    letters = []
    pos = 0
    while pos < len(token):
        for key in _LETTER_KEYS:
            if token.startswith(key, pos):
                letters.append(_LETTER_NAMES[key])
                pos += len(key)
                break
        else:
            break
    if len(letters) >= 2:
        return "".join(letters), token[pos:]
    return "", token


def _phones(value: object) -> list[list[str]]:
    """Per-token phone lists (classes upper-case, vowels lower-case)."""
    result = []
    for tok in tokens(value):
        if is_hangul(tok):
            letters, rest = split_letter_names(tok)
            phones = _english_phones(letters) + _hangul_phones(rest) if letters else _hangul_phones(tok)
        else:
            phones = _english_phones(tok)
        result.append(phones)
    return result


def _consonant_string(phones: list[str]) -> str:
    s = "".join(p for p in phones if not p.islower())
    return re.sub(r"([^0-9])\1+", r"\1", s)


def _vowel_string(phones: list[str]) -> str:
    s = "".join(p.upper() if p.islower() else p for p in phones)
    s = s.replace("EI", "A").replace("AI", "I").replace("OU", "O")
    return re.sub(r"([^0-9])\1+", r"\1", s)


@lru_cache(maxsize=None)
def name_reps(value: str) -> tuple[str, str, str, tuple[str, ...], frozenset[str], tuple[str, ...]]:
    """(latin_key, consonant_string, vowel_string, token_consonant_strings, digits, latin_tokens)."""
    phones = _phones(value)
    cons_tokens = tuple(_consonant_string(p) for p in phones)
    cons = "".join(cons_tokens)
    cons = re.sub(r"([^0-9])\1+", r"\1", cons)
    vow = "".join(_vowel_string(p) for p in phones)
    digits = frozenset(re.findall(r"\d+", _clean(value)))
    latin_tokens = tuple(latin_key(t) for t in tokens(value))
    return latin_key(value), cons, vow, cons_tokens, digits, latin_tokens


# --------------------------------------------------------------------------- #
# Similarity helpers
# --------------------------------------------------------------------------- #


def grams(value: str, width: int) -> frozenset[str]:
    if not value:
        return frozenset()
    padded = f"^{value}$"
    if len(padded) < width:
        return frozenset((padded,))
    return frozenset(padded[i: i + width] for i in range(len(padded) - width + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@lru_cache(maxsize=2_000_000)
def lev_sim(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / len(a)


@lru_cache(maxsize=2_000_000)
def osa_sim(a: str, b: str) -> float:
    """1 - normalised optimal-string-alignment distance (transposition costs 1)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return 1.0 - d[la][lb] / max(la, lb)


def positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def relative_gap(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 2.0
    return min(abs(left - right) / max(left, right, 1.0), 2.0)


# --------------------------------------------------------------------------- #
# Hard-coded answers memorised from train.csv
# --------------------------------------------------------------------------- #


def normalize_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(ch for ch in value if ch.isalnum())


class AnswerBook:
    """Exact-key lookup of training labels (call sign, then vessel name)."""

    def __init__(self, train: pd.DataFrame) -> None:
        self.by_call_sign: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        conflicts = 0
        for row in train.to_dict("records"):
            target = str(row["target_vessel_id"])
            cs = row.get("call_sign")
            if isinstance(cs, str) and cs.strip():
                prev = self.by_call_sign.setdefault(cs.strip(), target)
                conflicts += int(prev != target)
            name = normalize_name(row.get("vessel_name"))
            if name:
                prev = self.by_name.setdefault(name, target)
                conflicts += int(prev != target)
        self.conflicts = conflicts

    def lookup(self, row: dict) -> str | None:
        cs = row.get("call_sign")
        if isinstance(cs, str) and cs.strip() in self.by_call_sign:
            return self.by_call_sign[cs.strip()]
        name = normalize_name(row.get("vessel_name"))
        if name and name in self.by_name:
            return self.by_name[name]
        return None


# --------------------------------------------------------------------------- #
# Catalog index (numpy posting lists, bincount scoring)
# --------------------------------------------------------------------------- #


class _GramIndex:
    def __init__(self, strings: list[str], width: int) -> None:
        self.width = width
        self.sets = [grams(s, width) for s in strings]
        self.sizes = np.asarray([len(s) for s in self.sets], dtype=np.float32)
        postings: dict[str, list[int]] = defaultdict(list)
        for i, gs in enumerate(self.sets):
            for g in gs:
                postings[g].append(i)
        self.postings = {g: np.asarray(ix, dtype=np.int32) for g, ix in postings.items()}
        self.n = len(strings)

    def counts(self, query: str) -> tuple[np.ndarray, int]:
        qg = grams(query, self.width)
        arrays = [self.postings[g] for g in qg if g in self.postings]
        if not arrays:
            return np.zeros(self.n, dtype=np.int32), len(qg)
        return np.bincount(np.concatenate(arrays), minlength=self.n), len(qg)

    def scores(self, query: str) -> np.ndarray:
        counts, qn = self.counts(query)
        if qn == 0:
            return np.zeros(self.n, dtype=np.float32)
        return counts / (qn + self.sizes - counts)


class CatalogIndex:
    """Lookup structures built only from the official public catalog."""

    def __init__(self, catalog: pd.DataFrame) -> None:
        needed = {"vessel_id", "vessel_name", "gross_tonnage"}
        if not needed.issubset(catalog.columns):
            raise ValueError(f"Catalog is missing: {sorted(needed - set(catalog.columns))}")
        self.catalog = catalog.reset_index(drop=True)
        if not self.catalog["vessel_id"].is_unique:
            raise ValueError("vessel_id must be unique in the catalog")
        self.ids = self.catalog["vessel_id"].astype(str).tolist()
        self.id_to_index = {vid: i for i, vid in enumerate(self.ids)}
        raw_names = ["" if not isinstance(v, str) else v for v in self.catalog["vessel_name"]]
        reps = [name_reps(v) for v in raw_names]
        self.latin = [r[0] for r in reps]
        self.cons = [r[1] for r in reps]
        self.vow = [r[2] for r in reps]
        self.cons_tokens = [r[3] for r in reps]
        self.digits = [r[4] for r in reps]
        self.latin_tokens = [r[5] for r in reps]
        self.token_sets = [frozenset(tokens(v)) for v in raw_names]
        self.flags = self.catalog["flag_name"].astype(str).tolist() if "flag_name" in self.catalog else [""] * len(self.ids)
        self.types = self.catalog["vessel_type"].astype(str).tolist() if "vessel_type" in self.catalog else [""] * len(self.ids)
        self.gt = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)

        self.latin_bi = _GramIndex(self.latin, 2)
        self.latin_tri = _GramIndex(self.latin, 3)
        self.cons_bi = _GramIndex(self.cons, 2)
        self.cons_tri = _GramIndex(self.cons, 3)
        self.vow_bi = _GramIndex(self.vow, 2)
        self.exact_latin: dict[str, list[int]] = defaultdict(list)
        self.exact_cons: dict[str, list[int]] = defaultdict(list)
        self.exact_vow: dict[str, list[int]] = defaultdict(list)
        for i, (l, c, v) in enumerate(zip(self.latin, self.cons, self.vow)):
            if l:
                self.exact_latin[l].append(i)
            if c:
                self.exact_cons[c].append(i)
            if v:
                self.exact_vow[v].append(i)

        valid = np.flatnonzero(np.isfinite(self.gt) & (self.gt > 0))
        order = np.argsort(self.gt[valid], kind="mergesort")
        self.sorted_gt_indices = valid[order]
        self.sorted_gt_values = self.gt[self.sorted_gt_indices]

    def nearest_gt(self, value: float | None, count: int, max_ties: int = 60) -> list[int]:
        """All catalog rows with exactly this tonnage (up to ``max_ties``) plus the
        ``count`` nearest other rows."""
        if value is None or len(self.sorted_gt_values) == 0:
            return []
        lo_eq = int(np.searchsorted(self.sorted_gt_values, value, side="left"))
        hi_eq = int(np.searchsorted(self.sorted_gt_values, value, side="right"))
        exact = [int(i) for i in self.sorted_gt_indices[lo_eq:hi_eq][:max_ties]]
        lo, hi = max(0, lo_eq - count), min(len(self.sorted_gt_values), hi_eq + count)
        others = np.concatenate([self.sorted_gt_indices[lo:lo_eq], self.sorted_gt_indices[hi_eq:hi]])
        order = np.argsort(np.abs(self.gt[others] - value), kind="mergesort")[:count]
        return exact + [int(i) for i in others[order]]

    def gt_window(self, value: float | None, tolerance: float) -> np.ndarray:
        if value is None or len(self.sorted_gt_values) == 0:
            return np.empty(0, dtype=int)
        lo = np.searchsorted(self.sorted_gt_values, value * (1 - tolerance), side="left")
        hi = np.searchsorted(self.sorted_gt_values, value * (1 + tolerance), side="right")
        return self.sorted_gt_indices[lo:hi]


# --------------------------------------------------------------------------- #
# Ranker
# --------------------------------------------------------------------------- #


def _topk(scores: np.ndarray, k: int, subset: np.ndarray | None = None) -> list[int]:
    if subset is not None:
        if len(subset) == 0:
            return []
        local = scores[subset]
        k = min(k, len(local))
        part = np.argpartition(-local, k - 1)[:k]
        part = part[np.argsort(-local[part], kind="mergesort")]
        return [int(subset[i]) for i in part if local[i] > 0]
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    part = part[np.argsort(-scores[part], kind="mergesort")]
    return [int(i) for i in part if scores[i] > 0]


class Ranker:
    """Candidate retrieval + pairwise features, parameterised by train statistics."""

    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame) -> None:
        merged = train.merge(catalog, left_on="target_vessel_id", right_on="vessel_id",
                             suffixes=("_q", "_c"))
        self.flag_prob = self._conditional(merged, "flag_name_q", "flag_name_c")
        self.type_prob = self._conditional(merged, "vessel_type_q", "vessel_type_c")

    @staticmethod
    def _conditional(merged: pd.DataFrame, q: str, c: str) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        sub = merged.dropna(subset=[q, c])
        for key, group in sub.groupby(q):
            counts = group[c].value_counts()
            table[str(key)] = (counts / counts.sum()).to_dict()
        return table

    # ---- retrieval ------------------------------------------------------- #

    def candidates(self, row: dict, index: CatalogIndex, k_name: int = 30,
                   k_window: int = 30, k_near: int = 10, tolerance: float = 0.06,
                   max_candidates: int = 160) -> list[tuple[int, int]]:
        """Return (catalog_index, source_bits): 1 = name, 2 = tonnage, 4 = exact."""
        name = row.get("vessel_name")
        latin, cons, vow, _, _, _ = name_reps(name if isinstance(name, str) else "")
        gross = positive_float(row.get("gross_tonnage"))
        intl = positive_float(row.get("international_gross_tonnage"))
        values = sorted({gross, intl} - {None})
        output: dict[int, int] = {}

        def gap_of(i: int) -> float:
            cgt = positive_float(index.gt[i])
            return min((relative_gap(v, cgt) for v in values), default=2.0)

        def add(indices, bit):
            for i in indices:
                if i in output:
                    output[i] |= bit
                elif len(output) < max_candidates:
                    output[i] = bit

        def by_gap(indices, limit):
            return sorted(indices, key=gap_of)[:limit]

        if latin:
            add(by_gap(index.exact_latin.get(latin, []), k_name), 4 | 1)
        if cons:
            add(by_gap(index.exact_cons.get(cons, []), k_name), 4 | 1)
        if vow:
            add(by_gap(index.exact_vow.get(vow, []), k_name), 4 | 1)
        name_scores = None
        if latin or cons:
            parts = []
            if latin:
                parts.append(index.latin_bi.scores(latin))
                parts.append(index.latin_tri.scores(latin))
            if cons:
                parts.append(index.cons_bi.scores(cons))
                parts.append(index.cons_tri.scores(cons))
            if vow:
                parts.append(index.vow_bi.scores(vow))
            for p in parts:
                add(_topk(p, k_name), 1)
            name_scores = np.maximum.reduce(parts)
            add(_topk(name_scores, k_name), 1)
        for value in values:
            add(index.nearest_gt(value, k_near), 2)
            window = index.gt_window(value, tolerance)
            if len(window) and name_scores is not None:
                add(_topk(name_scores, k_window, window), 2)
        if not output:
            output[0] = 0
        return list(output.items())

    # ---- features -------------------------------------------------------- #

    def features(self, row: dict, index: CatalogIndex, cand: int, bits: int) -> list[float]:
        name = row.get("vessel_name")
        qlatin, qcons, qvow, qtoks, qdigits, qlt = name_reps(name if isinstance(name, str) else "")
        clatin, ccons, cvow = index.latin[cand], index.cons[cand], index.vow[cand]
        ctoks, cdigits, clt = index.cons_tokens[cand], index.digits[cand], index.latin_tokens[cand]
        gross = positive_float(row.get("gross_tonnage"))
        intl = positive_float(row.get("international_gross_tonnage"))
        cgt = positive_float(index.gt[cand])
        g_gap, i_gap = relative_gap(gross, cgt), relative_gap(intl, cgt)
        prefix = 0
        for a, b in zip(qlatin, clatin):
            if a != b:
                break
            prefix += 1
        qflag = row.get("flag_name")
        flag_missing = not (isinstance(qflag, str) and qflag.strip())
        flag_prob = 0.0 if flag_missing else self.flag_prob.get(qflag, {}).get(index.flags[cand], 0.0)
        qtype = row.get("vessel_type")
        type_prob = self.type_prob.get(qtype, {}).get(index.types[cand], 0.0) if isinstance(qtype, str) else 0.0
        if qtoks and ctoks:
            best = [max(lev_sim(a, b) for b in ctoks) for a in qtoks if a]
            tok_mean = float(np.mean(best)) if best else 0.0
            tok_min = float(min(best)) if best else 0.0
            tok_max = float(max(best)) if best else 0.0
        else:
            tok_mean = tok_min = tok_max = 0.0
        if qlt and clt:
            lt_best = [max(osa_sim(a, b) for b in clt) for a in qlt if a]
            latin_token_exact = float(sum(1 for a in qlt if a and a in clt))
            latin_token_osa_max = float(max(lt_best)) if lt_best else 0.0
            # the distinctive ship name is usually the last word (company prefix first)
            q_words = [t for t in qlt if t and not t.isdigit()] or list(qlt)
            c_words = [t for t in clt if t and not t.isdigit()] or list(clt)
            first_token_exact = float(q_words[0] == c_words[0])
            last_token_exact = float(q_words[-1] == c_words[-1])
            last_token_osa = osa_sim(q_words[-1], c_words[-1])
            qc_words = [t for t in qtoks if t and not t.isdigit()] or list(qtoks)
            cc_words = [t for t in ctoks if t and not t.isdigit()] or list(ctoks)
            last_token_cons_lev = lev_sim(qc_words[-1], cc_words[-1]) if qc_words and cc_words else 0.0
        else:
            latin_token_exact = latin_token_osa_max = 0.0
            first_token_exact = last_token_exact = last_token_osa = last_token_cons_lev = 0.0
        qtok_set = frozenset(tokens(name))
        if qdigits or cdigits:
            digits_match = float(qdigits == cdigits)
        else:
            digits_match = 0.5
        return [
            float(bool(qlatin) and qlatin == clatin),
            lev_sim(qlatin, clatin),
            osa_sim(qlatin, clatin),
            jaccard(grams(qlatin, 2), index.latin_bi.sets[cand]),
            jaccard(grams(qlatin, 3), index.latin_tri.sets[cand]),
            prefix / max(len(qlatin), len(clatin), 1),
            min(len(qlatin), len(clatin)) / max(len(qlatin), len(clatin), 1),
            float(bool(qcons) and qcons == ccons),
            lev_sim(qcons, ccons),
            jaccard(grams(qcons, 2), index.cons_bi.sets[cand]),
            jaccard(grams(qcons, 3), index.cons_tri.sets[cand]),
            min(len(qcons), len(ccons)) / max(len(qcons), len(ccons), 1),
            float(bool(qvow) and qvow == cvow),
            lev_sim(qvow, cvow),
            osa_sim(qvow, cvow),
            jaccard(grams(qvow, 2), index.vow_bi.sets[cand]),
            tok_mean,
            tok_min,
            tok_max,
            float(abs(len(qtoks) - len(ctoks))),
            jaccard(qtok_set, index.token_sets[cand]),
            latin_token_exact,
            latin_token_osa_max,
            first_token_exact,
            last_token_exact,
            last_token_osa,
            last_token_cons_lev,
            digits_match,
            float(bool(qdigits)),
            g_gap,
            i_gap,
            min(g_gap, i_gap),
            float(g_gap == 0.0 or i_gap == 0.0),
            float(gross is None and intl is None),
            float(cgt is None),
            flag_prob,
            float(flag_missing),
            type_prob,
            float(is_hangul(name)),
            float(bits & 1 > 0),
            float(bits & 2 > 0),
            float(bits & 4 > 0),
        ]


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #


_POOL_STATE: dict = {}


def _pool_init(ranker, index):
    _POOL_STATE["ranker"] = ranker
    _POOL_STATE["index"] = index


def _train_row(row: dict):
    ranker, index = _POOL_STATE["ranker"], _POOL_STATE["index"]
    true_idx = index.id_to_index[str(row["target_vessel_id"])]
    cands = ranker.candidates(row, index)
    hit = true_idx in dict(cands)
    if not hit:
        cands = cands + [(true_idx, 0)]
    X = pair_matrix(ranker, row, index, cands)
    y = np.asarray([int(i == true_idx) for i, _ in cands], dtype=np.int8)
    return X, y, hit


def _query_row(row: dict):
    ranker, index = _POOL_STATE["ranker"], _POOL_STATE["index"]
    cands = ranker.candidates(row, index)
    return pair_matrix(ranker, row, index, cands), [i for i, _ in cands]


def _parallel_map(func, rows: list[dict], ranker, index, n_jobs: int | None):
    n_jobs = n_jobs or int(os.environ.get("MATCHER_JOBS", os.cpu_count() or 1))
    if n_jobs <= 1 or len(rows) < 64:
        _pool_init(ranker, index)
        return [func(r) for r in rows]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_jobs, initializer=_pool_init, initargs=(ranker, index)) as pool:
        return pool.map(func, rows, chunksize=max(1, len(rows) // (n_jobs * 8)))


def _row_matrices(train: pd.DataFrame, ranker: "Ranker", index: "CatalogIndex",
                  n_jobs: int | None = None):
    """Per train row: (matrix over candidates, label vector). Also returns candidate recall."""
    out = _parallel_map(_train_row, train.to_dict("records"), ranker, index, n_jobs)
    rows = [(X, y) for X, y, _ in out]
    retrieved = sum(int(h) for _, _, h in out)
    return rows, retrieved / max(len(train), 1)


def _select(rows, max_negatives: int, scorer=None, keep_first: int = 15):
    Xs, ys = [], []
    for X, y in rows:
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if scorer is not None and len(neg) > max_negatives:
            scores = scorer(X[neg])
            order = np.argsort(-scores, kind="mergesort")
            chosen = list(range(min(keep_first, len(neg))))
            for j in order:
                if len(chosen) >= max_negatives:
                    break
                if int(j) not in chosen:
                    chosen.append(int(j))
            neg = neg[chosen]
        else:
            neg = neg[:max_negatives]
        keep = np.concatenate([pos, neg])
        Xs.append(X[keep])
        ys.append(y[keep])
    return np.vstack(Xs), np.concatenate(ys)


# Features that do not look at the vessel name: used by the fallback scorer for
# rows whose name matches nothing in the catalog (renamed vessels).
NONAME_FEATURES = tuple(f for f in FEATURE_NAMES if any(f.startswith(k) for k in (
    "gross_gap", "intl_gap", "best_gap", "gt_exact", "query_gt_missing", "cat_gt_missing",
    "flag_prob", "flag_missing", "type_prob", "query_is_hangul", "src_tonnage", "n_candidates",
    "is_min_gap", "is_max_flag_prob", "is_max_type_prob")))
_NONAME_IDX = np.asarray([FEATURE_NAMES.index(f) for f in NONAME_FEATURES])


def _fit_models(rows, n_estimators: int = 300, random_state: int = 42,
                use_rf: bool = False, max_negatives: int = 40) -> dict:
    """Train the scorers on cached per-row candidate matrices."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    X_all = np.vstack([X for X, _ in rows])
    y_all = np.concatenate([y for _, y in rows])

    def booster(seed: int):
        return HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )

    models = {"hgb": booster(random_state).fit(X_all, y_all)}
    models["noname"] = booster(random_state + 1).fit(X_all[:, _NONAME_IDX], y_all)
    if use_rf:
        X_h, y_h = _select(rows, max_negatives, scorer=lambda M: models["hgb"].predict_proba(M)[:, 1])
        models["rf"] = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state + 2,
        ).fit(X_h, y_h)
    return models


def _score(models, X: np.ndarray, fallback_threshold: float = 0.02) -> np.ndarray:
    """Raw ranking scores for the candidates of ONE query row.

    The main booster's decision function is used (no probability saturation).
    When even the best candidate is implausible by name (max probability below
    ``fallback_threshold``) the name-free scorer decides instead.
    """
    if isinstance(models, (list, tuple)):          # legacy bundles
        return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    hgb = models["hgb"]
    score = hgb.decision_function(X)
    if "rf" in models:
        rf_p = np.clip(models["rf"].predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        score = 0.5 * score + 0.5 * np.log(rf_p / (1 - rf_p))
    if "noname" in models:
        best_p = 1.0 / (1.0 + np.exp(-score.max()))
        if best_p < fallback_threshold:
            score = models["noname"].decision_function(X[:, _NONAME_IDX])
    return score


def fit(train: pd.DataFrame, catalog: pd.DataFrame, n_estimators: int = 300,
        random_state: int = 42, index: CatalogIndex | None = None,
        use_rf: bool = False, n_jobs: int | None = None, **_ignored) -> dict:
    if "target_vessel_id" not in train.columns:
        raise ValueError("Training data needs target_vessel_id")
    index = index or CatalogIndex(catalog)
    unknown = set(train["target_vessel_id"].astype(str)) - set(index.id_to_index)
    if unknown:
        raise ValueError(f"Training targets missing from catalog: {sorted(unknown)[:5]}")
    answers = AnswerBook(train)
    ranker = Ranker(train, catalog)
    rows, recall = _row_matrices(train, ranker, index, n_jobs=n_jobs)
    models = _fit_models(rows, n_estimators=n_estimators, random_state=random_state, use_rf=use_rf)
    return {
        "version": VERSION,
        "feature_names": FEATURE_NAMES,
        "answers": answers,
        "ranker": ranker,
        "model": models,
        "candidate_recall": recall,
        "n_train_rows": int(len(train)),
    }


def predict_ids(bundle: dict, queries: pd.DataFrame, index: CatalogIndex,
                use_answers: bool = True, batch_size: int = 1024,
                n_jobs: int | None = None) -> tuple[list[str], list[str]]:
    """Predict one catalog ID per row. Returns (ids, sources)."""
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model and feature code versions do not match")
    answers: AnswerBook = bundle["answers"]
    ranker: Ranker = bundle["ranker"]
    models = bundle["model"]
    ids: list[str | None] = [None] * len(queries)
    sources: list[str] = ["ranker"] * len(queries)
    records = queries.to_dict("records")
    pending: list[int] = []
    for pos, row in enumerate(records):
        hit = answers.lookup(row) if use_answers else None
        if hit is not None and hit in index.id_to_index:
            ids[pos] = hit
            sources[pos] = "answer_book"
        else:
            pending.append(pos)
    for start in range(0, len(pending), batch_size):
        chunk = pending[start: start + batch_size]
        results = _parallel_map(_query_row, [records[p] for p in chunk], ranker, index, n_jobs)
        for pos, (X, cand_idx) in zip(chunk, results):
            local = _score(models, X)
            ids[pos] = index.ids[cand_idx[int(np.argmax(local))]]
    return ids, sources  # type: ignore[return-value]


def weighted_accuracy(truth: pd.DataFrame, predicted: list[str]) -> dict[str, float]:
    """Competition-style metric: 0.7 * Hangul-name accuracy + 0.3 * other."""
    if len(truth) != len(predicted):
        raise ValueError("Prediction count differs from validation data")
    language = truth["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    correct = truth["target_vessel_id"].astype(str).to_numpy() == np.asarray(predicted)
    hangul = float(correct[language].mean()) if language.any() else float("nan")
    other = float(correct[~language].mean()) if (~language).any() else float("nan")
    return {
        "accuracy": float(correct.mean()),
        "hangul_accuracy": hangul,
        "non_hangul_accuracy": other,
        "weighted_accuracy": 0.7 * hangul + 0.3 * other,
    }
