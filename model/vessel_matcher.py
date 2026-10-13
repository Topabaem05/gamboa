"""Vessel entity matcher for the DACON vessel-identification competition.

Two layers:

1. ``AnswerBook`` — hard-coded answers memorised from the official ``train.csv``.
   ``call_sign`` and ``vessel_name`` each map one-to-one onto ``target_vessel_id``
   in the training data, so an exact lookup reproduces every training label.
2. ``Ranker`` — candidate retrieval over ``vessel_catalog.csv`` plus a
   RandomForest pairwise scorer, used for rows the answer book has never seen.

Every query row is processed on its own; no feature looks at other evaluation
rows or at the evaluation distribution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from heapq import nlargest
import re
import unicodedata

import numpy as np
import pandas as pd

VERSION = "hardcoded-lookup+rf-v1"

FEATURE_NAMES = (
    "name_exact",
    "name_bigram_jaccard",
    "name_trigram_jaccard",
    "name_prefix_ratio",
    "name_length_ratio",
    "skeleton_exact",
    "skeleton_bigram_jaccard",
    "token_overlap",
    "gross_tonnage_gap",
    "international_gross_tonnage_gap",
    "best_tonnage_gap",
    "query_tonnage_missing",
    "catalog_tonnage_missing",
    "flag_prob",
    "flag_missing",
    "type_prob",
    "query_is_hangul",
    "from_name_index",
    "from_tonnage_index",
)

# --------------------------------------------------------------------------- #
# Text normalisation and Hangul romanisation
# --------------------------------------------------------------------------- #

_INITIALS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "",
             "j", "jj", "ch", "k", "t", "p", "h"]
_MEDIALS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
            "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
_FINALS = ["", "k", "k", "k", "n", "n", "n", "t", "l", "lk", "lm", "lp", "ls",
           "lt", "lp", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t",
           "p", "t"]


def is_hangul(value: object) -> bool:
    return isinstance(value, str) and any("가" <= ch <= "힣" for ch in value)


def romanize(text: str) -> str:
    """Revised-Romanisation style transliteration of Hangul syllables."""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            code -= 0xAC00
            out.append(_INITIALS[code // 588])
            out.append(_MEDIALS[(code % 588) // 28])
            out.append(_FINALS[code % 28])
        else:
            out.append(ch)
    return "".join(out)


def normalize_name(value: object) -> str:
    """Uppercase alphanumerics only; Hangul is kept (used as an exact key)."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(ch for ch in value if ch.isalnum())


def latin_key(value: object) -> str:
    """Latin-only comparison key: Hangul is romanised, ``제N`` becomes ``NON``."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"제\s*(\d+)", r"NO\1", value)
    value = re.sub(r"\s*호$", "", value.strip())  # trailing 호 (ship suffix)
    value = romanize(value).upper()
    return "".join(ch for ch in value if ch.isalnum())


_SKELETON_MAP = str.maketrans({
    "C": "K", "Q": "K", "G": "K", "X": "K",
    "V": "B", "F": "P", "W": "B",
    "L": "R", "Z": "J", "D": "T", "S": "S",
})


def skeleton(latin: str) -> str:
    """Consonant skeleton with similar consonants merged; vowels dropped."""
    s = latin.translate(_SKELETON_MAP)
    s = s.replace("PH", "P").replace("TH", "T").replace("SH", "S").replace("KH", "K")
    s = re.sub(r"[AEIOUYH]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def grams(value: str, width: int) -> frozenset[str]:
    if len(value) < width:
        return frozenset((value,)) if value else frozenset()
    return frozenset(value[i: i + width] for i in range(len(value) - width + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


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


def tokens(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    value = unicodedata.normalize("NFKC", value).upper()
    return [t for t in re.split(r"[^0-9A-Z가-힣]+", value) if t]


# --------------------------------------------------------------------------- #
# Hard-coded answers memorised from train.csv
# --------------------------------------------------------------------------- #


class AnswerBook:
    """Exact-key lookup of training labels (call sign, then vessel name)."""

    def __init__(self, train: pd.DataFrame) -> None:
        self.by_call_sign: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        self.by_name_tonnage: dict[tuple[str, float], str] = {}
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
                gt = positive_float(row.get("gross_tonnage"))
                if gt is not None:
                    self.by_name_tonnage.setdefault((name, gt), target)
        self.conflicts = conflicts

    def lookup(self, row: dict) -> str | None:
        cs = row.get("call_sign")
        if isinstance(cs, str) and cs.strip() in self.by_call_sign:
            return self.by_call_sign[cs.strip()]
        name = normalize_name(row.get("vessel_name"))
        if name and name in self.by_name:
            return self.by_name[name]
        return None

    def __len__(self) -> int:
        return len(self.by_call_sign) + len(self.by_name)


# --------------------------------------------------------------------------- #
# Catalog index and pairwise features
# --------------------------------------------------------------------------- #


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
        self.names = [latin_key(v) for v in self.catalog["vessel_name"]]
        self.skeletons = [skeleton(n) for n in self.names]
        self.bigrams = [grams(n, 2) for n in self.names]
        self.trigrams = [grams(n, 3) for n in self.names]
        self.skel_bigrams = [grams(s, 2) for s in self.skeletons]
        self.token_sets = [frozenset(tokens(v)) for v in self.catalog["vessel_name"]]
        self.flags = self.catalog.get("flag_name", pd.Series([""] * len(self.catalog))).astype(str).tolist()
        self.types = self.catalog.get("vessel_type", pd.Series([""] * len(self.catalog))).astype(str).tolist()
        self.gt = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)

        self.exact_name: dict[str, list[int]] = defaultdict(list)
        self.trigram_postings: dict[str, list[int]] = defaultdict(list)
        self.skel_postings: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(self.names):
            if name:
                self.exact_name[name].append(i)
            for g in self.trigrams[i]:
                self.trigram_postings[g].append(i)
            for g in self.skel_bigrams[i]:
                self.skel_postings[g].append(i)

        valid = np.flatnonzero(np.isfinite(self.gt) & (self.gt > 0))
        order = np.argsort(self.gt[valid], kind="mergesort")
        self.sorted_gt_indices = valid[order]
        self.sorted_gt_values = self.gt[self.sorted_gt_indices]

    def nearest_gt(self, value: float | None, count: int) -> list[int]:
        if value is None or len(self.sorted_gt_values) == 0:
            return []
        pos = int(np.searchsorted(self.sorted_gt_values, value))
        left, right = pos - 1, pos
        found: list[int] = []
        n = len(self.sorted_gt_values)
        while len(found) < count and (left >= 0 or right < n):
            if right >= n:
                take_left = True
            elif left < 0:
                take_left = False
            else:
                take_left = abs(self.sorted_gt_values[left] - value) <= abs(self.sorted_gt_values[right] - value)
            found.append(int(self.sorted_gt_indices[left if take_left else right]))
            left -= int(take_left)
            right += int(not take_left)
        return found

    def gt_window(self, value: float | None, tolerance: float) -> np.ndarray:
        """Catalog rows whose gross tonnage lies within ``tolerance`` (relative)."""
        if value is None or len(self.sorted_gt_values) == 0:
            return np.empty(0, dtype=int)
        lo = np.searchsorted(self.sorted_gt_values, value * (1 - tolerance), side="left")
        hi = np.searchsorted(self.sorted_gt_values, value * (1 + tolerance), side="right")
        return self.sorted_gt_indices[lo:hi]


class Ranker:
    """Candidate retrieval + pairwise features, parameterised by train statistics."""

    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame) -> None:
        merged = train.merge(catalog, left_on="target_vessel_id", right_on="vessel_id",
                             suffixes=("_q", "_c"))
        self.flag_prob = self._conditional(merged, "flag_name_q", "flag_name_c")
        self.type_prob = self._conditional(merged, "vessel_type_q", "vessel_type_c")
        self.token_map = self._token_dictionary(merged)

    @staticmethod
    def _conditional(merged: pd.DataFrame, q: str, c: str) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        sub = merged.dropna(subset=[q, c])
        for key, group in sub.groupby(q):
            counts = group[c].value_counts()
            table[str(key)] = (counts / counts.sum()).to_dict()
        return table

    @staticmethod
    def _token_dictionary(merged: pd.DataFrame) -> dict[str, str]:
        """Hangul token -> Latin token learned from aligned train/catalog names."""
        pair_counts: Counter[tuple[str, str]] = Counter()
        for q, c in zip(merged["vessel_name_q"], merged["vessel_name_c"]):
            if not is_hangul(q):
                continue
            qt, ct = tokens(q), tokens(c)
            if len(qt) == len(ct):
                for a, b in zip(qt, ct):
                    if is_hangul(a):
                        pair_counts[(a, b)] += 1
            elif len(qt) == 1 and ct:
                pair_counts[(qt[0], "".join(ct))] += 1
        best: dict[str, tuple[str, int]] = {}
        for (a, b), n in pair_counts.items():
            if a not in best or n > best[a][1]:
                best[a] = (b, n)
        return {a: b for a, (b, _) in best.items()}

    # ---- query side ----------------------------------------------------- #

    def query_key(self, name: object) -> str:
        """Latin key for a query name, using the learned token dictionary first."""
        if not is_hangul(name):
            return latin_key(name)
        parts = []
        for tok in tokens(name):
            parts.append(self.token_map.get(tok, tok))
        return latin_key(" ".join(parts))

    def candidates(self, row: dict, index: CatalogIndex, name_limit: int = 20,
                   tonnage_limit: int = 12, max_candidates: int = 48) -> list[tuple[int, int]]:
        """Return (catalog_index, source_bits) for one query row."""
        qkey = self.query_key(row.get("vessel_name"))
        gross = positive_float(row.get("gross_tonnage"))
        intl = positive_float(row.get("international_gross_tonnage"))
        hangul = is_hangul(row.get("vessel_name"))
        output: dict[int, int] = {}

        def add(indices, bit):
            for i in indices:
                if len(output) >= max_candidates and i not in output:
                    continue
                output[i] = output.get(i, 0) | bit

        if qkey:
            add(index.exact_name.get(qkey, [])[:name_limit], 1)
            counts: Counter[int] = Counter()
            for g in grams(qkey, 3):
                counts.update(index.trigram_postings.get(g, ()))
            add(nlargest(name_limit, counts, key=lambda i: (counts[i], -i)), 1)
            if hangul:
                sk = skeleton(qkey)
                counts = Counter()
                for g in grams(sk, 2):
                    counts.update(index.skel_postings.get(g, ()))
                add(nlargest(name_limit, counts, key=lambda i: (counts[i], -i)), 1)

        for value in {gross, intl} - {None}:
            add(index.nearest_gt(value, tonnage_limit), 2)
            if hangul:
                # Within a tonnage window pick the best name matches.
                window = index.gt_window(value, 0.08)
                if len(window):
                    qb = grams(qkey, 2)
                    scored = nlargest(name_limit, window,
                                      key=lambda i: jaccard(qb, index.bigrams[i]))
                    add(scored, 2)
        if not output:
            output[0] = 0
        return list(output.items())

    def features(self, row: dict, index: CatalogIndex, cand: int, bits: int) -> list[float]:
        qkey = self.query_key(row.get("vessel_name"))
        ckey = index.names[cand]
        qsk = skeleton(qkey)
        csk = index.skeletons[cand]
        gross = positive_float(row.get("gross_tonnage"))
        intl = positive_float(row.get("international_gross_tonnage"))
        cgt = positive_float(index.gt[cand])
        g_gap = relative_gap(gross, cgt)
        i_gap = relative_gap(intl, cgt)
        prefix = 0
        for a, b in zip(qkey, ckey):
            if a != b:
                break
            prefix += 1
        qflag = row.get("flag_name")
        flag_missing = not (isinstance(qflag, str) and qflag.strip())
        flag_prob = 0.0 if flag_missing else self.flag_prob.get(qflag, {}).get(index.flags[cand], 0.0)
        qtype = row.get("vessel_type")
        type_prob = self.type_prob.get(qtype, {}).get(index.types[cand], 0.0) if isinstance(qtype, str) else 0.0
        qtok = frozenset(self.token_map.get(t, t) for t in tokens(row.get("vessel_name")))
        ctok = index.token_sets[cand]
        return [
            float(bool(qkey) and qkey == ckey),
            jaccard(grams(qkey, 2), index.bigrams[cand]),
            jaccard(grams(qkey, 3), index.trigrams[cand]),
            prefix / max(len(qkey), len(ckey), 1),
            min(len(qkey), len(ckey)) / max(len(qkey), len(ckey), 1),
            float(bool(qsk) and qsk == csk),
            jaccard(grams(qsk, 2), index.skel_bigrams[cand]),
            jaccard(qtok, ctok),
            g_gap,
            i_gap,
            min(g_gap, i_gap),
            float(gross is None and intl is None),
            float(cgt is None),
            flag_prob,
            float(flag_missing),
            type_prob,
            float(is_hangul(row.get("vessel_name"))),
            float(bits & 1 > 0),
            float(bits & 2 > 0),
        ]


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #


def make_training_pairs(train: pd.DataFrame, ranker: Ranker, index: CatalogIndex,
                        max_negatives: int = 24) -> tuple[np.ndarray, np.ndarray, float]:
    feats: list[list[float]] = []
    labels: list[int] = []
    retrieved = 0
    for row in train.to_dict("records"):
        true_idx = index.id_to_index[str(row["target_vessel_id"])]
        cands = ranker.candidates(row, index)
        cand_map = dict(cands)
        retrieved += int(true_idx in cand_map)
        negatives = [(i, b) for i, b in cands if i != true_idx][:max_negatives]
        for i, b in [(true_idx, cand_map.get(true_idx, 0))] + negatives:
            feats.append(ranker.features(row, index, i, b))
            labels.append(int(i == true_idx))
    return (np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int8),
            retrieved / max(len(train), 1))


def fit(train: pd.DataFrame, catalog: pd.DataFrame, n_estimators: int = 200,
        random_state: int = 42) -> dict:
    """Build the full bundle: hard-coded answer book + fallback ranker."""
    from sklearn.ensemble import RandomForestClassifier

    if "target_vessel_id" not in train.columns:
        raise ValueError("Training data needs target_vessel_id")
    index = CatalogIndex(catalog)
    unknown = set(train["target_vessel_id"].astype(str)) - set(index.id_to_index)
    if unknown:
        raise ValueError(f"Training targets missing from catalog: {sorted(unknown)[:5]}")
    answers = AnswerBook(train)
    ranker = Ranker(train, catalog)
    X, y, recall = make_training_pairs(train, ranker, index)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=16, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
    )
    model.fit(X, y)
    return {
        "version": VERSION,
        "feature_names": FEATURE_NAMES,
        "answers": answers,
        "ranker": ranker,
        "model": model,
        "candidate_recall": recall,
        "n_train_rows": int(len(train)),
    }


def predict_ids(bundle: dict, queries: pd.DataFrame, index: CatalogIndex,
                use_answers: bool = True, batch_size: int = 256) -> tuple[list[str], list[str]]:
    """Predict one catalog ID per row. Returns (ids, source) with source in
    {"answer_book", "ranker"}."""
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model and feature code versions do not match")
    answers: AnswerBook = bundle["answers"]
    ranker: Ranker = bundle["ranker"]
    model = bundle["model"]
    ids: list[str] = []
    sources: list[str] = []
    for start in range(0, len(queries), batch_size):
        rows = queries.iloc[start: start + batch_size].to_dict("records")
        pending: list[tuple[int, list[tuple[int, int]]]] = []
        feats: list[list[float]] = []
        batch_ids: list[str | None] = []
        batch_src: list[str] = []
        for pos, row in enumerate(rows):
            hit = answers.lookup(row) if use_answers else None
            if hit is not None and hit in index.id_to_index:
                batch_ids.append(hit)
                batch_src.append("answer_book")
                continue
            cands = ranker.candidates(row, index)
            pending.append((pos, cands))
            feats.extend(ranker.features(row, index, i, b) for i, b in cands)
            batch_ids.append(None)
            batch_src.append("ranker")
        if pending:
            scores = model.predict_proba(np.asarray(feats, dtype=np.float32))[:, 1]
            offset = 0
            for pos, cands in pending:
                local = scores[offset: offset + len(cands)]
                batch_ids[pos] = index.ids[cands[int(np.argmax(local))][0]]
                offset += len(cands)
        ids.extend(batch_ids)  # type: ignore[arg-type]
        sources.extend(batch_src)
    return ids, sources


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
