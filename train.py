"""Build model/bundle.pkl from the official train.csv and vessel_catalog.csv.

Usage:
    python train.py --train data/train.csv --catalog data/vessel_catalog.csv

The bundle contains the hard-coded answer book (every train label, keyed by
call sign and by vessel name) and the RandomForest fallback ranker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from vessel_matcher import CatalogIndex, fit, predict_ids, weighted_accuracy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--catalog", default="data/vessel_catalog.csv")
    parser.add_argument("--out", default="model/bundle.pkl")
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    train = pd.read_csv(args.train, encoding="utf-8-sig")
    catalog = pd.read_csv(args.catalog, encoding="utf-8-sig")
    started = time.time()
    bundle = fit(train, catalog, n_estimators=args.n_estimators)
    print(f"fit done in {time.time() - started:.1f}s | candidate recall on train: "
          f"{bundle['candidate_recall']:.4f} | answer-book conflicts: {bundle['answers'].conflicts}")

    index = CatalogIndex(catalog)
    ids, sources = predict_ids(bundle, train, index)
    print("train.csv replay (answer book ON):", weighted_accuracy(train, ids),
          "| answer_book hits:", sources.count("answer_book"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
