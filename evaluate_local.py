"""Local validation.

1. Replay train.csv through the saved bundle exactly as script.py would
   (answer book ON) -> must be 100%.
2. GroupKFold by vessel (answer book OFF for held-out vessels) -> honest
   estimate of the fallback ranker on vessels never seen in training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from vessel_matcher import CatalogIndex, fit, predict_ids, weighted_accuracy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--catalog", default="data/vessel_catalog.csv")
    parser.add_argument("--bundle", default="model/bundle.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--only-fold", type=int, default=None, help="run only this fold index")
    args = parser.parse_args()

    train = pd.read_csv(args.train, encoding="utf-8-sig")
    catalog = pd.read_csv(args.catalog, encoding="utf-8-sig")
    index = CatalogIndex(catalog)

    if not args.skip_replay:
        bundle = joblib.load(args.bundle)
        ids, sources = predict_ids(bundle, train, index)
        full = weighted_accuracy(train, ids)
        print("[replay] train.csv through bundle (answer book ON):", full,
              "| sources:", {s: sources.count(s) for s in set(sources)})
        if full["accuracy"] < 1.0:
            wrong = train[np.asarray(ids) != train["target_vessel_id"].astype(str).to_numpy()]
            print(wrong.head(20).to_string())

    if args.skip_cv:
        return
    gkf = GroupKFold(n_splits=args.folds)
    results = []
    for k, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train["target_vessel_id"])):
        if args.only_fold is not None and k != args.only_fold:
            continue
        started = time.time()
        fold_bundle = fit(train.iloc[tr_idx], catalog, n_estimators=args.n_estimators, index=index)
        va = train.iloc[va_idx]
        pred, _ = predict_ids(fold_bundle, va, index, use_answers=False)
        metrics = weighted_accuracy(va, pred)
        pred_arr = np.asarray(pred)
        wrong_mask = pred_arr != va["target_vessel_id"].astype(str).to_numpy()
        for r, p in zip(va[wrong_mask].to_dict("records"), pred_arr[wrong_mask]):
            print(f"   miss: {r['vessel_name']!r} gt={r['gross_tonnage']} flag={r['flag_name']} "
                  f"type={r['vessel_type']} true={r['target_vessel_id']} pred={p}")
        metrics["candidate_recall_train"] = fold_bundle["candidate_recall"]
        results.append(metrics)
        print(f"[cv fold {k}] {metrics} ({time.time() - started:.0f}s)")
    if len(results) > 1:
        mean = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        print("[cv mean] unseen-vessel fallback:", mean)


if __name__ == "__main__":
    main()
