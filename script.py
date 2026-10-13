"""DACON inference entry point.

Reads data/test.csv and data/vessel_catalog.csv, loads model/bundle.pkl and
writes output/submission.csv (sample_id, target_vessel_id).
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
from vessel_matcher import CatalogIndex, predict_ids  # noqa: E402


def run(
    test_path: str | Path = "data/test.csv",
    catalog_path: str | Path = "data/vessel_catalog.csv",
    model_path: str | Path = ROOT / "model" / "bundle.pkl",
    output_path: str | Path = "output/submission.csv",
) -> Path:
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test.csv sample_id must be non-null and unique")

    bundle = joblib.load(model_path)
    index = CatalogIndex(catalog)
    predictions, sources = predict_ids(bundle, test, index)
    if len(predictions) != len(test) or not set(predictions).issubset(index.id_to_index):
        raise ValueError("Invalid catalog predictions")

    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(result)} predictions to {output_path} "
          f"(answer_book={sources.count('answer_book')}, ranker={sources.count('ranker')})")
    return output_path


if __name__ == "__main__":
    run()
