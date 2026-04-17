"""Batch prediction for the LRM-Fusion branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ceus_lrm_fusion.fusion.data import load_fusion_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LRM-Fusion predictions.")
    parser.add_argument("--bundle", default="runs/fusion/model_bundle.pkl", help="Path to model_bundle.pkl.")
    parser.add_argument("--ceus", required=True, help="CEUS prediction CSV with sample_name and prob_hcc columns.")
    parser.add_argument("--clinical", required=True, help="Clinical prediction CSV with sample_name and prob_hcc columns.")
    parser.add_argument("--output", default="reports/fusion_predictions", help="Prediction output directory.")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    frame = load_fusion_table(args.ceus, args.clinical, label_csv=None)
    x = frame[bundle["feature_columns"]]
    y_prob = bundle["model"].predict_proba(x)[:, 1]
    y_pred = bundle["model"].predict(x)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_name": frame["sample_name"],
            "pred_label": y_pred,
            "prob_hcc": y_prob,
            "prob_non_hcc": 1.0 - y_prob,
            "ceus_prob_hcc": frame["ceus_prob_hcc"],
            "clinical_prob_hcc": frame["clinical_prob_hcc"],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
