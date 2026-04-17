"""Batch prediction for the Clinical-LR branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ceus_lrm_fusion.clinical.data import load_unlabeled_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Clinical-LR predictions for unlabeled feature files.")
    parser.add_argument("--bundle", default="runs/clinical_lr/model_bundle.pkl", help="Path to model_bundle.pkl.")
    parser.add_argument("--input", required=True, help="Directory containing unlabeled .txt feature files.")
    parser.add_argument("--output", default="reports/clinical_predictions", help="Prediction output directory.")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    x_df, sample_names = load_unlabeled_directory(args.input)
    x = bundle["preprocessor"].transform(x_df)
    y_prob = bundle["model"].predict_proba(x)[:, 1]
    y_pred = bundle["model"].predict(x)
    class_names = bundle["class_names"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_name": sample_names,
            "pred_label": [class_names[index] for index in y_pred],
            "prob_hcc": y_prob,
            "prob_non_hcc": 1.0 - y_prob,
        }
    ).to_csv(output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
