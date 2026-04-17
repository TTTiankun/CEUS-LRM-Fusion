"""Evaluate the LRM-Fusion branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ceus_lrm_fusion.clinical.metrics import binary_classification_metrics
from ceus_lrm_fusion.clinical.visualization import plot_confusion, plot_pr, plot_roc
from ceus_lrm_fusion.fusion.data import load_fusion_table


def evaluate_split(bundle: dict, ceus_csv: str, clinical_csv: str, label_csv: str, output_dir: Path, split_name: str):
    frame = load_fusion_table(ceus_csv=ceus_csv, clinical_csv=clinical_csv, label_csv=label_csv)
    x = frame[bundle["feature_columns"]]
    y_true = frame["label"].to_numpy()
    y_prob = bundle["model"].predict_proba(x)[:, 1]
    y_pred = bundle["model"].predict(x)

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    metrics = binary_classification_metrics(y_true, y_pred, y_prob)
    with open(split_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    pd.DataFrame(
        {
            "sample_name": frame["sample_name"],
            "label": y_true,
            "pred_label": y_pred,
            "prob_hcc": y_prob,
            "prob_non_hcc": 1.0 - y_prob,
            "ceus_prob_hcc": frame["ceus_prob_hcc"],
            "clinical_prob_hcc": frame["clinical_prob_hcc"],
        }
    ).to_csv(split_dir / "per_sample_predictions.csv", index=False)

    plot_confusion(y_true, y_pred, ["non-HCC", "HCC"]).savefig(split_dir / "confusion_matrix.png", bbox_inches="tight")
    plot_roc(y_true, y_prob).savefig(split_dir / "roc_curve.png", bbox_inches="tight")
    plot_pr(y_true, y_prob).savefig(split_dir / "pr_curve.png", bbox_inches="tight")

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(split_dir / "roc_points.csv", index=False)
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(split_dir / "pr_points.csv", index=False)
    pd.DataFrame({"threshold": pr_thresholds}).to_csv(split_dir / "pr_thresholds.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LRM-Fusion branch.")
    parser.add_argument("--bundle", default="runs/fusion/model_bundle.pkl", help="Path to model_bundle.pkl.")
    parser.add_argument("--output", default="reports/fusion", help="Directory for evaluation outputs.")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    config = bundle["config"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    if config.get("val_ceus_csv") and config.get("val_clinical_csv") and config.get("val_label_csv"):
        summary["val"] = evaluate_split(
            bundle,
            config["val_ceus_csv"],
            config["val_clinical_csv"],
            config["val_label_csv"],
            output_dir,
            "val",
        )
    if config.get("test_ceus_csv") and config.get("test_clinical_csv") and config.get("test_label_csv"):
        summary["test"] = evaluate_split(
            bundle,
            config["test_ceus_csv"],
            config["test_clinical_csv"],
            config["test_label_csv"],
            output_dir,
            "test",
        )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
