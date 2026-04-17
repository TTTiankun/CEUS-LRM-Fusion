"""Evaluate the Clinical-LR branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ceus_lrm_fusion.clinical.data import load_labeled_split
from ceus_lrm_fusion.clinical.metrics import binary_classification_metrics
from ceus_lrm_fusion.clinical.visualization import (
    plot_coefficients,
    plot_confusion,
    plot_pr,
    plot_roc,
)


def evaluate_one_split(bundle: dict, split_dir: str, output_dir: Path, split_name: str) -> dict:
    x_df, y_true, sample_names, class_names = load_labeled_split(
        root_dir=split_dir,
        positive_label=bundle["positive_label"],
        augment=False,
    )
    x = bundle["preprocessor"].transform(x_df)
    y_prob = bundle["model"].predict_proba(x)[:, 1]
    y_pred = bundle["model"].predict(x)

    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    metrics = binary_classification_metrics(y_true, y_pred, y_prob)
    with open(split_output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    pd.DataFrame(
        {
            "sample_name": sample_names,
            "true_label": [class_names[index] for index in y_true],
            "pred_label": [class_names[index] for index in y_pred],
            "prob_hcc": y_prob,
            "prob_non_hcc": 1.0 - y_prob,
        }
    ).to_csv(split_output_dir / "per_sample_predictions.csv", index=False)

    plot_confusion(y_true, y_pred, class_names).savefig(split_output_dir / "confusion_matrix.png", bbox_inches="tight")
    plot_roc(y_true, y_prob).savefig(split_output_dir / "roc_curve.png", bbox_inches="tight")
    plot_pr(y_true, y_prob).savefig(split_output_dir / "pr_curve.png", bbox_inches="tight")

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
        split_output_dir / "roc_points.csv",
        index=False,
    )
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(
        split_output_dir / "pr_points.csv",
        index=False,
    )
    pd.DataFrame({"threshold": pr_thresholds}).to_csv(split_output_dir / "pr_thresholds.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Clinical-LR branch.")
    parser.add_argument("--bundle", default="runs/clinical_lr/model_bundle.pkl", help="Path to model_bundle.pkl.")
    parser.add_argument("--output", default="reports/clinical_lr", help="Directory for evaluation outputs.")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    config = bundle["config"]
    if config.get("val_dir"):
        summary["val"] = evaluate_one_split(bundle, config["val_dir"], output_dir, "val")
    if config.get("test_dir"):
        summary["test"] = evaluate_one_split(bundle, config["test_dir"], output_dir, "test")

    coefficient_path = Path(args.bundle).with_name("coefficients.csv")
    if coefficient_path.exists():
        coefficients = pd.read_csv(coefficient_path)
        coefficients.to_csv(output_dir / "coefficients.csv", index=False)
        plot_coefficients(coefficients, top_k=int(config.get("top_k_coefficients", 20))).savefig(
            output_dir / "coefficients.png",
            bbox_inches="tight",
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
