"""Evaluate the CEUS-GRU branch and export publication-friendly outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from ceus_lrm_fusion.ceus.data import TimeSeriesDataset, pad_collate_fn
from ceus_lrm_fusion.ceus.metrics import binary_classification_metrics, optimal_youden_threshold
from ceus_lrm_fusion.ceus.models import AttentionGRUModelPro
from ceus_lrm_fusion.ceus.visualization import (
    plot_attention_heatmap,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    return torch.load(checkpoint_path, map_location=device)


def build_model(config: Dict, device: torch.device) -> AttentionGRUModelPro:
    model = AttentionGRUModelPro(
        input_dim=int(config["input_dim"]),
        attention_dim=int(config.get("attention_dim", config["input_dim"])),
        gru_dims=config.get("gru_dims", [128, 64]),
        num_classes=int(config["num_classes"]),
        n_heads=int(config.get("num_attention_heads", 4)),
        dropout=float(config.get("dropout", 0.1)),
        use_attention_mapper=bool(config.get("use_attention_mapper", True)),
    ).to(device)
    return model


def evaluate_split(
    model: AttentionGRUModelPro,
    dataset: TimeSeriesDataset,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    model.eval()

    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    attention_all = []
    sample_names = []

    with torch.no_grad():
        for inputs, labels, lengths, names in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits, attention, _ = model(inputs, lengths)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            y_true_all.append(labels.cpu().numpy())
            y_pred_all.append(predictions.cpu().numpy())
            y_prob_all.append(probabilities[:, 1].cpu().numpy())
            attention_all.append(attention.cpu().numpy())
            sample_names.extend(names)

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(y_prob_all),
        np.concatenate(attention_all),
        sample_names,
    )


def export_split_outputs(
    output_dir: Path,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    attention: np.ndarray,
    sample_names: List[str],
    class_names: List[str],
) -> Dict:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    metrics = binary_classification_metrics(y_true, y_pred, y_prob)
    with open(split_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    pd.DataFrame(
        {
            "sample_name": sample_names,
            "true_label": [class_names[index] for index in y_true],
            "pred_label": [class_names[index] for index in y_pred],
            "prob_hcc": y_prob,
            "prob_non_hcc": 1.0 - y_prob,
        }
    ).to_csv(split_dir / "per_sample_predictions.csv", index=False)

    pd.DataFrame(attention).assign(sample_name=sample_names).to_csv(
        split_dir / "attention_weights.csv", index=False
    )

    fig_cm, _ = plot_confusion_matrix(y_true, y_pred, class_names)
    fig_cm.savefig(split_dir / "confusion_matrix.png", bbox_inches="tight")
    fig_roc, _ = plot_roc_curve(y_true, np.column_stack([1 - y_prob, y_prob]), class_names)
    fig_roc.savefig(split_dir / "roc_curve.png", bbox_inches="tight")
    fig_pr, _ = plot_pr_curve(y_true, np.column_stack([1 - y_prob, y_prob]), class_names)
    fig_pr.savefig(split_dir / "pr_curve.png", bbox_inches="tight")

    max_attention_plots = min(5, len(sample_names))
    for index in range(max_attention_plots):
        fig_attn, _ = plot_attention_heatmap(attention[index], title=f"{split_name}: {sample_names[index]}")
        fig_attn.savefig(split_dir / f"attention_{index+1}_{sample_names[index]}.png", bbox_inches="tight")

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
        split_dir / "roc_points.csv",
        index=False,
    )
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(
        split_dir / "pr_points.csv",
        index=False,
    )
    pd.DataFrame({"threshold": pr_thresholds}).to_csv(split_dir / "pr_thresholds.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CEUS-GRU branch.")
    parser.add_argument("--checkpoint", default="runs/ceus_gru/best.pt", help="Path to a CEUS checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--output", default="reports/ceus_gru", help="Evaluation output directory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config.update(yaml.safe_load(handle))

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])
    class_names = [name for name, _ in sorted(config["label_map"].items(), key=lambda item: item[1])]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    validation_threshold = None
    for split_name, split_key in [("val", "val_dir"), ("test", "test_dir")]:
        if not config.get(split_key):
            continue
        dataset = TimeSeriesDataset(
            directory=config[split_key],
            label_map=config["label_map"],
            augment=False,
            confidence_cfg=config.get("confidence_suppression", {}),
        )
        y_true, y_pred, y_prob, attention, sample_names = evaluate_split(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(config.get("batch_size", 16)),
        )
        split_metrics = export_split_outputs(
            output_dir=output_dir,
            split_name=split_name,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            attention=attention,
            sample_names=sample_names,
            class_names=class_names,
        )
        summary[split_name] = split_metrics
        if split_name == "val":
            validation_threshold = optimal_youden_threshold(y_true, y_prob)
            with open(output_dir / split_name / "youden_threshold.json", "w", encoding="utf-8") as handle:
                json.dump(validation_threshold, handle, indent=2)

    if validation_threshold is not None:
        summary["validation_threshold"] = validation_threshold
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
