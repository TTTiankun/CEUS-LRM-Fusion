"""Train the LRM-Fusion branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml
from sklearn.linear_model import LogisticRegression

from ceus_lrm_fusion.clinical.metrics import binary_classification_metrics
from ceus_lrm_fusion.fusion.data import load_fusion_table


def train_fusion_model(x_train, y_train, c_value: float = 1.0, seed: int = 42) -> LogisticRegression:
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=seed)
    model.fit(x_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LRM-Fusion branch.")
    parser.add_argument("--config", default="configs/fusion.yaml", help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    save_dir = Path(config.get("save_dir", "runs/fusion"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    train_frame = load_fusion_table(
        ceus_csv=config["train_ceus_csv"],
        clinical_csv=config["train_clinical_csv"],
        label_csv=config["train_label_csv"],
    )
    feature_columns = ["ceus_prob_hcc", "clinical_prob_hcc"]
    model = train_fusion_model(
        x_train=train_frame[feature_columns].to_numpy(),
        y_train=train_frame["label"].to_numpy(),
        c_value=float(config.get("c_value", 1.0)),
        seed=int(config.get("random_seed", 42)),
    )

    bundle = {"model": model, "feature_columns": feature_columns, "config": config}
    joblib.dump(bundle, save_dir / "model_bundle.pkl")

    train_prob = model.predict_proba(train_frame[feature_columns])[:, 1]
    train_pred = model.predict(train_frame[feature_columns])
    summary = {"train_metrics": binary_classification_metrics(train_frame["label"], train_pred, train_prob)}

    if config.get("val_ceus_csv") and config.get("val_clinical_csv") and config.get("val_label_csv"):
        val_frame = load_fusion_table(
            ceus_csv=config["val_ceus_csv"],
            clinical_csv=config["val_clinical_csv"],
            label_csv=config["val_label_csv"],
        )
        val_prob = model.predict_proba(val_frame[feature_columns])[:, 1]
        val_pred = model.predict(val_frame[feature_columns])
        summary["val_metrics"] = binary_classification_metrics(val_frame["label"], val_pred, val_prob)

    with open(save_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
