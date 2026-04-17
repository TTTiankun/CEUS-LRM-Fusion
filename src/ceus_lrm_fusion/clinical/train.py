"""Train the Clinical-LR branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml

from ceus_lrm_fusion.clinical.data import build_preprocessor, load_labeled_split
from ceus_lrm_fusion.clinical.metrics import binary_classification_metrics
from ceus_lrm_fusion.clinical.model_zoo import bootstrap_logistic_coefficients, train_logistic_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Clinical-LR branch.")
    parser.add_argument("--config", default="configs/clinical_lr.yaml", help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    save_dir = Path(config.get("save_dir", "runs/clinical_lr"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    x_train_df, y_train, _, class_names = load_labeled_split(
        root_dir=config["train_dir"],
        positive_label=config.get("positive_label", "HCC"),
        augment=bool(config.get("use_augmentation", False)),
        augment_n=int(config.get("augment_n", 0)),
        augment_noise_std=float(config.get("augment_noise_std", 0.01)),
        seed=int(config.get("random_seed", 42)),
    )
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(x_train_df)
    feature_names = preprocessor.get_feature_names_out().tolist()

    model = train_logistic_pipeline(
        x_train=x_train,
        y_train=y_train,
        c_value=float(config.get("c_value", 1.0)),
        penalty=str(config.get("penalty", "l2")),
        seed=int(config.get("random_seed", 42)),
        max_iter=int(config.get("max_iter", 5000)),
    )

    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "class_names": class_names,
        "feature_names": feature_names,
        "positive_label": config.get("positive_label", "HCC"),
        "config": config,
    }
    joblib.dump(bundle, save_dir / "model_bundle.pkl")

    summary = {"train_num_samples": int(len(y_train))}
    train_prob = model.predict_proba(x_train)[:, 1]
    train_pred = model.predict(x_train)
    summary["train_metrics"] = binary_classification_metrics(y_train, train_pred, train_prob)

    if config.get("val_dir"):
        x_val_df, y_val, _, _ = load_labeled_split(
            root_dir=config["val_dir"],
            positive_label=config.get("positive_label", "HCC"),
            augment=False,
        )
        x_val = preprocessor.transform(x_val_df)
        val_prob = model.predict_proba(x_val)[:, 1]
        val_pred = model.predict(x_val)
        summary["val_metrics"] = binary_classification_metrics(y_val, val_pred, val_prob)

    if config.get("export_bootstrap_coefficients", True):
        coefficient_frame = bootstrap_logistic_coefficients(
            x_train=x_train,
            y_train=y_train,
            feature_names=feature_names,
            c_value=float(config.get("c_value", 1.0)),
            penalty=str(config.get("penalty", "l2")),
            seed=int(config.get("random_seed", 42)),
            max_iter=int(config.get("max_iter", 5000)),
            n_bootstraps=int(config.get("n_bootstraps", 200)),
            alpha=float(config.get("alpha", 0.05)),
        )
        coefficient_frame.to_csv(save_dir / "coefficients.csv", index=False)

    with open(save_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
