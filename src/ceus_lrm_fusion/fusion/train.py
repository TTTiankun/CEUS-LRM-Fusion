"""Train the neural LRM-Fusion branch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from ceus_lrm_fusion.ceus.early_stopping import EarlyStopping
from ceus_lrm_fusion.ceus.models import AttentionGRUModelPro
from ceus_lrm_fusion.ceus.train import run_epoch, save_checkpoint
from ceus_lrm_fusion.fusion.data import build_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: Dict, device: torch.device) -> AttentionGRUModelPro:
    return AttentionGRUModelPro(
        input_dim=int(config["input_dim"]),
        attention_dim=int(config.get("attention_dim", config["input_dim"])),
        gru_dims=config.get("gru_dims", [64, 32]),
        num_classes=int(config["num_classes"]),
        n_heads=int(config.get("num_attention_heads", 8)),
        dropout=float(config.get("dropout", 0.2)),
        use_attention_mapper=bool(config.get("use_attention_mapper", True)),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the neural LRM-Fusion branch.")
    parser.add_argument("--config", default="configs/fusion.yaml", help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("random_seed", 24))
    set_seed(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    save_dir = Path(config.get("save_dir", "runs/fusion"))
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_dataset = build_dataloaders(config)
    config["label_map"] = train_dataset.label_map
    config["input_dim"] = train_dataset.feature_dim
    config["num_classes"] = len(train_dataset.label_map)
    with open(save_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    model = build_model(config, device)

    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    learning_rate = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_epochs = int(config.get("epochs", 500))
    warmup_epochs = int(config.get("warmup_epochs", 0))
    scheduler = None
    if total_epochs > warmup_epochs:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
        )
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0,
        )

    use_swa = bool(config.get("use_swa", False))
    swa_model = None
    swa_scheduler = None
    swa_start = None
    if use_swa:
        swa_start = int(total_epochs * float(config.get("swa_start_epoch", 0.8)))
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=float(config.get("swa_lr", learning_rate)))

    early_stopping = None
    if val_loader is not None and config.get("early_stop_patience"):
        early_stopping = EarlyStopping(
            patience=int(config.get("early_stop_patience", 150)),
            mode="max",
            min_delta=float(config.get("min_delta", 0.0)),
        )

    history = []
    best_score = -np.inf
    best_epoch = 0
    for epoch in range(1, total_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            aux_weight=float(config.get("auxiliary_loss_weight", 0.3)),
            grad_clip=float(config.get("grad_clip", 0.5)),
            label_smoothing=float(config.get("label_smoothing", 0.05)),
        )
        record = {"epoch": epoch, "train": train_metrics}

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                aux_weight=float(config.get("auxiliary_loss_weight", 0.3)),
                label_smoothing=float(config.get("label_smoothing", 0.05)),
            )
            record["val"] = val_metrics
            monitored_score = float(val_metrics["accuracy"])
            if monitored_score > best_score:
                best_score = monitored_score
                best_epoch = epoch
                save_checkpoint(save_dir / "best.pt", model, config, epoch, val_metrics)
            if early_stopping and early_stopping.step(monitored_score):
                history.append(record)
                print(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                    f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['accuracy']:.4f} [early stop]"
                )
                break
            print(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_auc={train_metrics['auc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )
        else:
            monitored_score = float(train_metrics["accuracy"])
            if monitored_score > best_score:
                best_score = monitored_score
                best_epoch = epoch
                save_checkpoint(save_dir / "best.pt", model, config, epoch, train_metrics)
            print(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                f"train_auc={train_metrics['auc']:.4f} train_acc={train_metrics['accuracy']:.4f}"
            )

        history.append(record)
        save_checkpoint(save_dir / "last.pt", model, config, epoch, record.get("val", train_metrics))

        if warmup_scheduler is not None and epoch <= warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()
        if use_swa and swa_model is not None and swa_scheduler is not None and swa_start is not None and epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    with open(save_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    if use_swa and swa_model is not None:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        save_checkpoint(save_dir / "best_swa.pt", swa_model.module, config, total_epochs, {"swa": True})
    with open(save_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, "best_accuracy": best_score, "num_epochs": len(history)}, handle, indent=2)


if __name__ == "__main__":
    main()
