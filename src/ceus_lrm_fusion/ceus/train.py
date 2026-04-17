"""Train the CEUS-GRU branch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ceus_lrm_fusion.ceus.data import TimeSeriesDataset, pad_collate_fn
from ceus_lrm_fusion.ceus.early_stopping import EarlyStopping
from ceus_lrm_fusion.ceus.metrics import binary_classification_metrics
from ceus_lrm_fusion.ceus.models import AttentionGRUModelPro


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader | None]:
    train_dataset = TimeSeriesDataset(
        directory=config["train_dir"],
        augment=bool(config.get("use_augmentation", True)),
        aug_cfg=config.get("augmentations", {}),
    )
    val_dataset = None
    if config.get("val_dir"):
        val_dataset = TimeSeriesDataset(
            directory=config["val_dir"],
            label_map=train_dataset.label_map,
            augment=False,
        )

    batch_size = int(config.get("batch_size", 16))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_collate_fn,
        )
    config["label_map"] = train_dataset.label_map
    config["input_dim"] = train_dataset.feature_dim
    config["num_classes"] = len(train_dataset.label_map)
    return train_loader, val_loader


def build_model(config: Dict, device: torch.device) -> AttentionGRUModelPro:
    return AttentionGRUModelPro(
        input_dim=int(config["input_dim"]),
        attention_dim=int(config.get("attention_dim", config["input_dim"])),
        gru_dims=config.get("gru_dims", [128, 64]),
        num_classes=int(config["num_classes"]),
        n_heads=int(config.get("num_attention_heads", 4)),
        dropout=float(config.get("dropout", 0.1)),
        use_attention_mapper=bool(config.get("use_attention_mapper", True)),
    ).to(device)


def compute_loss(
    logits: torch.Tensor,
    auxiliary_logits: torch.Tensor | None,
    labels: torch.Tensor,
    criterion: nn.Module,
    aux_weight: float,
) -> torch.Tensor:
    loss = criterion(logits, labels)
    if auxiliary_logits is not None and aux_weight > 0:
        loss = loss + aux_weight * criterion(auxiliary_logits, labels)
    return loss


def run_epoch(
    model: AttentionGRUModelPro,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    aux_weight: float = 0.2,
    grad_clip: float = 0.0,
) -> Dict:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    labels_all = []
    preds_all = []
    probs_all = []

    for inputs, labels, lengths, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        if train_mode:
            optimizer.zero_grad()

        logits, _, auxiliary_logits = model(inputs, lengths)
        loss = compute_loss(logits, auxiliary_logits, labels, criterion, aux_weight)

        if train_mode:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        total_loss += loss.item() * inputs.size(0)
        labels_all.append(labels.cpu().numpy())
        preds_all.append(predictions.cpu().numpy())
        probs_all.append(probabilities[:, 1].detach().cpu().numpy())

    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(preds_all)
    y_prob = np.concatenate(probs_all)
    metrics = binary_classification_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def save_checkpoint(path: Path, model: nn.Module, config: Dict, epoch: int, metrics: Dict) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CEUS-GRU branch.")
    parser.add_argument("--config", default="configs/ceus_gru.yaml", help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    set_seed(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    save_dir = Path(config.get("save_dir", "runs/ceus_gru"))
    save_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_dataloaders(config)
    with open(save_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))

    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    learning_rate = float(config.get("lr", 1e-3))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = None
    if config.get("use_cosine_scheduler", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("epochs", 100)),
        )

    early_stopping = None
    if val_loader is not None and config.get("early_stop_patience"):
        early_stopping = EarlyStopping(
            patience=int(config.get("early_stop_patience", 20)),
            mode="max",
            min_delta=float(config.get("min_delta", 0.0)),
        )

    history = []
    best_auc = -np.inf
    best_epoch = 0
    for epoch in range(1, int(config.get("epochs", 100)) + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            aux_weight=float(config.get("auxiliary_loss_weight", 0.2)),
            grad_clip=float(config.get("grad_clip", 0.0)),
        )
        record = {"epoch": epoch, "train": train_metrics}

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                criterion=criterion,
                optimizer=None,
                aux_weight=float(config.get("auxiliary_loss_weight", 0.2)),
            )
            record["val"] = val_metrics
            monitored_auc = float(val_metrics["auc"]) if val_metrics["auc"] is not None else float(val_metrics["accuracy"])
            if monitored_auc > best_auc:
                best_auc = monitored_auc
                best_epoch = epoch
                save_checkpoint(save_dir / "best.pt", model, config, epoch, val_metrics)
            if early_stopping and early_stopping.step(monitored_auc):
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
            if train_metrics["auc"] is not None and train_metrics["auc"] > best_auc:
                best_auc = float(train_metrics["auc"])
                best_epoch = epoch
                save_checkpoint(save_dir / "best.pt", model, config, epoch, train_metrics)
            print(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                f"train_auc={train_metrics['auc']:.4f} train_acc={train_metrics['accuracy']:.4f}"
            )

        history.append(record)
        save_checkpoint(save_dir / "last.pt", model, config, epoch, record.get("val", train_metrics))
        if scheduler is not None:
            scheduler.step()

    with open(save_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(save_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, "best_auc": best_auc, "num_epochs": len(history)}, handle, indent=2)


if __name__ == "__main__":
    main()
