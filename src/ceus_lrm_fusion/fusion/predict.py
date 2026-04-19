"""Batch prediction for the neural LRM-Fusion branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ceus_lrm_fusion.ceus.data import TimeSeriesDataset, pad_collate_fn
from ceus_lrm_fusion.ceus.models import AttentionGRUModelPro


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LRM-Fusion predictions for a directory of sequences.")
    parser.add_argument("--checkpoint", default="runs/fusion/best.pt", help="Path to a trained LRM-Fusion checkpoint.")
    parser.add_argument("--input", required=True, help="Directory containing .txt or .npz fusion sequences.")
    parser.add_argument("--output", default="reports/fusion_predictions", help="Directory for prediction outputs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    label_map = config["label_map"]
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]

    dataset = TimeSeriesDataset(
        directory=args.input,
        label_map=label_map,
        augment=False,
        confidence_cfg=config.get("confidence_suppression", {}),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_fn)

    model = AttentionGRUModelPro(
        input_dim=int(config["input_dim"]),
        attention_dim=int(config.get("attention_dim", config["input_dim"])),
        gru_dims=config.get("gru_dims", [64, 32]),
        num_classes=int(config["num_classes"]),
        n_heads=int(config.get("num_attention_heads", 8)),
        dropout=float(config.get("dropout", 0.2)),
        use_attention_mapper=bool(config.get("use_attention_mapper", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for inputs, _, lengths, names in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            logits, _, _ = model(inputs, lengths)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            predicted_index = int(probabilities.argmax())
            rows.append(
                {
                    "sample_name": names[0],
                    "pred_label": class_names[predicted_index],
                    "prob_hcc": float(probabilities[label_map["HCC"]]) if "HCC" in label_map else float(probabilities[1]),
                    "prob_non_hcc": float(probabilities[label_map["UNHCC"]]) if "UNHCC" in label_map else float(probabilities[0]),
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
