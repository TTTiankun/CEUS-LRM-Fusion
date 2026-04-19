"""Report parameter counts for the legacy CEUS-GRU / LRM-Fusion setup without requiring torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml


def count_attention_gru_parameters(
    input_dim: int,
    attention_dim: int,
    gru_dims: Iterable[int],
    num_classes: int = 2,
    num_heads: int = 8,
    use_attention_mapper: bool = True,
) -> dict:
    gru_dims = list(gru_dims)
    if not gru_dims:
        raise ValueError("gru_dims must contain at least one hidden dimension")

    parts = {}
    parts["conv_depthwise"] = input_dim * 3 + input_dim
    parts["conv_pointwise"] = attention_dim * input_dim + attention_dim
    parts["pre_layer_norm"] = 2 * attention_dim
    parts["attention_projection"] = 3 * attention_dim * attention_dim
    parts["attention_scale"] = num_heads
    parts["attention_output"] = attention_dim * attention_dim + attention_dim
    parts["attention_layer_norm"] = 2 * attention_dim

    first_dim = gru_dims[0]
    parts["mapper"] = attention_dim * first_dim + first_dim if use_attention_mapper else 0

    gru_total = 0
    residual_total = 0
    norm_total = 0
    input_width = first_dim
    for hidden_dim in gru_dims:
        gru_total += 2 * (3 * hidden_dim * input_width + 3 * hidden_dim * hidden_dim + 6 * hidden_dim)
        output_width = hidden_dim * 2
        if input_width != output_width:
            residual_total += output_width * input_width + output_width
        norm_total += output_width * 2
        input_width = output_width

    parts["gru_stack"] = gru_total
    parts["residual_projection"] = residual_total
    parts["gru_layer_norm"] = norm_total

    classifier_hidden = max(input_width // 2, num_classes)
    parts["classifier"] = (
        classifier_hidden * input_width
        + classifier_hidden
        + num_classes * classifier_hidden
        + num_classes
    )
    parts["aux_head"] = num_classes * (first_dim * 2) + num_classes
    parts["total"] = sum(parts.values())
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Report parameter counts for CEUS-GRU / LRM-Fusion.")
    parser.add_argument("--ceus-config", default="configs/ceus_gru.yaml")
    parser.add_argument("--fusion-config", default="configs/fusion.yaml")
    parser.add_argument("--ceus-input-dim", type=int, default=96, help="Legacy CEUS-GRU input feature width.")
    parser.add_argument("--fusion-input-dim", type=int, default=2, help="Legacy LRM-Fusion input feature width.")
    parser.add_argument(
        "--clinical-feature-count",
        type=int,
        default=20,
        help="Number of Clinical-LR features kept in the manuscript-facing logistic model.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    with open(args.ceus_config, "r", encoding="utf-8") as handle:
        ceus_config = yaml.safe_load(handle)
    with open(args.fusion_config, "r", encoding="utf-8") as handle:
        fusion_config = yaml.safe_load(handle)

    ceus = count_attention_gru_parameters(
        input_dim=args.ceus_input_dim,
        attention_dim=int(ceus_config["attention_dim"]),
        gru_dims=ceus_config["gru_dims"],
        num_classes=int(ceus_config.get("num_classes", 2)),
        num_heads=int(ceus_config.get("num_attention_heads", 8)),
        use_attention_mapper=bool(ceus_config.get("use_attention_mapper", True)),
    )
    fusion = count_attention_gru_parameters(
        input_dim=args.fusion_input_dim,
        attention_dim=int(fusion_config["attention_dim"]),
        gru_dims=fusion_config["gru_dims"],
        num_classes=int(fusion_config.get("num_classes", 2)),
        num_heads=int(fusion_config.get("num_attention_heads", 8)),
        use_attention_mapper=bool(fusion_config.get("use_attention_mapper", True)),
    )
    clinical = {"total": args.clinical_feature_count + 1}

    report = {
        "ceus_gru": ceus,
        "fusion_module": fusion,
        "clinical_lr": clinical,
        "full_system_excluding_image_backbone": ceus["total"] + fusion["total"] + clinical["total"],
        "notes": {
            "feature_extractor": "The released repositories do not contain the image-backbone training code or an explicit ConvNeXtV2 variant identifier, so its exact parameter count cannot be recovered unambiguously from the available artifacts.",
            "fusion_input_dim": args.fusion_input_dim,
            "ceus_input_dim": args.ceus_input_dim,
        },
    }

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
