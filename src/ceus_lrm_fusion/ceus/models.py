"""Model definitions for the CEUS-GRU branch."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Lightweight self-attention block used before temporal recurrence."""

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.projection = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = inputs.shape
        qkv = self.projection(inputs).chunk(3, dim=-1)
        query, key, value = [
            tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
            for tensor in qkv
        ]
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_dim)
        return self.output(attended), weights.mean(dim=1)


class AttentionGRUModelPro(nn.Module):
    """Reference CEUS-GRU model used for the public repository."""

    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        gru_dims: Iterable[int],
        num_classes: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_attention_mapper: bool = True,
    ) -> None:
        super().__init__()
        gru_dims = list(gru_dims)
        if not gru_dims:
            raise ValueError("gru_dims must contain at least one hidden dimension")

        self.feature_projection = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim),
            nn.Conv1d(input_dim, attention_dim, kernel_size=1),
            nn.GELU(),
        )
        self.pre_norm = nn.LayerNorm(attention_dim)
        self.attention = MultiHeadSelfAttention(attention_dim, num_heads=n_heads)
        self.attention_norm = nn.LayerNorm(attention_dim)

        first_gru_dim = gru_dims[0]
        self.mapper = nn.Linear(attention_dim, first_gru_dim) if use_attention_mapper else nn.Identity()

        self.gru_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        previous_dim = first_gru_dim
        for hidden_dim in gru_dims:
            self.gru_layers.append(nn.GRU(previous_dim, hidden_dim, batch_first=True, bidirectional=True))
            output_dim = hidden_dim * 2
            self.skip_layers.append(
                nn.Linear(previous_dim, output_dim) if previous_dim != output_dim else nn.Identity()
            )
            self.layer_norms.append(nn.LayerNorm(output_dim))
            previous_dim = output_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(previous_dim, max(previous_dim // 2, num_classes)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(previous_dim // 2, num_classes), num_classes),
        )
        self.auxiliary_head = nn.Linear(gru_dims[0] * 2, num_classes)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.feature_projection(inputs.transpose(1, 2)).transpose(1, 2)
        sequence = self.pre_norm(sequence)

        mask = None
        if lengths is not None:
            time_index = torch.arange(sequence.size(1), device=sequence.device)
            mask = time_index[None, :] < lengths[:, None]

        attended, weights = self.attention(sequence, mask=mask)
        sequence = self.attention_norm(attended + sequence)
        sequence = self.mapper(sequence)

        auxiliary_logits = None
        for layer_index, (gru, skip, layer_norm) in enumerate(
            zip(self.gru_layers, self.skip_layers, self.layer_norms)
        ):
            residual = skip(sequence)
            sequence, _ = gru(sequence)
            sequence = layer_norm(sequence + residual)
            sequence = self.dropout(sequence)
            if layer_index == 0:
                auxiliary_logits = self.auxiliary_head(sequence[:, -1, :])

        logits = self.classifier(sequence[:, -1, :])
        return logits, weights.mean(dim=1), auxiliary_logits


AttentionGRUModel_Pro = AttentionGRUModelPro
