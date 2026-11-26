"""Belief encoder: CNN + BiLSTM + attention pool + aux concat + MLP head.

Architecture (spec locked with user, ~180k params):

  msg_id_seq (B, 256) int64
    -> Embedding(vocab=30, dim=16)             -> (B, 256, 16)
    -> Conv1d(16, 64, k=5, pad=2) + LN + ReLU  -> (B, 256, 64)
    -> Conv1d(64, 64, k=3, pad=1) + LN + ReLU  -> (B, 256, 64)
    -> BiLSTM(64, hidden=64, 2 layers, drop=0.2) -> (B, 256, 128)
    -> AttentionPool(learned query)            -> context (B, 128), weights (B, 256)

  aux (B, 6) float32
    -> Linear(6, 32) + ReLU + LayerNorm        -> (B, 32)

  concat context + aux (B, 160)
    -> Linear(160, 64) + ReLU + Dropout(0.3)
    -> Linear(64, 6)                           -> logits (B, 6)

Forward returns (logits, attention_weights). The attention weights over the
256 timesteps are preserved so the paper can visualize which msg positions
the encoder finds discriminative per attack class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BeliefEncoderConfig:
    vocab_size: int = 30
    embed_dim: int = 16
    conv_channels: int = 64
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    aux_dim: int = 6
    aux_hidden: int = 32
    mlp_hidden: int = 64
    mlp_dropout: float = 0.3
    n_classes: int = 6
    window_size: int = 256


class AttentionPool(nn.Module):
    """Additive attention with a learned query vector.

    score_t = v^T tanh(W h_t)      shape (B, T)
    a_t     = softmax(score_t)     shape (B, T)
    ctx     = sum_t a_t * h_t      shape (B, H)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self._proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self._query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self._proj.weight)
        nn.init.zeros_(self._proj.bias)
        nn.init.normal_(self._query, mean=0.0, std=0.1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, H)
        scores = torch.tanh(self._proj(h)) @ self._query  # (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        ctx = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return ctx, weights


class BeliefEncoder(nn.Module):
    """CNN + BiLSTM + attention + aux MLP classifier for MAVLink belief windows."""

    def __init__(self, cfg: BeliefEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BeliefEncoderConfig()
        c = self.cfg

        self.embed = nn.Embedding(c.vocab_size, c.embed_dim, padding_idx=0)

        # Two Conv1d layers. Input needs (B, C, T) so we permute after embed.
        self.conv1 = nn.Conv1d(c.embed_dim, c.conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(
            c.conv_channels, c.conv_channels, kernel_size=3, padding=1
        )
        self.ln1 = nn.LayerNorm(c.conv_channels)
        self.ln2 = nn.LayerNorm(c.conv_channels)

        self.lstm = nn.LSTM(
            input_size=c.conv_channels,
            hidden_size=c.lstm_hidden,
            num_layers=c.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=c.lstm_dropout if c.lstm_layers > 1 else 0.0,
        )
        lstm_out = c.lstm_hidden * 2  # bidirectional
        self.attn = AttentionPool(lstm_out)

        self.aux_mlp = nn.Sequential(
            nn.Linear(c.aux_dim, c.aux_hidden),
            nn.ReLU(),
            nn.LayerNorm(c.aux_hidden),
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_out + c.aux_hidden, c.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(c.mlp_dropout),
            nn.Linear(c.mlp_hidden, c.n_classes),
        )

    def forward(
        self, msg_ids: torch.Tensor, aux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            msg_ids: (B, T) int64 dense-remapped msg indices (0=UNK).
            aux:     (B, aux_dim) float32 z-score-normalized auxiliary features.

        Returns:
            logits:  (B, n_classes)
            attn_w:  (B, T) attention weights (sum to 1 per row)
        """
        # (B, T) -> (B, T, E)
        x = self.embed(msg_ids)
        # (B, T, E) -> (B, E, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv1(x)
        # LayerNorm expects (..., C); permute back for norm then permute again.
        x = x.transpose(1, 2)  # (B, T, C)
        x = torch.relu(self.ln1(x))
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = torch.relu(self.ln2(x))
        # BiLSTM takes (B, T, C)
        h, _ = self.lstm(x)  # (B, T, 2*H)
        ctx, attn_w = self.attn(h)  # (B, 2H), (B, T)

        a = self.aux_mlp(aux)  # (B, aux_hidden)
        z = torch.cat([ctx, a], dim=1)  # (B, 2H + aux_hidden)
        logits = self.head(z)  # (B, n_classes)
        return logits, attn_w

    @torch.no_grad()
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["BeliefEncoder", "BeliefEncoderConfig", "AttentionPool"]
