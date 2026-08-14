"""Reward model for trajectory preference learning."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RewardModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=500,
        pooling="mean",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.pooling = pooling

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, chosen_seq, reject_seq, valid_mask=None):
        return self.logit(chosen_seq, valid_mask), self.logit(reject_seq, valid_mask)

    def score(self, seq, valid_mask=None):
        """Bounded preference score in (0, 1): sigmoid of the raw logit.

        Pairwise training reads `logit()` directly -- the Bradley-Terry objective needs an
        unbounded score, so the sigmoid here is not applied a second time on that path. This
        method is for callers that score one trajectory on its own, e.g. reward reporting.
        """
        return torch.sigmoid(self.logit(seq, valid_mask))

    def logit(self, seq, valid_mask=None):
        batch_size, seq_len, _ = seq.shape
        if valid_mask is not None:
            if valid_mask.dtype != torch.bool or valid_mask.shape != (batch_size, seq_len):
                raise ValueError("valid_mask must be bool with shape [batch, sequence]")
            if not valid_mask.any(dim=1).all():
                raise ValueError("every sequence must contain at least one valid token")

        x = self.feature_proj(seq)
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if valid_mask is not None:
                valid_mask = torch.cat(
                    [torch.ones((batch_size, 1), dtype=torch.bool, device=seq.device), valid_mask],
                    dim=1,
                )

        padding_mask = None if valid_mask is None else ~valid_mask
        encoded = self.transformer(
            self.pos_encoding(x),
            src_key_padding_mask=padding_mask,
        )

        if self.pooling == "mean":
            if valid_mask is None:
                pooled = encoded.mean(dim=1)
            else:
                weights = valid_mask.unsqueeze(-1)
                pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1)
        elif self.pooling == "max":
            values = encoded if valid_mask is None else encoded.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf)
            pooled = values.max(dim=1).values
        elif self.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return self.reward_head(pooled)
