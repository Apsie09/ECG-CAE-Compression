"""Compact 1D convolutional autoencoder.

Encoder (the part meant for the ESP32-S3) uses only Conv1d, ReLU, Flatten and Linear.
With padding = k // 2 each stride-2 conv halves the length: 256 -> 128 -> 64 -> 32.
The latent vector and the output have no activation (ECG is signed, roughly zero-mean).
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, channels=(8, 16, 32), kernel_size: int = 5, window: int = 256):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv1d(c_in, c, kernel_size, stride=2, padding=kernel_size // 2), nn.ReLU()]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.out_len = window // 2 ** len(channels)
        self.proj = nn.Linear(channels[-1] * self.out_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B, 1, W] -> [B, z]
        return self.proj(self.features(x).flatten(1))


class Decoder(nn.Module):
    """Linear projection, then (upsample x2 -> Conv1d -> ReLU) per stage, then a 1-channel conv.
    Upsample + conv instead of ConvTranspose1d avoids checkerboard artefacts."""

    def __init__(self, latent_dim: int, channels=(8, 16, 32), kernel_size: int = 5, window: int = 256):
        super().__init__()
        self.c0 = channels[-1]
        self.l0 = window // 2 ** len(channels)
        self.proj = nn.Linear(latent_dim, self.c0 * self.l0)
        rev = list(channels[::-1])                    # 32, 16, 8
        outs = rev[1:] + [rev[-1]]                    # 16, 8, 8
        layers = []
        for c_in, c_out in zip(rev, outs):
            layers += [nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                       nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2), nn.ReLU()]
        layers.append(nn.Conv1d(outs[-1], 1, kernel_size, padding=kernel_size // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:        # [B, z] -> [B, 1, W]
        return self.net(self.proj(z).view(-1, self.c0, self.l0))


class CAE(nn.Module):
    def __init__(self, latent_dim: int = 16, channels=(8, 16, 32), kernel_size: int = 5, window: int = 256):
        super().__init__()
        if window % 2 ** len(channels):
            raise ValueError("window must be divisible by 2**len(channels)")
        self.encoder = Encoder(latent_dim, channels, kernel_size, window)
        self.decoder = Decoder(latent_dim, channels, kernel_size, window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_model(cfg: dict) -> CAE:
    m = cfg["model"]
    return CAE(m["latent_dim"], tuple(m["channels"]), m["kernel_size"], cfg["window"]["size"])


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
