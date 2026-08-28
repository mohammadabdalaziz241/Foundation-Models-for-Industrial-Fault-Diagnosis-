"""Native PyTorch InceptionTime for the four-domain raw-waveform baseline."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
CLASS_ORDERS = {
    "CWRU": ("inner_race", "outer_race", "ball"),
    "JNU": ("n", "ib", "ob", "tb"),
    "HIT": ("0", "1", "2"),
    "MAFAULDA": (
        "normal", "imbalance", "horizontal-misalignment",
        "vertical-misalignment", "underhang/ball_fault",
        "underhang/cage_fault", "underhang/outer_race",
        "overhang/ball_fault", "overhang/cage_fault",
        "overhang/outer_race",
    ),
}


class SameConv1d(nn.Conv1d):
    """Stride-one convolution with exact same-length asymmetric padding."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = self.dilation[0] * (self.kernel_size[0] - 1)
        left = total // 2
        return super().forward(F.pad(x, (left, total - left)))


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, bottleneck: int = 32,
                 filters: int = 32) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
        self.convs = nn.ModuleList(
            SameConv1d(bottleneck, filters, k, bias=False)
            for k in (40, 20, 10)
        )
        self.pool_conv = nn.Conv1d(in_channels, filters, 1, bias=False)
        self.bn = nn.BatchNorm1d(filters * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        branches = [conv(z) for conv in self.convs]
        branches.append(self.pool_conv(F.max_pool1d(x, 3, stride=1,
                                                    padding=1)))
        return F.relu(self.bn(torch.cat(branches, dim=1)))


class ResidualShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.projection(residual))


class InceptionTimeEncoder(nn.Module):
    """Six modules, residual shortcuts after modules 3 and 6, 128-D GAP."""

    feature_dim = 128

    def __init__(self) -> None:
        super().__init__()
        channels = [1, 128, 128, 128, 128, 128]
        self.modules_ = nn.ModuleList(InceptionModule(c) for c in channels)
        self.shortcuts = nn.ModuleList((ResidualShortcut(1, 128),
                                        ResidualShortcut(128, 128)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for i, module in enumerate(self.modules_):
            x = module(x)
            if i in (2, 5):
                x = self.shortcuts[i // 3](residual, x)
                residual = x
        return x.mean(dim=-1)


class FourDomainInceptionTime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = InceptionTimeEncoder()
        self.heads = nn.ModuleDict({
            ds: nn.Linear(self.encoder.feature_dim, len(CLASS_ORDERS[ds]))
            for ds in DATASETS
        })

    def forward(self, x: torch.Tensor, dataset: str) -> torch.Tensor:
        if dataset not in DATASETS:
            raise AssertionError(f"forbidden/unexpected baseline dataset: {dataset}")
        return self.heads[dataset](self.encoder(x))


# Historical import compatibility only. Corrected runs use the new class and root.
ThreeDomainInceptionTime = FourDomainInceptionTime

ARCHITECTURE = {
    "name": "native_pytorch_inceptiontime",
    "input_channels": 1, "modules": 6, "bottleneck_channels": 32,
    "filters_per_branch": 32, "kernel_sizes": [40, 20, 10],
    "pool_branch": "MaxPool1d(k=3,s=1,same)+Conv1d(k=1)",
    "post_concat": "BatchNorm1d(128)+ReLU",
    "residual_every_modules": 3, "global_average_pooling": True,
    "feature_dim": 128, "heads": {"CWRU": 3, "JNU": 4, "HIT": 3, "MAFAULDA": 10},
}
