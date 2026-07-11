"""Frequency-channel attention retained from the original hybrid model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _frequency_indices(method):
    if method != "bot16":
        raise ValueError("This cleaned implementation supports the used setting freq_sel_method='bot16' only.")
    xs = [6, 6, 5, 6, 5, 4, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
    ys = [6, 5, 6, 4, 5, 6, 6, 5, 4, 3, 2, 3, 4, 5, 6, 6]
    return xs, ys


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channels):
        super().__init__()
        if channels % len(mapper_x) != 0:
            raise ValueError("channels must be divisible by the number of selected frequencies")
        self.register_buffer("weight", self._build_filter(height, width, mapper_x, mapper_y, channels), persistent=True)

    @staticmethod
    def _basis(pos, freq, size):
        value = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
        return value if freq == 0 else value * math.sqrt(2)

    def _build_filter(self, height, width, mapper_x, mapper_y, channels):
        weight = torch.zeros(channels, height, width)
        group = channels // len(mapper_x)
        for i, (u, v) in enumerate(zip(mapper_x, mapper_y)):
            for x in range(height):
                for y in range(width):
                    weight[i * group:(i + 1) * group, x, y] = self._basis(x, u, height) * self._basis(y, v, width)
        return weight

    def forward(self, x):
        return (x * self.weight).sum(dim=(2, 3))


class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channels=2048, dct_h=7, dct_w=7, reduction=4, freq_sel_method="bot16"):
        super().__init__()
        mapper_x, mapper_y = _frequency_indices(freq_sel_method)
        self.dct_h, self.dct_w = dct_h, dct_w
        self.dct = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channels)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        pooled = x if x.shape[-2:] == (self.dct_h, self.dct_w) else F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        weights = self.fc(self.dct(pooled)).view(x.shape[0], x.shape[1], 1, 1)
        return x * weights
