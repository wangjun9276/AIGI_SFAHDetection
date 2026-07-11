"""Minimal Xception backbone used by the two-stage detector."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False) if out_filters != in_filters or strides != 1 else None
        self.skip_bn = nn.BatchNorm2d(out_filters) if self.skip is not None else None
        layers = []
        filters = in_filters
        if grow_first:
            layers += [nn.ReLU(inplace=False), SeparableConv2d(in_filters, out_filters, 3, padding=1), nn.BatchNorm2d(out_filters)]
            filters = out_filters
        for _ in range(reps - 1):
            layers += [nn.ReLU(inplace=False), SeparableConv2d(filters, filters, 3, padding=1), nn.BatchNorm2d(filters)]
        if not grow_first:
            layers += [nn.ReLU(inplace=False), SeparableConv2d(in_filters, out_filters, 3, padding=1), nn.BatchNorm2d(out_filters)]
        if not start_with_relu:
            layers = layers[1:]
        if strides != 1:
            layers.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.skip_bn(self.skip(x)) if self.skip is not None else x
        return self.rep(x) + residual


class Xception(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)
        self.block4 = Block(728, 728, 3)
        self.block5 = Block(728, 728, 3)
        self.block6 = Block(728, 728, 3)
        self.block7 = Block(728, 728, 3)
        self.block8 = Block(728, 728, 3)
        self.block9 = Block(728, 728, 3)
        self.block10 = Block(728, 728, 3)
        self.block11 = Block(728, 728, 3)
        self.block12 = Block(728, 1024, 2, 2, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block3(self.block2(self.block1(x)))
        x = self.block7(self.block6(self.block5(self.block4(x))))
        x = self.block12(self.block11(self.block10(self.block9(self.block8(x)))))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.bn4(self.conv4(x))

    def classifier(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.last_linear(x)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)
