from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
import torch
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork

class Encoder(nn.Module):

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        if self.inc:
            x = self.inc(x)

        encoded_feature_maps = OrderedDict()
        for idx, encoding_block in enumerate(self.encoding_blocks):
            x = encoding_block(x)
            encoded_feature_maps[f"p{idx + 2}"] = x

        return encoded_feature_maps

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

class ResnetEncoder(Encoder):
    def __init__(self, model: str = "resnet18", in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        if model == "resnet18":
            self.backbone = models.resnet18(pretrained)
            self.out_channels = [64, 128, 256, 512]
        elif model == 'resnet34':
            self.backbone = models.resnet34(pretrained)
            self.out_channels = [64, 128, 256, 512]
        elif model == 'resnet50':
            self.backbone = models.resnet50(pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        elif model == 'resnet101':
            self.backbone = models.resnet101(pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        elif model == 'resnet152':
            self.backbone = models.resnet152(pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        elif model == 'resnext50':
            self.backbone = models.resnext50_32x4d(pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"{model} is not supported")

        self.down_dim = [4, 8, 16, 32]

        if pretrained:
            self.backbone.eval()
            
        if in_channels != 3 and pretrained:
            raise ValueError("Pretrained models only support RGB input")
        elif in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.inc = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )

        self.encoding_blocks = nn.ModuleList(
            [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
        )


class DensenetEncoder(Encoder):
    def __init__(self, model: str = "densenet121", in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        if name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            self.out_channels = []
        elif name == 'densenet161':
            self.backbone = models.densenet161(pretrained=pretrained)
            self.out_channels = []
        elif name == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
            self.out_channels = []
        elif name == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            self.out_channels = []
        else:
            raise ValueError(f"{model} not supported")

        self.down_dim = []

        self.inc = nn.Sequential(
            )

        self.encoding_blocks = nn.ModuleList(
            [
                self.backbone.denseblock1,
                self.backbone.denseblock2,
                self.backbone.denseblock3,
                self.backbone.denseblock4
            ]
        )


