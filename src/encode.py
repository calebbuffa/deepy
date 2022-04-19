from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
import torch
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork

class Encoder(nn.Module):
    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        encoded_feature_maps = OrderedDict()
        if self.inc:
            x = self.inc(x)
        for idx, encoding_block in enumerate(self.encoding_blocks):
            x = encoding_block(x)
            encoded_feature_maps[f"p{idx + 2}"] = x

        return encoded_feature_maps

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

class ResnetEncoder(Encoder):
    def __init__(
        self, model: str = "resnet18", pretrained: bool = True, in_channels: int = 3
    ):
        super().__init__()
        self.backbone = getattr(models, model)(pretrained=pretrained)
        if model in {"resnet18", "resnet34"}:
            self.out_channels = [64, 128, 256, 512]
        elif model in {
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
        }:
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"{model} is not supported")

        self.down_dimensions = [4, 8, 16, 32]

        if pretrained:
            self.backbone.eval()

        if in_channels != 3 and pretrained:
            raise ValueError("Pretrained models only support RGB input")
        elif in_channels !=3:
            self.backbone.conv1.in_channels = in_channels

        self.inc = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
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
    def __init__(
        self, model: str = "densenet121", in_channels: int = 3, pretrained: bool = True
    ):
        super().__init__()
        self.backbone = getattr(models, model)(pretrained=pretrained)

        self.down_dimensions = [4, 8, 16, 16]
        self.out_channels = [128, 256, 512, 1024]

        if pretrained:
            self.backbone.eval()

        if in_channels != 3 and pretrained:
            raise ValueError("Pretained models only support RGB input")
        elif in_channels != 3:
            self.backbone.features.conv0.in_channels = in_channels

        self.inc = nn.Sequential(
            self.backbone.features.conv0,
            self.backbone.features.norm0,
            self.backbone.features.relu0,
        )

        layer1 = nn.Sequential(
            self.backbone.features.denseblock1, self.backbone.features.transition1,
        )
        layer2 = nn.Sequential(
            self.backbone.features.denseblock2, self.backbone.features.transition2,
        )
        layer3 = nn.Sequential(
            self.backbone.features.denseblock3, self.backbone.features.transition3,
        )
        layer4 = nn.Sequential(
            self.backbone.features.denseblock4, self.backbone.features.norm5,
        )

        self.encoding_blocks = nn.ModuleList([layer1, layer2, layer3, layer4,])


