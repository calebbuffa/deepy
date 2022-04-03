from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
import torch
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork

def Encoder(nn.Module):
    
    def forward(self, x: Tensor) -> OrderedDict[Tensor]:
        encoded_feature_maps = OrderedDict()
        for idx, encoding_block in enumerate(self.encoding_blocks):
            x = encoding_block(x)
            encoded_feature_maps[f"x{idx}"] = x

        return encoded_feature_maps

class ResNetEncoder(Encoder):
    def __init__(self, model: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if model == "resnet18":
            resnet = models.resnet18(pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif model == 'resnet34':
            resnet = models.resnet34(pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif model == 'resnet50':
            resnet = models.resnet50(pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnet101':
            resnet = models.resnet101(pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnet152':
            resnet = models.resnet152(pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnext50':
            resnet = models.resnext50_32x4d(pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"{model} is not supported")

        if pretrained:
            resnet.eval()

        self.encoding_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu
                ),
                nn.Sequential(
                    resnet.maxpool,
                    resnet.layer1
                ),
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            ]
        )

    
class DenseNetEncoder(Encoder):
    def __init__(self, model: str = "densenet121", pretrained: bool = True):
        super().__init__()
        elif name == 'densenet121':
            backbone = models.densenet121(pretrained=pretrained)
            self.out_channels = []
        elif name == 'densenet161':
            backbone = models.densenet161(pretrained=pretrained)
            self.out_channels = []
        elif name == 'densenet169':
            backbone = models.densenet169(pretrained=pretrained)
            self.out_channels = []
        elif name == 'densenet201':
            backbone = models.densenet201(pretrained=pretrained)
            self.out_channels = []


class UNetEncoder(Encoder):
    def __init__(self, in_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        factor = 2 if bilinear else 1
        self.encoding_blocks = nn.ModuleList(
            [
                DoubleConv(in_channels, 64),
                Down(64, 128),
                Down(128, 256),
                Down(256, 512),
                Down(512, 1024 // factor),
             ]
        )
        self.out_channels = [64, 128, 256, 512, 1024]

