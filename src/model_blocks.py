"""Helpful blocks for larger models."""

from __future__ import annotations

import torch.nn as nn
import torch
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

from encode import ResnetEncoder
from utils import Conv3x3, Conv1x1, Atrous3x3Conv, UpSample, ASPPHead

class FPNUpSample(nn.Module):
    """
    One upsampling block for feature pyramid network.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            Conv3x3(in_channels, out_channels),
            nn.GroupNorm(
                num_groups=kwargs.get("num_groups", 4), 
                num_channels=out_channels
            ),
            nn.ReLU(),
            UpSample(out_channels, out_channels, factor=2, bilinear=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)



class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    dilation_rates = [6, 12, 18]

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv1x1(in_channels, out_channels)
        self.asp1 = ASPPHead(in_channels, out_channels, self.dilation_rates[0])
        self.asp2 = ASPPHead(in_channels, out_channels, self.dilation_rates[1])
        self.asp3 = ASPPHead(in_channels, out_channels, self.dilation_rates[2])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            COnv1x1(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
        )
        self.outc = Conv1x1(in_channels * 5, out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.asp1(x)
        x3 = self.asp2(x)
        x4 = self.asp3(x)
        x5 = self.pool(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.outc(x)


class FPN(nn.Module):
    """Feature Pyramid Network with backbone."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        fpn_out_channels: int = 256,
    ):
        super().__init__()
        if "res" in backbone:
            self.encoder = ResnetEncoder(model=backbone, pretrained=pretrained)
        else:
            raise ValueError(f"{backbone} is not supported")

        if pretrained:
            self.encoder.freeze()

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.encoder.out_channels, out_channels=fpn_out_channels
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fpn(self.encoder(x))
