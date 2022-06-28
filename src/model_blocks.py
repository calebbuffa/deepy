"""Helpful blocks for larger models."""

from __future__ import annotations

import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

from encode import ResnetEncoder
from utils import Conv3x3, Conv1x1, AtrousConv3x3, Upsample, AtrousConvBNAct

class FPNUpsample(nn.Module):
    """
    One upsampling block for feature pyramid network.
    """

    def __init__(
        self, in_channels: int = 256, out_channels: int = 256, **kwargs
    ):
        """
        Upsampling block for SemanticFPN.

        Parameters
        ----------
        in_channels : int
            Number of input channels, defaults to 256.
        out_channels : int
            Number of output channels, defaults to 256.
        kwargs : Any
        """
        super().__init__()
        self.block = nn.Sequential(
            Conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )
        
        self.up = Upsample(
            out_channels, out_channels, factor=2, bilinear=kwargs.get("bilinear", True)
        )

        self.residual = kwargs.get("residual", False)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.block(x)
        x1 = x1 + x if self.residual and x1.shape == x.shape else x1
        return self.up(x1)



class AtrousSpatialPyramidPooling(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    dilation_rates = [6, 12, 18]

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv1x1(in_channels, out_channels)
        self.asp1 = AtrousConvBNAct(in_channels, out_channels, self.dilation_rates[0])
        self.asp2 = AtrousConvBNAct(in_channels, out_channels, self.dilation_rates[1])
        self.asp3 = AtrousConvBNAct(in_channels, out_channels, self.dilation_rates[2])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv1x1(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
        )
        self.outc = Conv1x1(out_channels * 5, out_channels)

    def forward(self, x: Tensor) -> Tensor:
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
        in_channels: int = 3,
        pretrained: bool = True,
        fpn_out_channels: int = 256,
    ):
        super().__init__()
        self.encoder = ResnetEncoder(model=backbone, pretrained=pretrained)
        if pretrained:
            self.encoder.freeze()

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.encoder.out_channels, 
            out_channels=fpn_out_channels,
            in_channels=in_channels
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fpn(self.encoder(x))

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, r: int = 2):
        """
        Self attention block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        r : int
            Ratio of intermediate channels, defaults to 2.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            Conv1x1(in_channels, in_channels // r),
            nn.ReLU(),
            Conv1x1(in_channels // r, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        pooled = self.pool(x)
        weights = self.fc(pooled)
        return x * weights


class AttentionGatingBlock(nn.Module):
    def __init__(self, g_in_channels: int, x_in_channels: int):
        """
        Attention Gating Block.

        Parameters
        ----------
        g_in_chanels : int
            Number of input channels from tensor of shape [B, N, w, h]
        x_in_channels : int
            Number of input channels from tensor of shape [B, n, W, H]
        """
        super().__init__()
        self.g_conv = Conv1x1(g_in_channels, g_in_channels)
        self.x_conv = Conv1x1(x_in_channels, g_in_channels, stride=2)
        self.activate = nn.Sequential(
            Conv1x1(g_in_channels, 1),
            nn.ReLU(),
            nn.Sigmoid(),
        )
    
    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Tensor from skip connection.
        g : Tensor
            Incoming tensor from lower level.

        Returns
        -------
        Tensor
            Tensor of the same shape as x.
        """
        x1 = self.x_conv(x)
        g1 = self.g_conv(g)
        xg = x1 + g1
        xg = self.activate(xg)
        x_weights = F.interpolate(
            xg, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        return x * x_weights
