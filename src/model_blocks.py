from __future__ import annotations

import torch.nn as nn
import torch
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
     
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
    
class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=self.kwargs.get("padding", 1), **kwargs)
        
     def forward(self, x: Tensor) -> Tensor:
        return selv.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv3x3(in_channels, mide_channels, bias=False)
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            Conv3x3(mid_channels, out_channels, bias=False)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int, bilinear: bool = False, **kwargs):
        if bilinear:
            self.up = nn.Upsample(scale_factor=factor, mode=kwargs.get("mode", "bilinear"), align_corners=True)
         else:
            self.up = nn.ConvTraponse2d(
                in_channels, out_channels // 2), kernel_size=factor, stride=factor)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.up = Upsample(in_channels, out_channels, factor=2, bilinear)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // factor)

    def forward(self, x1: Tensor, x2: Tensor) -> tensor:
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Atrous3x3Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, **kwargs):
        super(Atrous3x3Conv, self).__init__()
        self.conv = Conv3x3(
            in_channels,
            out_channels,
            padding=dilation_rate,
            dilation=dilation_rate,
            **kwargs
        )

    def forward(self, x):
        return self.conv(x)

class ASPPHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rate: int):
        super().__init__()
        self.block = nn.Sequential(
             Atrous3x3Conv(in_channels, out_channels, rate, bias=False),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)



class ASPP(nn.Module):
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


class SkipConnection:
    def __init__(self, method: str = "residual") -> None:
        assert method in {"residual", "dense"}
        self.method = method

    def __call__(self, x1: Tensor, x2: Tensor) -> Tensor:
        if self.method == "residual":
            diff_c = (x1.shape[1] - x2.shape[1]) // 2
            return x1 + F.pad(x2, (0, 0, diff_c, diff_c))
        elif self.method == "dense":
            return torch.cat([x1, x2], dim=1)
        else:
            raise ValueError
