"""Utility classes."""

from __future__ import annotations

from torch import Tensor, nn
import torch

def get_model_size(model: nn.Module) -> tuple[int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{trainable} trainable parameters")
    print(f"{total} total parameters")
    return trainable, total


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        """
        1x1 2D convolution.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kwargs : Any
            Keyword arguments to pass to nn.Conv2d
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        """
        3x3 2D convolution.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kwargs : Any
            Keyword arguments to pass to nn.Conv2d
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=kwargs.get("padding", 1), **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class AtrousConv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation_rate: int, **kwargs):
        """
        3x3 dilated 2D convolution.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        dilation_rate : int
            Rate of dilation for convolution.
        kwargs : Any
            Keyword arguments to pass to Conv3x3
        """
        out_channels : int
        super().__init__()
        self.conv = Conv3x3(
            in_channels,
            out_channels,
            padding=dilation_rate,
            dilation=dilation_rate,
            **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int, bilinear: bool = False, **kwargs):
        """
        2D bilinear interpolation or transposed convolution upsampling.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        factor : int
            Factor to upsample H and W dimensions.
        bilinear : bool
            If True, bilinear upsampling, defaults to False.
        kwargs : Any
            Keyword arguments to pass to nn.UpSample or nn.ConvTranspose2d
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=factor, stride=factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(x)

class AtrousConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rate: int):
        """
        Atrous 3x3 convolution, batch norm, and relu activation.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        rate : int
            Dilation rate for AtrousConv3x3
        """
        super().__init__()
        self.block = nn.Sequential(
             AtrousConv3x3(in_channels, out_channels, rate, bias=False),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
