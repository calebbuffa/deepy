from __future__ import annotations

from torch import Tensor, nn
import torch

class DistanceMetric:
    def __init__(self, k: int) -> None:
        """
        Parameters
        ----------
        k : int
            K neighbors to return.
        """
        self.k = k

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Euclidian(DistanceMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        K nearest neighbor search using euclidian distance.
        """

    def __call__(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor. Of shape (B, C, N).

        Returns
        -------
        Tensor
            Indices of the k-nearest neighbors of each point.
        """
        inner = -2 * torch.matmul(x.permute(0, 2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.permute(0, 2, 1)

        return pairwise_distance.topk(k=self.k, dim=-1)[1]


class Cosine(DistanceMetric):
    def __init__(self, *args, **kwargs) -> None:
        """
        K nearest neighbor search using cosine similarity.
        """
        super().__init__(*args, **kwargs)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor. Of shape (B, C, N).

        Returns
        -------
        Tensor
            Indices of the k-nearest neighbors of each point.
        """
        norm = (x * x).sum(1, keepdims=True) ** 0.5
        x_norm = x / norm
        similarity_matrix = x_norm.transpose(2, 1) @ x_norm

        return torch.topk(similarity_matrix, k=self.k, axis=-1)[1]

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

class Atrous3x3Conv(nn.Module):
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

class UpSample(nn.Module):
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

class ASPPHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rate: int):
        """
        One Atrous Spatial Pyramid Poooling Head. Consists of Atrous 3x3 
        convolution, batch norm, and relu activation.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        rate : int
            Dilation rate for Atrou3x3Conv
        """
        super().__init__()
        self.block = nn.Sequential(
             Atrous3x3Conv(in_channels, out_channels, rate, bias=False),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

