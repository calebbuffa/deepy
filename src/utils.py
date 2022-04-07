from __future__ import annotations

from torch import Tensor
iport torch

class DistanceMetric:
    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Euclidian(DistanceMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, x: Tensor) -> Tensor:
        """
        K nearest neighbor search using euclidian distance.

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
        super().__init__(*args, **kwargs)

    def __call__(self, x: Tensor) -> Tensor:
        """
        K nearest neighbor search using cosine similarity.

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
    
class Atrous3x3Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, **kwargs):
        super().__init__()
        self.conv = Conv3x3(
            in_channels,
            out_channels,
            padding=dilation_rate,
            dilation=dilation_rate,
            **kwargs
        )

    def forward(self, x):
        return self.conv(x)
    
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int, bilinear: bool = False, **kwargs):
        if bilinear:
            self.up = nn.Upsample(scale_factor=factor, mode=kwargs.get("mode", "bilinear"), align_corners=True)
         else:
            self.up = nn.ConvTraponse2d(
                in_channels, out_channels // 2), kernel_size=factor, stride=factor)
            
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

