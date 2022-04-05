from __future__ import annotations

import torch.nn as nn
import torch
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> tensor:
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

 
class KernelHead(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, fpn_features: OrderedDict[Tensor]) -> Tensor:
        """
        first resize the input feature FI ∈ RHI ×WI ×C into shape of S × S × C. 
        Then 4×convs and a final 3 × 3 × D conv are employed to generate the kernel G. 
        We add the spatial functionality to FI by giving the first convolution access to
        the normalized coordinates following CoordConv [23], i.e., concatenating two
        additional input channels which contains pixel coordinates normalized to 
        [−1, 1]. Weights for the head are shared across different feature map levels.
        For each grid, the kernel branch predicts the D-dimensional output to indicate
        predicted convolution kernel weights, where D is the number of parameters. 
        For generating the weights of a 1×1 convolution with E input channels, D equals
        E. As for 3×3 convolution, D equals 9E. These generated weights re conditioned
        on the locations, i.e., the grid cells. If we divide the input image into S×S
        grids, the output space will be S×S×D, There is no activation function on the 
        output.
        """
        return 

class FeatureHead(nn.Module):
    def __init__(self):
        super().__init__()
        ...
    
    def forward(self, fpn_features: OrderedDict[Tensor]) -> Tensor:
        """
        After repeated stages of 3 × 3 conv, group norm [ 34 ], ReLU and 2× bilinear 
        upsampling, the FPN features P2 to P5 are merged into a single output at 1/4 
        scale. The last layer after the element-wise summation consists of 1 × 1 
        convolution, group norm and ReLU. More details can be referred to supplementary 
        material. It should be noted that we feed normalized pixel coordinates to the 
        deepest FPN level (at 1/32 scale), before the convolutions and bilinear 
        upsamplings.
        """
        return


class MaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        ...
    
    def forward(self, mask_kernel, features):
        # convolve mask kernel over features
        return
    
class GraphFeature(nn.Module):
    def __init__(self, k: int = 20, method: DistanceMetric = Euclidian):
        """
        Initialize DGCNN graph feature.

        Parameters
        ----------
        k : int
            Number of neighbors, defaults to 20.
        dim9 : bool
            Whether to use 9D feature, defaults to False.
        """
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.knn = method(k)
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        """
        Dynamically update DGCNN graph feature.

        Parameters
        ----------
        x : Tensor
            Input tensor. Of shape (batch, features, points)

        Returns
        -------
        Tensor
            Updated graph feature of shape (batch_size, 2*features, num_points, k).
        """
        batch_size, _, num_points = x.size()
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)

        idx = self.knn(x[:, :3])

        idx_base = (
            torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
        )

        idx += idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        # (batch_size, num_points, features)  -> (batch_size*num_points, features)
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature  # (batch_size, 2*features, num_points, k)
 
class Convolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        conv_dim: int = 2,
    ) -> None:
        super().__init__()
        conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
        bn = nn.BatchNorm2d if conv_dim == 2 else nn.BatchNorm1d
        self.conv = (
            nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1, bias=False),
                bn(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )
            if batch_norm
            else nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class EdgeConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        method: DistanceMetric,
        n_convs: int,
        batch_norm: bool = True,
        conv_dim: int = 2,
    ) -> None:
        super().__init__()
        modules = nn.ModuleList()
        modules.append(GraphFeature(k=k, method=method))
        for i in range(n_convs):
            in_channels = in_channels if i == 0 else out_channels
            modules.append(Convolution(in_channels, out_channels, batch_norm, conv_dim))

        self.edge_conv = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.edge_conv(x)

    class Pooling(nn.Module):
    def __init__(
        self,
        method: str = "max",
        keep_dim: bool = False,
        adaptive: bool = False,
        size: tuple = None,
    ) -> None:
        super().__init__()
        self.method = method

        if adaptive:
            assert size is not None

        self.adaptive = adaptive
        self.size = size
        self.keep_dim = keep_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.method == "max" and not self.adaptive:
            return x.max(dim=-1, keepdim=self.keep_dim)[0]
        elif self.method == "mean" and not self.adaptive:
            return x.mean(dim=-1, keepdim=self.keep_dim)[0]
        elif self.method == "max":
            return F.adaptive_max_pool2d(x, self.size)  # B, C, 1
        elif self.method == "mean":
            return F.adaptive_avg_pool2d(x, self.size)  # B, C, 1


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


class Concatenate:
    def __call__(self, *args) -> Tensor:
        return torch.cat(args, dim=1)


class Repeat:
    def __call__(self, x: Tensor, n: int) -> Tensor:
        return x.repeat(1, 1, n)
