"""Model heads."""

from __future__ import annotations

import torch
from collections import OrderedDict
from torch import Tensor, nn

from model_blocks import FPNUpSample, Conv3x3, Conv1x1, UpSample

class SemanticFPNHead(nn.Module):
    def __init__(self, in_channels: int = 256, out_channels: int = 128, n_classes: int = 2):
        """
        Semantic Feature Pyramid Network.
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Feature_Pyramid_Networks_CVPR_2019_paper.pdf

        Parameters
        ----------
        in_channels : int 
            Number of channels out by Feature Pyramid Network, defaults to 256.
        out_channels : int
            Number of output channels per upsampling, defaults to 128.
        n_classes : int
            Number of clases to predict. Must be greater than 2, Defaults to 2.
        """
        super().__init__()
        mid_channels = in_channels - int((in_channels - out_channels) * 0.5)
        self.up1 = nn.Sequential(
            FPNUpSample(
                in_channels=in_channels, out_channels=mid_channels
            ),
            FPNUpSample(
                in_channels=mid_channels,
                out_channels=mid_channels
            ),
            FPNUpSample(
                in_channels=mid_channels, 
                out_channels=out_channels
            ),
        )
        self.up2 = nn.Sequential(
            FPNUpSample(
                in_channels=in_channels, out_channels=mid_channels
            ),
            FPNUpSample(
                in_channels=mid_channels,
                out_channels=out_channels
            ),
        )
        self.up3 = FPNUpSample(in_channels=in_channels, out_channels=out_channels)
        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)

        self.outc = nn.Sequential(
            UpSample(
                in_channels=out_channels * 4, 
                out_channels=n_classes, factor=4, bilinear=True
            ),
            Conv1x1(in_channels=out_channels * 4, out_channels=n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, feature_maps: OrderedDict[str, Tensor]) -> Tensor:
        x1 = self.up1(feature_maps["p5"])
        x2 = self.up2(feature_maps["p4"])
        x2 += x1
        x3 = self.up3(feature_maps["p3"])
        x3 += x2
        x4 = self.conv(feature_maps["p2"])
        x4 += x3

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return self.outc(x)


class KernelHead(nn.Module):
    """SoloV2 Kernel Head"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError

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

class SegmentationHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError
    
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
