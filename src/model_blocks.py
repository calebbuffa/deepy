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

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
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

    def forward(self, x1, x2):
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

    def forward(self, x):
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

class MatrixNMS(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        return

class MaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        ...
    
    def forward(self, mask_kernel, features):
        # convolve mask kernel over features
        return


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        ...
    
    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        ...

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        ...
