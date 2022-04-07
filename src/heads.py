import torch.nn as nn
from collections import OrderedDict
from torch import tensor

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
