"""Models."""

from __future__ import annotations

import torch.nn as nn
import torch
from torch import Tensor
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

from encode import ResnetEncoder, DensenetEncoder
from heads import SemanticFPNHead
from model_blocks import FPN
    
class SemanticFPN(nn.Module):
    name: str = "semantic_fpn"

    def __init__(
        self,
        backbone: str = "resnet18",
        n_classes: int = 2,
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        decode_out_channels: int = 128,
        residual: bool = False,
        attention: bool = False
    ):
        """
        Semantic segmentation with Feature Pyramid Network.
        
        Parameters
        ----------
        backbone : str
            Encoder to use, defaults to 'resnet18'.
        n_classes : int
            Number of classes to predict. Must be greater than 2, defaults to 2.
        pretrained : bool
            Whether to use pretrained encoder, defaults to True.
        fpn_out_channels : int
            Number of channels outputted by the Feature Pyramid Network,
            defaults to 256.
        decode_out_channels : int
            Number of channels after upsampling, defaults to 128.
        residual : bool
            Whether to apply residual skip connections in every upsampling block,
            defaults to False.
        attention : bool
            Whether to apply self attention via squeeze and excitation blocks
            after every upsamplingn block, defaults to False.
        """
        super().__init__()
        self.n_classes = n_classes

        self.fpn = FPN(
            backbone=backbone, 
            pretrained=pretrained, 
            fpn_out_channels=fpn_out_channels,
        )

        self.head = SemanticFPNHead(
            in_channels=fpn_out_channels, 
            out_channels=decode_out_channels, 
            n_classes=n_classes,
            residual=residual,
            attention=attention
        )

    def forward(self, x: Tensor) -> Tensor:
        features: OrderedDict[str, Tensor] = self.fpn(x)
        return self.head(features)


class SoloV2(nn.Module):
    name: str = "solov2"
    def __init__(self, backbone: str, pretrained: bool, fpn_out_channels: int) -> None:
        super().__init__()
        self.kernel_head = KernelHead()
        self.mask_head = SemanticFPN(
            backbone=backbone, 
            pretrained=pretrained, 
            fpn_out_channels=fpn_out_channels
        )
        self.nms = MatrixNMS()

    def forward(self, x: Tensor) -> Tensor:
        mask_kernel = self.kernel_head(fpn_feature_maps)
        mask = self.mask_head(mask_kernel, x)
        return self.nms(mask)
    
