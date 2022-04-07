from __future__ import annotations

import torch.nn as nn
import torch
from torch import Tensor
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

from encode import ResNetEncoder, DenseNetEncoder, UNetEncoder
from model_blocks import *


class Unet(nn.Module):
    name: str = "unet"

    def __init__(self, n_channels, n_classes, backbone: str, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = Down(n_channels, 64)
        self.down1(64, 128)
        self.down2(128, 256)
        self.down3(256, 512)
        self.down4(512, 1024)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = Conv1x1(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

class SoloV2(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, fpn_out: int) -> None:
        super().__init__()
        self.backbone = ResNetEncoder(backbone, pretrained)
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, fpn_out)
        self.kernel_head = KernelHead()
        self.feature_head = FeatureHead()
        self.mask_head = MaskHead()
        self.nms = MatrixNMS()

    def forward(self, x: Tensor) -> Tensor:
        encoded_feature_maps = self.backbone(x)
        fpn_feature_maps = self.fpn(encoded_feature_maps)
        mask_kernel = self.kernel_head(fpn_feature_maps)
        features = self.feature_head(fpn_feature_maps)
        mask = self.mask_head(mask_kernel, features)
        return self.nms(mask)
    
