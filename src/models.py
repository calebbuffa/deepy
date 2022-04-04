from __future__ import annotations

import torch.nn as nn
import torch
from torch import Tensor
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

from encode import ResNetEncoder, DenseNetEncoder, UNetEncoder
from model_blocks import *


class UNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        in_channels: int = 3,
        **kwargs
    ):
        super().__init__()
        bilinear = kwargs.get("bilinear", False)
        factor = 2 if bilinear else 1
        self.n_classes = n_classes
        self.encoder = UNetEncoder(in_channels=in_channels, bilinear=bilinear)

        self.decoder = nn.ModuleList()
        out_channels = self.encoder.out_channels[::-1]
        for i in range(len(out_channels)):
            in_channel = self.encoder.out_channels[i]
            try:
                out_channel = out_channelsi + 1]
                self.decoder.append(Up(in_channel, out_channel // factor, bilinear))
            except IndexError:
                out_channel = n_classes
                self.outc = OutConv(in_channel, n_classes)

            print(in_channel, out_channel)

    def forward(self, x: Tensor) -> Tensor:
        encoded_feature_maps = self.encoder(x)
        x = self.decoder[0](encoded_feature_maps["x5"], encoded_feature_maps["x4"])
        x = self.decoder[1](x, encoded_feature_maps["x3"])
        x = self.decoder[2](x, encoded_feature_maps["x2"])
        x = self.decoder[3](x, encoded_feature_maps["x1"])
        out = self.outc(x)
        return out

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
