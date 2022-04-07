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
    
class FPNSegmentation(nn.Module):
    name: str = "fpn_segmentation"

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        decode_channels: int = 128,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.fpn = FPN(
            backbone=backbone, pretrained=pretrained, fpn_out_channels=fpn_out_channels
        )

        # for 1/4 scale features
        self.head1 = nn.Sequential(
            Conv3x3(fpn_out_channels, decode_channels),
            Conv3x3(decode_channels, decode_channels),
        )
        self.head2 = nn.Sequential(
            UpSample(
                fpn_out_channels,
                fpn_out_channels,
                factor=self.fpn.encoder.down_dimensions[1]
                // self.fpn.encoder.down_dimensions[0],
            ),
            Conv3x3(fpn_out_channels, decode_channels),
            Conv3x3(decode_channels, decode_channels),
        )
        self.head3 = nn.Sequential(
            UpSample(
                fpn_out_channels,
                fpn_out_channels,
                factor=self.fpn.encoder.down_dimensions[2]
                // self.fpn.encoder.down_dimensions[0],
            ),
            Conv3x3(fpn_out_channels, decode_channels),
            Conv3x3(decode_channels, decode_channels),
        )
        self.head4 = nn.Sequential(
            UpSample(
                fpn_out_channels,
                fpn_out_channels,
                factor=self.fpn.encoder.down_dimensions[3]
                // self.fpn.encoder.down_dimensions[0],
            ),
            Conv3x3(fpn_out_channels, decode_channels),
            Conv3x3(decode_channels, decode_channels),
        )
        # concatenated features
        self.outc = nn.Sequential(
            UpSample(
                in_channels=len(self.fpn.encoder.down_dimensions) * decode_channels,
                out_channels=len(self.fpn.encoder.down_dimensions) * decode_channels,
                factor=self.fpn.encoder.down_dimensions[0],
            ),
            Conv3x3(in_channels=4 * decode_channels, out_channels=2 * decode_channels,),
            Conv1x1(in_channels=2 * decode_channels, out_channels=n_classes,),
        )

    def forward(self, x: Tensor) -> Tensor:
        features: OrderedDict[str, Tensor] = self.fpn(x)

        p2 = self.head1(features["p2"])  # -> [B, 128, H/4, W/4]
        p3 = self.head2(features["p3"])  # -> [B, 128, H/4, W/4]
        p4 = self.head3(features["p4"])  # -> [B, 128, H/4, W/4]
        p5 = self.head4(features["p5"])  # -> [B, 128, H/4, W/4]

        x = torch.cat([p2, p3, p4, p5], dim=1)  # -> [B, 512, H/4, W/4]
        return self.outc(x)  # -> [B, num_classes, H, W]

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
    
