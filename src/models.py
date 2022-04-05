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
    
    
class DGCNN(nn.Module):
    def __init__(self, num_classes: int, k: int, emb_dims: int, dropout: float) -> None:
        """
        Initialize Dynamic Graph CNN.

        Parameters
        ----------
        num_classes : int
            Number of classes to predict.
        k : int
            K nearest neighbors to update graph.
        emb_dims : int
            Embedding dimensions.
        dropout : float
            Percentage of nodes to drop.

        Example
        -------
        >>> from p2p3d.models import DGCNN
        >>> model = DGCNN(num_classes=2, k=20, emb_dims=64, dropout=0.5)
        """
        super().__init__()
        self.embedding_block_1 = nn.Sequential(
            EdgeConv(
                in_channels=6 * 2,
                out_channels=64,
                k=k,
                method=Euclidian,
                n_convs=2,
                batch_norm=True,
                conv_dim=2,
            ),
            Pooling(method="max"),
        )

        self.embedding_block_2 = nn.Sequential(
            EdgeConv(
                in_channels=64 * 2,
                out_channels=64,
                k=k,
                method=Cosine,
                n_convs=2,
                batch_norm=True,
                conv_dim=2,
            ),
            Pooling(method="max"),
        )

        self.embedding_block_3 = nn.Sequential(
            EdgeConv(
                in_channels=64 * 2, 
                out_channels=64,
                k=k,
                method=Cosine,
                n_convs=2,
                batch_norm=True,
                conv_dim=2,
            ),
            Pooling(method="max"),
        )

        self.embedding_block_4 = nn.Sequential(
            Convolution(
                in_channels=64 * 3,
                out_channels=emb_dims,
                batch_norm=False,
                conv_dim=1,
            ),
            Pooling(method="max", keep_dim=True),
        )

        self.fc = nn.Sequential(
            nn.Conv1d(emb_dims + 192 + 6, 512, kernel_size=1, bias=False),  # 192/18
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, num_classes, kernel_size=1, bias=False),
        )

        self.concat = Concatenate()
        self.repeat = Repeat()
        self.residual_skip = SkipConnection(method="residual")
        self.adaptive_pooling = Pooling(method="max", adaptive=True, size=(64, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Input: (batch_size, 6, num_points)

        Block 1:
        Graph:
            (batch_size, 6, num_points) ->
            (batch_size, 6*2, num_points, k)
        Convolution:
            (batch_size, 6*2, num_points, k) ->
            (batch_size, 64, num_points, k)
        Convolution:
            (batch_size, 64, num_points, k) ->
            (batch_size, 64, num_points, k)
        Max:
            (batch_size, 64, num_points, k) ->
            (batch_size, 64, num_points)

        Block 2:
        Graph:
            (batch_size, 64, num_points) ->
            (batch_size, 64*2, num_points, k)
        Convolution:
            (batch_size, 64*2, num_points, k) ->
            (batch_size, 64, num_points, k)
        Convolution:
            (batch_size, 64, num_points, k) ->
            (batch_size, 64, num_points, k)
        Max:
            (batch_size, 64, num_points, k) ->
            (batch_size, 64, num_points)

        Block 3:
        Graph:
            (batch_size, 64, num_points) ->
            (batch_size, 64*2, num_points, k)
        Convolution:
            (batch_size, 64*2, num_points, k) ->
            (batch_size, 64, num_points, k)
        Max:
            (batch_size, 64, num_points, k) ->
            (batch_size, 64, num_points)

        Concatenate -> (batch_size, 64*3, num_points)

        Block 4:
        Convolution:
            (batch_size, 64*3, num_points) ->
            (batch_size, emb_dims, num_points)
        Max:
            (batch_size, emb_dims, num_points) ->
            (batch_size, emb_dims, 1)

        Repeat -> (batch_size, emb_dims, num_points)
        Concatenation -> (batch_size, emb_dims+64*3, num_points)

        MLP:
        Convolution:
            (batch_size, emb_dims+192+6, num_points) ->
            (batch_size, 512, num_points)
        Convolution:
            (batch_size, 512, num_points) ->
            (batch_size, 256, num_points)
        Dropout
        Convolution:
            (batch_size, 256, num_points) ->
            (batch_size, num_classes, num_points)
        """
        num_points = x.size(2)

        x1 = self.embedding_block_1(x)
        x2 = self.embedding_block_2(x1)

        x3 = self.embedding_block_3(x2)
        x4 = self.concat(x1, x2, x3)

        x4 = self.embedding_block_4(x4)
        x4 = self.repeat(x4, num_points)
        x4 = self.residual_skip(x4, x3)
        x5 = self.concat(x, x1, x2, x3, x4)

        return self.fc(x5)
