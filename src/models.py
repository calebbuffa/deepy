import torch.nn as nn
import torch
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

class ResNetEncoder(nn.Module):
    def __init__(self, model: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if model == "resnet18":
            resnet = torchvision.models.resnet.resnet18(pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif model == 'resnet34':
            resnet = torchvision.models.resnet.resnet34(pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif model == 'resnet50':
            resnet = torchvision.models.resnet.resnet50(pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnet101':
            resnet = torchvision.models.resnet.resnet101(pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnet152':
            resnet = torchvision.models.resnet.resnet152(pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnext50':
            resnet = torchvision.models.resnet.resnext50_32x4d(pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"{model} is not supported")

        if pretrained:
            resnet.eval()

        self.encoding_blocks = nn.ModuleList(
            [nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu
            ),

            nn.Sequential(
                resnet.maxpool,
                resnet.layer1
            ),

            resnet.layer2,
            resnet.layer3,
            resnet.layer4]
        )

    def forward(self, x: Tensor) -> OrderedDict[Tensor]:
        encoded_feature_maps = OrderedDict()
        for idx, encoding_block in enumerate(self.encoding_blocks):
            x = encoding_block(x)
            encoded_feature_maps[f"x{idx}"] = x


        return encoded_feature_maps


class UNetEncoder(nn.Module):
    def __init__(self, n_channels, bilinear: bool = False) -> None:
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x: Tensor) -> list[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x5, x4, x3, x2, x1]


class UNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        encoder: nn.Module = UNetEncoder,
        **kwargs,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        if encoder == ResNeXtEncoder:
            assert self.n_channels == 3

        if encoder == UNetEncoder:
            self.encoder = encoder(n_channels=n_channels, **kwargs)

        bilinear = kwargs.get("bilinear", False)
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x5, x4, x3, x2, x1 = self.encoder(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
