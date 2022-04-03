import torch.nn as nn
import torch

class ResNeXtEncoder(nn.Module):
    def __init__(self, size: int = 50, pretrained: bool = True):
        super().__init__()
        version = "resnext50_32x4d" if size == 50 else "resnext101_32x8d"
        model = torch.hub.load("pytorch/vision:v0.10.0", version, pretrained=pretrained)

        if pretrained:
            for param in model.parameters():
                param.requires_grad = False

        inconv = nn.ModuleList()
        for name, module in model.named_children():
            if name in {"conv1", "bn1", "relu", "maxpool"}:
                inconv.append(module)
            elif name == "layer1":
                self.down1 = module
            elif name == "layer2":
                self.down2 = module
            elif name == "layer3":
                self.down3 = module
            elif name == "layer4":
                self.down4 = module

        self.inc = nn.Sequential(*inconv)

    def forward(self, x: Tensor) -> list[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x1 = self.invc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return [x5, x4, x3, x2, x1]


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
