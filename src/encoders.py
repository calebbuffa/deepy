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
    def __init__(self, in_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        factor = 2 if bilinear else 1
        self.encoding_blocks = nn.ModuleList(
            [
                self.inc = DoubleConv(in_channels, 64),
                self.down1 = Down(64, 128),
                self.down2 = Down(128, 256),
                self.down3 = Down(256, 512),
                self.down4 = Down(512, 1024 // factor),
                self.out_channels = [64, 128, 256, 512, 1024]
             ]
        )

    def forward(self, x: Tensor) -> OrderedDict[Tensor]:
        encoded_feature_maps = OrderedDict()
        for idx, encoding_block in enumerate(self.encoding_blocks):
            x = encoding_block(x)
            encoded_feature_maps[f"x{idx}"] = x

        return encoded_feature_maps
