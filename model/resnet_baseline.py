import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SeparableConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,
                      in_ch,
                      kernel_size=kernel_size,
                      padding=padding,
                      groups=in_ch,
                      bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = SeparableConvBNReLU(in_ch + skip_ch,
                                        out_ch,
                                        kernel_size=3)

    def forward(self, x, skip):
        x = F.interpolate(x,
                          size=skip.shape[-2:],
                          mode="bilinear",
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ResNetBaseline(nn.Module):
    """
    A stronger CNN baseline for dense ocean field prediction.
    The model keeps a ResNet-style encoder, but uses U-Net-style skip fusion
    in the decoder so that it can better preserve fine spatial structures.

    Input:
        x: [B, C_in, H, W]
    Output:
        feature: bottleneck feature [B, C_feat, H/8, W/8]
        output: predicted field [B, C_out, H, W]
    """

    def __init__(self, params):
        super().__init__()

        img_size = params["img_size"]
        in_chans = params["net_image"]["N_in_channels"]
        out_chans = params["net_image"]["N_out_channels"]

        base_channels = params.get("resnet_base_channels", 32)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans,
                      c1,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1, c1, stride=1),
        )

        self.enc1 = nn.Sequential(
            ResidualBlock(c1, c1, stride=1),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2),
            ResidualBlock(c2, c2, stride=1),
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2),
            ResidualBlock(c3, c3, stride=1),
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock(c3, c4, stride=2),
            ResidualBlock(c4, c4, stride=1),
        )

        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)

        self.refine = nn.Sequential(
            SeparableConvBNReLU(c1 + c1, c1, kernel_size=3),
        )
        self.out_conv = nn.Conv2d(c1, out_chans, kernel_size=1, bias=False)

        self.feature_down_factor = 8
        self.img_size = img_size

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        feat = self.bottleneck(x3)

        d3 = self.dec3(feat, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        d0 = F.interpolate(d1,
                           size=x0.shape[-2:],
                           mode="bilinear",
                           align_corners=False)
        d0 = torch.cat([d0, x0], dim=1)
        d0 = self.refine(d0)

        out = self.out_conv(d0)
        return feat, out


if __name__ == "__main__":
    params = {
        "img_size": (720, 960),
        "net_image": {
            "N_in_channels": 32,
            "N_out_channels": 32,
        },
        "resnet_base_channels": 32,
    }

    model = ResNetBaseline(params=params)
    x = torch.randn(1, 32, 720, 960)
    feature, output = model(x)
    print("feature shape:", feature.shape)
    print("output shape:", output.shape)
