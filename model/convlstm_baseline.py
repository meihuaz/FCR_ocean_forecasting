import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    单步 ConvLSTM Cell，支持输入形状 [B, C, H, W]。
    这里我们只在瓶颈层上用一层 ConvLSTM 做时序建模（即使当前是单时刻，也保持接口统一）。
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # i, f, o, g 四个门一次性卷积
        self.conv = nn.Conv2d(input_dim + hidden_dim,
                              4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_prev=None, c_prev=None):
        # x: [B, C_in, H, W]
        B, _, H, W = x.shape
        if h_prev is None:
            h_prev = torch.zeros(B,
                                 self.hidden_dim,
                                 H,
                                 W,
                                 device=x.device,
                                 dtype=x.dtype)
        if c_prev is None:
            c_prev = torch.zeros_like(h_prev)

        combined = torch.cat([x, h_prev], dim=1)  # [B, C_in + C_h, H, W]
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class BasicConvBlock(nn.Module):
    """
    简单的 Conv-BN-ReLU 块。
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvLSTMBaseline(nn.Module):
    """
    ConvLSTM 编码-解码 baseline，用于与 Swin / ResNet 对比。
    输入:  x [B, C_in, H, W]
    输出:  feature, output
        - feature: 编码器瓶颈+ConvLSTM 后特征 [B, C_feat, H/8, W/8]
        - output : 预测场 [B, C_out, H, W]
    """

    def __init__(self, params):
        super().__init__()

        img_size = params['img_size']
        in_ch = params['net_image']['N_in_channels']
        out_ch = params['net_image']['N_out_channels']

        # 控制整体参数量的基础通道数（与 ResNetBaseline 类似）
        base_channels = params.get('convlstm_base_channels', 64)
        c1 = base_channels  # 64
        c2 = base_channels * 2  # 128
        c3 = base_channels * 4  # 256
        c4 = base_channels * 4  # 256

        # --- Encoder ---
        # stem: 不改变分辨率
        self.stem = BasicConvBlock(in_ch, c1, kernel_size=7, stride=1)

        # stage1: H, W
        self.enc1 = nn.Sequential(
            BasicConvBlock(c1, c1),
            BasicConvBlock(c1, c1),
        )
        # stage2: H/2, W/2
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            BasicConvBlock(c2, c2),
            BasicConvBlock(c2, c2),
        )
        # stage3: H/4, W/4
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Sequential(
            BasicConvBlock(c3, c3),
            BasicConvBlock(c3, c3),
        )
        # stage4: H/8, W/8  (bottleneck before ConvLSTM)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        self.enc4 = nn.Sequential(
            BasicConvBlock(c4, c4),
            BasicConvBlock(c4, c4),
        )

        # --- ConvLSTM at bottleneck ---
        self.convlstm = ConvLSTMCell(input_dim=c4,
                                     hidden_dim=c4,
                                     kernel_size=3)

        # --- Decoder ---
        # up1: H/8 -> H/4
        self.up1_conv = BasicConvBlock(c4, c3)
        # up2: H/4 -> H/2
        self.up2_conv = BasicConvBlock(c3, c2)
        # up3: H/2 -> H
        self.up3_conv = BasicConvBlock(c2, c1)

        # 输出头：卷积到 out_ch
        self.out_conv = nn.Conv2d(c1, out_ch, kernel_size=1, bias=False)

        self.feature_down_factor = 8
        self.img_size = img_size

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, )):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: [B, C_in, H, W]
        B, C, H, W = x.shape

        # --- Encoder ---
        x = self.stem(x)  # [B, c1, H,   W]
        x1 = self.enc1(x)  # [B, c1, H,   W]

        x2_in = self.down1(x1)
        x2 = self.enc2(x2_in)  # [B, c2, H/2, W/2]

        x3_in = self.down2(x2)
        x3 = self.enc3(x3_in)  # [B, c3, H/4, W/4]

        x4_in = self.down3(x3)
        x4 = self.enc4(x4_in)  # [B, c4, H/8, W/8]

        # --- ConvLSTM (单步) ---
        h, c = self.convlstm(x4)  # [B, c4, H/8, W/8]
        feat = h  # 作为 feature 输出

        # --- Decoder ---
        d = F.interpolate(h,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        d = self.up1_conv(d)  # [B, c3, H/4, W/4]

        d = F.interpolate(d,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        d = self.up2_conv(d)  # [B, c2, H/2, W/2]

        d = F.interpolate(d,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        d = self.up3_conv(d)  # [B, c1, H, W]

        out = self.out_conv(d)  # [B, C_out, H, W]

        return feat, out


if __name__ == "__main__":
    # 简单自测，确保尺寸与配置兼容
    params = {
        'img_size': (720, 960),
        'net_image': {
            'N_in_channels': 32,
            'N_out_channels': 32,
            'embed_dim': 192,
            'depths': [1, 2, 2, 2, 1],
            'num_heads': [3, 6, 6, 6, 3],
            'window_size': (6, 12),
            'mlp_ratio': 4.0,
            'patch_size': 2,
            'earth_specific_pos': False,
        }
    }

    model = ConvLSTMBaseline(params=params)

    x = torch.randn(1, 32, 720, 960)

    feature, output = model(x)

    print('feature shape:', feature.shape)
    print('output shape:', output.shape)
