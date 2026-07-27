import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels,
                                out_channels,
                                modes1,
                                modes2,
                                dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels,
                                out_channels,
                                modes1,
                                modes2,
                                dtype=torch.cfloat))

    def compl_mul2d(self, input_ft, weights):
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(batch_size,
                             self.out_channels,
                             x.size(-2),
                             x.size(-1) // 2 + 1,
                             dtype=torch.cfloat,
                             device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=x.shape[-2:], norm="ortho")
        return x


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        x = self.norm(x)
        x = F.gelu(x)
        return x


class FNOBaseline(nn.Module):
    """
    Minimal 2D FNO baseline adapted to the current dense ocean field setup.
    Input:
        x: [B, C_in, H, W]
    Output:
        feature: hidden representation [B, width, H, W]
        output: predicted field [B, C_out, H, W]
    """

    def __init__(self, params):
        super().__init__()

        in_chans = params["net_image"]["N_in_channels"]
        out_chans = params["net_image"]["N_out_channels"]
        self.width = params.get("fno_width", 48)
        self.modes1 = params.get("fno_modes1", 20)
        self.modes2 = params.get("fno_modes2", 20)
        self.num_layers = params.get("fno_num_layers", 4)

        self.input_proj = nn.Conv2d(in_chans + 2, self.width, kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlock(self.width, self.modes1, self.modes2)
            for _ in range(self.num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.width, out_chans, kernel_size=1),
        )

    def get_grid(self, shape, device):
        batch_size, _, size_x, size_y = shape
        gridx = torch.linspace(0, 1, size_x, device=device)
        gridy = torch.linspace(0, 1, size_y, device=device)
        gridx = gridx.view(1, 1, size_x, 1).repeat(batch_size, 1, 1, size_y)
        gridy = gridy.view(1, 1, 1, size_y).repeat(batch_size, 1, size_x, 1)
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat([x, grid], dim=1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        feature = x
        output = self.output_proj(x)
        return feature, output


if __name__ == "__main__":
    params = {
        "net_image": {
            "N_in_channels": 32,
            "N_out_channels": 32,
        },
        "fno_width": 48,
        "fno_modes1": 20,
        "fno_modes2": 20,
        "fno_num_layers": 4,
    }
    model = FNOBaseline(params)
    x = torch.randn(2, 32, 720, 960)
    feature, output = model(x)
    print("feature shape:", feature.shape)
    print("output shape:", output.shape)
