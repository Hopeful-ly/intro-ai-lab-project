import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class DoubleConv(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, residual: bool = False, dropout: float = 0.0
    ):
        super().__init__()
        self.residual = residual
        if residual:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.proj = (
                nn.Conv2d(in_ch, out_ch, 1, bias=False)
                if in_ch != out_ch
                else nn.Identity()
            )
            self.act = nn.ReLU(inplace=True)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        if self.residual:
            out = self.act(self.conv2(self.conv1(x)) + self.proj(x))
        else:
            out = self.block(x)
        return self.drop(out)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, residual: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, residual))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, residual: bool = False, dropout: float = 0.0
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch, residual, dropout)  # *2 for the skip

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:  # robust to non-power-of-two input sizes
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DilatedBottleneck(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0, dilations=(1, 2, 4)):
        super().__init__()
        layers = []
        for d in dilations:
            layers += [
                nn.Conv2d(ch, ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        out_ch: int,
        base_channels: int = 64,
        depth: int = 2,
        in_ch: int = 1,
        mode: str = "lab_regression",
        residual: bool = False,
        dropout: float = 0.0,
        dilated_bottleneck: bool = False,
    ):
        super().__init__()
        self.mode = mode

        self.inc = DoubleConv(in_ch, base_channels, residual)  # encoder: no dropout
        self.downs = nn.ModuleList()
        c = base_channels
        for _ in range(depth):
            self.downs.append(Down(c, c * 2, residual))
            c *= 2
        # optional dilated bottleneck (identity when off -> no extra params/keys)
        self.bottleneck = (
            DilatedBottleneck(c, dropout) if dilated_bottleneck else nn.Identity()
        )
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(c, c // 2, residual, dropout))  # decoder: dropout here
            c //= 2
        self.outc = nn.Conv2d(base_channels, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i + 2)])
        out = self.outc(x)

        if self.mode == "lab_regression":
            out = torch.tanh(out)  # ab target is normalised to ~[-1, 1]
        elif self.mode == "rgb_regression":
            out = torch.sigmoid(out)  # RGB in [0, 1]
        return out


def build_model(mcfg: ModelConfig) -> UNet:
    if mcfg.mode == "lab_regression":
        out_ch = 2
    elif mcfg.mode == "rgb_regression":
        out_ch = 3
    elif mcfg.mode == "lab_classification":
        out_ch = mcfg.grid_size**2
    else:
        raise ValueError(f"unknown mode: {mcfg.mode}")
    return UNet(
        out_ch,
        base_channels=mcfg.base_channels,
        depth=mcfg.depth,
        mode=mcfg.mode,
        residual=mcfg.residual,
        dropout=mcfg.dropout,
        dilated_bottleneck=mcfg.dilated_bottleneck,
    )
