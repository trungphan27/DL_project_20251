import torch
import torch.nn as nn

class PartialConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias
        )

        self.mask_conv = nn.Conv2d(
            1,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

    def forward(self, x, mask):
        """
        x    : (B, C, H, W)
        mask : (B, 1, H, W)
        """

        with torch.no_grad():
            mask_out = self.mask_conv(mask)

        out = self.input_conv(x * mask)

        eps = 1e-8
        out = out * (mask_out > 0).float() / (mask_out + eps)

        new_mask = (mask_out > 0).float()
        new_mask = torch.max(new_mask, dim=1, keepdim=True)[0]

        return out, new_mask

class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample

        self.pconv = PartialConv2d(in_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x, mask):

        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        x = self.relu(x)

        if self.downsample:
            x = self.pool(x)
            mask = self.pool(mask)

        return x, mask

class PConvUpBlock(nn.Module):
    def __init__(self, in_ch_from_decoder, in_ch_from_encoder_skip, out_ch):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.pconv = PartialConv2d(
            in_ch_from_decoder + in_ch_from_encoder_skip,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask, skip_x, skip_mask):
        x = self.up(x)
        mask = self.up(mask)

        if skip_x is not None:
            x = torch.cat([x, skip_x], dim=1)
            mask = torch.max(mask, skip_mask)

        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        x = self.relu(x)

        return x, mask

class PConvUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = PConvBlock(3, 64)      # 256 → 128
        self.e2 = PConvBlock(64, 128)    # 128 → 64
        self.e3 = PConvBlock(128, 256)   # 64 → 32
        self.e4 = PConvBlock(256, 512)   # 32 → 16

        self.mid = PConvBlock(512, 512, downsample=False)

        self.d4 = PConvUpBlock(512, 256, 256)  # 16 → 32
        self.d3 = PConvUpBlock(256, 128, 128)  # 32 → 64
        self.d2 = PConvUpBlock(128, 64, 64)    # 64 → 128

        self.d1 = PConvUpBlock(64, 0, 64)      # 128 → 256

        self.final = PartialConv2d(
            64,
            3,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x, mask):

        e1, m1 = self.e1(x, mask)
        e2, m2 = self.e2(e1, m1)
        e3, m3 = self.e3(e2, m2)
        e4, m4 = self.e4(e3, m3)

        mid, mm = self.mid(e4, m4)

        d4, md4 = self.d4(mid, mm, e3, m3)
        d3, md3 = self.d3(d4, md4, e2, m2)
        d2, md2 = self.d2(d3, md3, e1, m1)

        d1, md1 = self.d1(d2, md2, None, None)

        out, _ = self.final(d1, md1)
        return torch.tanh(out)

def snconv(in_c, out_c, k, s, p):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_c, out_c, k, s, p)
    )

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            snconv(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            snconv(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            snconv(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            snconv(256, 512, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            snconv(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

def feature_matching_loss(D, real, fake):
    loss = 0
    x_real, x_fake = real, fake

    for layer in D.net[:-1]:
        x_real = layer(x_real)
        x_fake = layer(x_fake)
        loss += torch.mean(torch.abs(x_real - x_fake))

    return loss

