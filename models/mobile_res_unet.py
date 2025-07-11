import torch
import torch.nn as nn
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=4):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.silu = nn.SiLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU()
        )
        self.residual_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        time_emb = self.silu(self.time_mlp(t))
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return h + self.residual_connection(x)


from constants import TIME_EMB_DIM, NUM_GROUPS

class MobileResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=TIME_EMB_DIM, channel_multiplier=1.0, num_groups=NUM_GROUPS):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
        ch1, ch2, ch3, ch4 = int(32*channel_multiplier), int(64*channel_multiplier), int(128*channel_multiplier), int(256*channel_multiplier)
        self.down1, self.pool1 = ResidualBlock(in_channels, ch1, time_emb_dim, num_groups), nn.MaxPool2d(2)
        self.down2, self.pool2 = ResidualBlock(ch1, ch2, time_emb_dim, num_groups), nn.MaxPool2d(2)
        self.down3, self.pool3 = ResidualBlock(ch2, ch3, time_emb_dim, num_groups), nn.MaxPool2d(2)
        self.bot1 = ResidualBlock(ch3, ch4, time_emb_dim, num_groups)
        self.upconv1, self.up1 = nn.ConvTranspose2d(ch4, ch3, 2, 2), ResidualBlock(ch3*2, ch3, time_emb_dim, num_groups)
        self.upconv2, self.up2 = nn.ConvTranspose2d(ch3, ch2, 2, 2), ResidualBlock(ch2*2, ch2, time_emb_dim, num_groups)
        self.upconv3, self.up3 = nn.ConvTranspose2d(ch2, ch1, 2, 2), ResidualBlock(ch1*2, ch1, time_emb_dim, num_groups)
        self.out = nn.Conv2d(ch1, out_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.down1(x, t)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t)
        p2 = self.pool2(x2)
        x3 = self.down3(p2, t)
        p3 = self.pool3(x3)
        b = self.bot1(p3, t)
        up1 = self.up1(torch.cat([x3, self.upconv1(b)], 1), t)
        up2 = self.up2(torch.cat([x2, self.upconv2(up1)], 1), t)
        up3 = self.up3(torch.cat([x1, self.upconv3(up2)], 1), t)
        return self.out(up3)