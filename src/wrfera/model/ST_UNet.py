from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 基础块 (源于你的实现) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_ch, eps=1e-5)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_ch, eps=1e-5)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear \
                  else nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return self.conv(torch.cat([x2, x1], dim=1))

# --- 注意力与MLP (保留核心思想) ---
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, attn_drop: float = 0.1) -> None:
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        Q, K, V = self.q(t), self.k(t), self.v(t)
        attn = torch.matmul(Q.transpose(-1, -2), K) / (C ** 0.5)
        attn = self.softmax(attn - attn.max(dim=-1, keepdim=True).values)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, V.transpose(-1, -2))
        ctx = ctx.permute(0, 2, 1).reshape(B, C, H, W)
        return ctx + x

class SpatialAttention(nn.Module):
    def __init__(self, channels: int, win: int = 8, heads: int = 4, attn_drop: float = 0.1) -> None:
        super().__init__()
        self.win = win
        self.h = heads
        self.dim = channels
        assert channels % heads == 0
        self.head_dim = channels // heads
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(-1)

    def _split_windows(self, x: torch.Tensor):
        B, C, H, W = x.shape
        pr, pb = (self.win - W % self.win) % self.win, (self.win - H % self.win) % self.win
        x = F.pad(x, (0, pr, 0, pb))
        B, C, Hp, Wp = x.shape
        x = x.view(B, C, Hp // self.win, self.win, Wp // self.win, self.win)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.win * self.win, C)
        return x, Hp, Wp, pb, pr

    def _merge(self, out, B, Hp, Wp, pb, pr):
        out = out.view(B, Hp // self.win, Wp // self.win, self.win, self.win, self.dim)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, self.dim, Hp, Wp)
        return out[:, :, : Hp - pb, : Wp - pr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xw, Hp, Wp, pb, pr = self._split_windows(x)
        q, k, v = self.q(xw), self.k(xw), self.v(xw)
        def transpose_heads(t):
            Bn, N, C = t.shape
            t = t.view(Bn, N, self.h, self.head_dim).permute(0, 2, 1, 3)
            return t
        qh, kh, vh = transpose_heads(q), transpose_heads(k), transpose_heads(v)
        attn = torch.matmul(qh, kh.transpose(-1, -2))
        attn = self.softmax(attn - attn.max(dim=-1, keepdim=True).values)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, vh).permute(0, 2, 1, 3).contiguous().view(xw.size(0), xw.size(1), C)
        out = self.proj(ctx)
        out = self._merge(out, B, Hp, Wp, pb, pr)
        return out

class MLP(nn.Module):
    def __init__(self, in_ch: int, hid: int, drop: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hid)
        self.fc2 = nn.Linear(hid, in_ch)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.view(B, C, -1).permute(0, 2, 1)
        res = self.drop(t)
        t = self.drop(self.act(self.fc1(t)))
        t = self.drop(self.fc2(t)) + res
        return t.permute(0, 2, 1).view(B, C, H, W)

# --- UNet 主体 ---
class STUNet(nn.Module):
    """
    输入: history (B, T=24, C=11, H, W), geo (B?, H, W)
    输出: (T, 1, H, W) —— 针对第1通道的校正(与你的训练/评估脚本一致)
    """
    def __init__(self, in_ch: int = 11, out_ch: int = 1,
                 base_ch: int = 32, win: int = 8, heads: int = 4) -> None:
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        downs = [32, 64, 128, 256, 256]
        self.down = nn.ModuleList([Down(downs[i-1], downs[i]) for i in range(1, len(downs))])
        self.up = nn.ModuleList([
            Up(514, 129), Up(258, 65), Up(130, 32), Up(64, 32)
        ])
        self.chan_attn = nn.ModuleList([
            ChannelAttention(c) for c in [64, 128, 256, 256]
        ])
        self.spa_attn = nn.ModuleList([
            SpatialAttention(s, win=win, heads=heads) for s in [65, 129, 257, 257]
        ])
        self.mlps = nn.ModuleList([
            MLP(65, 64), MLP(129, 128), MLP(257, 256), MLP(257, 256)
        ])
        self.outc = nn.Sequential(nn.Conv2d(32, out_ch, 1), nn.InstanceNorm2d(out_ch, eps=1e-5))

    def forward(self, history: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        # history: (B?, T, C, H, W) 来自数据集 batch_size=1 的 (T,C,H,W)，这里统一为 (B=1, T,...)
        if history.dim() == 4:
            history = history.unsqueeze(0)
        x = torch.nan_to_num(history, nan=0.0).squeeze(0).permute(1, 0, 2, 3)  # (C=11, T=24, H, W) -> (11,24,H,W)
        x1 = self.inc(x)

        downs = [x1]
        for d in self.down:
            downs.append(d(downs[-1]))

        geo_exp = geo.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        for i in range(1, len(downs)):
            gi = F.interpolate(geo_exp, size=downs[i].shape[-2:], mode="bilinear", align_corners=False)
            gi = gi.expand(downs[i].size(0), -1, -1, -1).contiguous()
            downs[i] = self.chan_attn[i-1](downs[i])
            downs[i] = torch.cat([downs[i], gi], dim=1)
            downs[i] = self.spa_attn[i-1](downs[i])
            downs[i] = self.mlps[i-1](downs[i])

        x = downs[-1]
        for i, up in enumerate(self.up):
            x = up(x, downs[-(i+2)])

        out = self.outc(x).permute(1, 0, 2, 3)  # (T, 1, H, W)
        return out
