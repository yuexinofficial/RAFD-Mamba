from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class SplitLocalGlobalEnhance(nn.Module):

    def __init__(self, dim: int, *, dilations=(1, 2, 3), global_scales=(1, 2, 3), bottle_ratio: int=8):
        super().__init__()
        dim = int(dim)
        if dim % 2 != 0:
            raise ValueError(f'dim 必须是偶数才能劈半,得到 {dim}')
        self.dim = dim
        h = dim // 2
        self.half = h
        self.dilations = tuple((int(d) for d in dilations))
        self.local_convs = nn.ModuleList([nn.Conv2d(h, h, 3, padding=d, dilation=d, groups=h) for d in self.dilations])
        self.local_fuse = nn.Conv2d(h * len(self.dilations), h, 1)
        self.bn_local = nn.BatchNorm2d(h)
        self.global_scales = tuple((int(s) for s in global_scales))
        hid = max(h // int(bottle_ratio), 8)
        self.g_proj = nn.Conv2d(h, hid, 1)
        self.g_out = nn.Conv2d(hid, h, 1)
        self.bn_global = nn.BatchNorm2d(h)
        self.act = nn.GELU()
        self.out = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'expected BHWC, got {tuple(x.shape)}')
        if x.shape[-1] != self.dim:
            raise ValueError(f'expected {self.dim} channels, got {x.shape[-1]}')
        h_img, w_img = (x.shape[1], x.shape[2])
        xb = x.permute(0, 3, 1, 2).contiguous()
        x1, x2 = xb.chunk(2, dim=1)
        y1 = self.local_fuse(torch.cat([c(x1) for c in self.local_convs], dim=1))
        y1 = self.act(self.bn_local(y1))
        gs = [F.adaptive_avg_pool2d(x2, s) for s in self.global_scales]
        gs = [self.g_proj(g) for g in gs]
        g = gs[0]
        for gi in gs[1:]:
            g = g + F.interpolate(gi, size=gs[0].shape[-2:], mode='bilinear', align_corners=False)
        g = self.g_out(g)
        y2 = self.act(self.bn_global(F.interpolate(g, size=(h_img, w_img), mode='bilinear', align_corners=False)))
        y = torch.cat([y1, y2], dim=1)
        return x + self.out(y).permute(0, 2, 3, 1)
LGE = SplitLocalGlobalEnhance
__all__ = ['SplitLocalGlobalEnhance', 'LGE']
