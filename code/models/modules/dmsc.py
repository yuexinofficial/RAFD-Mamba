from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
GATE_BIAS_INIT = -4.0

class DepthwiseModulatedDeformConv2d(nn.Module):

    def __init__(self, channels: int, kernel_size: int=3, *, dilation: int=1, use_torchvision: bool=True):
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive')
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd number')
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = self.dilation * (self.kernel_size // 2)
        self.stride = (1, 1)
        n = self.kernel_size ** 2
        self.offset_conv = nn.Conv2d(self.channels, 2 * n, 3, padding=1)
        self.mask_conv = nn.Conv2d(self.channels, n, 3, padding=1)
        self.weight = nn.Parameter(torch.empty(self.channels, 1, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.empty(self.channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.kernel_size ** 2)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)
        self._tv = None
        if use_torchvision:
            try:
                from torchvision.ops import deform_conv2d
                self._tv = deform_conv2d
            except Exception:
                self._tv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'expected [B,C,H,W], got {tuple(x.shape)}')
        if x.shape[1] != self.channels:
            raise ValueError(f'expected {self.channels} channels, got {x.shape[1]}')
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        if self._tv is not None:
            try:
                return self._tv(x, offset, self.weight, self.bias, stride=self.stride, padding=(self.padding, self.padding), dilation=(self.dilation, self.dilation), mask=mask)
            except (RuntimeError, NotImplementedError):
                pass
        return self._fallback(x, offset, mask)

    def _fallback(self, x, offset, mask):
        b, c, h, w = x.shape
        k = self.kernel_size
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
        out = None
        for ky in range(k):
            for kx in range(k):
                p = ky * k + kx
                dy, dx = (offset[:, 2 * p], offset[:, 2 * p + 1])
                gy = yy[None] - self.padding + ky * self.dilation + dy
                gx = xx[None] - self.padding + kx * self.dilation + dx
                grid = torch.stack(((gx + 0.5) * (2.0 / w) - 1.0, (gy + 0.5) * (2.0 / h) - 1.0), dim=-1)
                s = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                s = s * mask[:, p:p + 1]
                s = s * self.weight[:, 0, ky, kx].view(1, c, 1, 1)
                out = s if out is None else out + s
        return out + self.bias.view(1, c, 1, 1)

class GAPMLP(nn.Module):

    def __init__(self, dim: int, n_out: int, ratio: int=4):
        super().__init__()
        hidden = max(dim // ratio, 8)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.mean(dim=(2, 3))
        return self.fc2(self.act(self.fc1(g))).softmax(dim=-1)

class DynamicMultiScaleSparseContext(nn.Module):

    def __init__(self, dim: int, *, num_heads: int=4, attention_dim=None, dilation: int=2, deform_kernel_size: int=3, topk_fractions=(0.5, 0.25), gate_ratio: int=4, gate_bias_init: float=GATE_BIAS_INIT, use_lambda: bool=True, attention_chunk_size=1024, use_torchvision_deform: bool=True):
        super().__init__()
        self.use_lambda = bool(use_lambda)
        dim = int(dim)
        if dim <= 0:
            raise ValueError('dim must be positive')
        self.dim = dim
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError('num_heads must be positive')
        self.attention_dim = int(dim // 2 if attention_dim is None else attention_dim)
        if self.attention_dim <= 0 or self.attention_dim % self.num_heads != 0:
            raise ValueError('attention_dim 必须为正且能被 num_heads 整除')
        self.head_dim = self.attention_dim // self.num_heads
        frac = tuple((float(v) for v in topk_fractions))
        if len(frac) != 2 or any((not math.isfinite(v) or v <= 0 or v > 1 for v in frac)):
            raise ValueError('topk_fractions 必须是 (0,1] 内的两个数')
        self.topk_fractions = frac
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.dil = nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim, bias=True)
        self.deform = DepthwiseModulatedDeformConv2d(dim, deform_kernel_size, use_torchvision=use_torchvision_deform)
        self.bn_dw = nn.BatchNorm2d(dim)
        self.bn_dil = nn.BatchNorm2d(dim)
        self.bn_deform = nn.BatchNorm2d(dim)
        self.eta_gate = GAPMLP(dim, 3, gate_ratio)
        self.ab_gate = GAPMLP(dim, 2, gate_ratio)
        self.q_proj = nn.Linear(dim, self.attention_dim)
        self.kv_proj = nn.Linear(dim, 2 * self.attention_dim)
        self.kv_norm = nn.LayerNorm(2 * self.attention_dim)
        self.out_proj = nn.Linear(self.attention_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if self.use_lambda:
            self.lam_gate = nn.Linear(2 * dim, dim)
            self.gate_bias = nn.Parameter(torch.full((dim,), float(gate_bias_init)))
            nn.init.zeros_(self.lam_gate.bias)
        self.attention_chunk_size = attention_chunk_size
        self.reset_stats()

    @staticmethod
    def _flatten(x):
        return x.flatten(2).transpose(1, 2).contiguous()

    def _resolve_topk(self, n):
        out = []
        for f in self.topk_fractions:
            out.append(max(1, min(n, int(math.ceil(f * n)))))
        return (out[0], out[1])

    def _sparse_attention(self, q, k, v, keep):
        b, heads, n, d = q.shape
        kt = k.transpose(-2, -1)
        chunk = self.attention_chunk_size or n
        scale = 1.0 / math.sqrt(d)
        neg_inf = torch.finfo(q.dtype).min
        outs = []
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            logits = torch.matmul(q[:, :, s:e], kt) * scale
            if keep < n:
                idx = torch.topk(logits, k=keep, dim=-1, largest=True, sorted=False).indices
                sel = torch.zeros_like(logits, dtype=torch.bool)
                sel.scatter_(-1, idx, True)
                logits = logits.masked_fill(~sel, neg_inf)
            outs.append(torch.matmul(torch.softmax(logits, dim=-1), v))
        return torch.cat(outs, dim=2)

    def reset_stats(self):
        self._n = 0.0
        self._lam = 0.0
        self._lam_sq = 0.0
        self._f = 0.0
        self._i = 0.0
        self._eta = torch.zeros(3)
        self._ab = torch.zeros(2)

    @torch.no_grad()
    def _accumulate(self, x, inject, lam, eta, ab):
        b = x.shape[0]
        self._n += b
        lv = 1.0 if lam is None else float(lam.mean())
        self._lam += lv * b
        self._lam_sq += lv ** 2 * b
        self._eta += eta.detach().mean(dim=0).cpu() * b
        self._ab += ab.detach().mean(dim=0).cpu() * b
        self._f += float(x.detach().reshape(b, -1).norm(dim=1).sum())
        self._i += float(inject.detach().reshape(b, -1).norm(dim=1).sum())

    @torch.no_grad()
    def stats(self):
        nan = float('nan')
        if self._n == 0:
            return dict(lambda_mean=nan, lambda_std=nan, eta1=nan, eta2=nan, eta3=nan, alpha=nan, beta=nan, E=nan)
        n = self._n
        lm = self._lam / n
        var = max(self._lam_sq / n - lm ** 2, 0.0)
        f, i = (self._f, self._i)
        return dict(lambda_mean=lm, lambda_std=var ** 0.5, eta1=float(self._eta[0] / n), eta2=float(self._eta[1] / n), eta3=float(self._eta[2] / n), alpha=float(self._ab[0] / n), beta=float(self._ab[1] / n), E=i / f if f > 0 else nan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'expected BHWC, got {tuple(x.shape)}')
        if x.shape[-1] != self.dim:
            raise ValueError(f'expected {self.dim} channels, got {x.shape[-1]}')
        xb = x.permute(0, 3, 1, 2).contiguous()
        b, c, h, w = xb.shape
        eta = self.eta_gate(xb)
        f1 = F.relu(self.bn_dw(self.dw(xb)))
        f2 = F.relu(self.bn_dil(self.dil(xb)))
        f3 = F.relu(self.bn_deform(self.deform(xb)))
        fm = eta[:, 0].view(b, 1, 1, 1) * f1 + eta[:, 1].view(b, 1, 1, 1) * f2 + eta[:, 2].view(b, 1, 1, 1) * f3
        n = h * w
        q = self.q_proj(self._flatten(xb))
        k, v = self.kv_norm(self.kv_proj(self._flatten(fm))).chunk(2, dim=-1)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        keep1, keep2 = self._resolve_topk(n)
        o1 = self._sparse_attention(q, k, v, keep1)
        o2 = self._sparse_attention(q, k, v, keep2)
        ab = self.ab_gate(xb)
        o = ab[:, 0].view(b, 1, 1, 1) * o1 + ab[:, 1].view(b, 1, 1, 1) * o2
        o = o.transpose(1, 2).contiguous().view(b, n, self.attention_dim)
        o = self.out_proj(o).transpose(1, 2).reshape(b, c, h, w)
        o_hwc = o.permute(0, 2, 3, 1)
        if self.use_lambda:
            lam = torch.sigmoid(self.lam_gate(torch.cat([x, o_hwc], dim=-1)) + self.gate_bias)
            inject = lam * o_hwc
        else:
            lam = None
            inject = o_hwc
        if not self.training:
            self._accumulate(x, inject, lam, eta, ab)
        return x + inject
DMSC = DynamicMultiScaleSparseContext
__all__ = ['DynamicMultiScaleSparseContext', 'DMSC', 'DepthwiseModulatedDeformConv2d', 'GAPMLP', 'GATE_BIAS_INIT']
