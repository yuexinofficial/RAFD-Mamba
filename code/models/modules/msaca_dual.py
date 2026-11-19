# -*- coding: utf-8 -*-
"""双入口 MSACA —— **本项目的改造版,不是论文原版**。

与 `msaca.py`(官方复现,未改动)的区别,只有一处:

    原版(单入口)                     本版(双入口)
    ─────────────────               ─────────────────────────────
    Q    ← Linear(Flatten(x))        Q    ← Linear(Flatten(skip_i))
    fagg ← 三分支(x)                  fagg ← 三分支(skip_{i+1})
    out  = out_proj(Attn)            out  = skip_i + out_proj(Attn)   ← 残差也加了
    (无残差)

即:用**深一层**的上下文去增强**浅一层**的特征。
三分支(AvgPool / ModulatedDeformConv / DilatedConv)、双 Top-k 稀疏注意力、
η/α/β 权重等全部沿用原版,未改。

🔴 分辨率与通道差异:
    skip_i    (B, Cq,  Hq, Wq)      Q 侧,  Nq = Hq*Wq 个 token
    skip_{i+1}(B, Ckv, Hk, Wk)      K/V 侧,Nkv = Hk*Wk 个 token(通常 Nkv = Nq/4)
    注意力矩阵是 (B, heads, Nq, Nkv) —— 合法,输出按 **Q 的尺寸** reshape 回来。
    通道靠 q_proj / kv_proj 各自映射到同一个 attention_dim 对齐。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .msaca import ModulatedDeformConv2d


Number = Union[int, float]


def _as_float_tuple(values: Sequence[Number], length: int, name: str) -> Tuple[float, ...]:
    seq = tuple(float(v) for v in values)
    if len(seq) != length:
        raise ValueError(f"{name} must have {length} values, got {len(seq)}")
    return seq


class DualInputMSACA(nn.Module):
    """Q 读当层 skip,三分支读下一层特征,输出与当层 skip 残差相加。"""

    def __init__(
        self,
        channels_q: int,
        channels_kv: int,
        *,
        attention_dim: Optional[int] = None,
        num_heads: int = 4,
        pool_kernel_size: int = 3,
        dilation: int = 2,
        deform_kernel_size: int = 3,
        eta: Sequence[Number] = (0.4, 0.3, 0.3),
        topk_fractions: Sequence[Number] = (0.5, 0.25),
        alpha_beta: Sequence[Number] = (0.5, 0.5),
        attention_chunk_size: Optional[int] = 1024,
        use_torchvision_deform: bool = True,
    ) -> None:
        super().__init__()
        self.channels_q = int(channels_q)
        self.channels_kv = int(channels_kv)
        self.attention_dim = int(channels_q if attention_dim is None else attention_dim)
        self.num_heads = int(num_heads)
        if self.attention_dim % self.num_heads != 0:
            raise ValueError("attention_dim 必须能被 num_heads 整除")
        self.head_dim = self.attention_dim // self.num_heads
        self.pool_kernel_size = int(pool_kernel_size)
        self.dilation = int(dilation)

        eta_t = _as_float_tuple(eta, 3, "eta")
        self.register_buffer("eta", torch.tensor(eta_t, dtype=torch.float32))
        fr = _as_float_tuple(topk_fractions, 2, "topk_fractions")
        self.topk_fractions = fr
        self.register_buffer("mix_weights",
                             torch.tensor(_as_float_tuple(alpha_beta, 2, "alpha_beta"),
                                          dtype=torch.float32))

        # ---- 三分支:作用于 skip_{i+1}(通道 = channels_kv)----
        self.pool = nn.AvgPool2d(kernel_size=self.pool_kernel_size, stride=1,
                                 padding=self.pool_kernel_size // 2,
                                 count_include_pad=False)
        self.deform = ModulatedDeformConv2d(channels_kv, channels_kv,
                                            kernel_size=deform_kernel_size,
                                            use_torchvision=use_torchvision_deform)
        self.dilated = nn.Conv2d(channels_kv, channels_kv, kernel_size=3, stride=1,
                                 padding=dilation, dilation=dilation)
        self.bn_pool = nn.BatchNorm2d(channels_kv)
        self.bn_deform = nn.BatchNorm2d(channels_kv)
        self.bn_dilated = nn.BatchNorm2d(channels_kv)

        # ---- Q 侧读 skip_i(通道 = channels_q)----
        self.q_proj = nn.Linear(channels_q, self.attention_dim)
        self.kv_proj = nn.Linear(channels_kv, 2 * self.attention_dim)
        self.kv_norm = nn.LayerNorm(2 * self.attention_dim)
        # 输出必须回到 channels_q(残差要用),所以 out_proj 恒存在
        self.out_proj = nn.Linear(self.attention_dim, channels_q)
        self.attention_chunk_size = attention_chunk_size

    @staticmethod
    def _flatten_map(x: Tensor) -> Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()

    def _resolve_topk(self, n: int) -> Tuple[int, int]:
        out = []
        for f in self.topk_fractions:
            out.append(max(1, min(n, int(math.ceil(f * n)))))
        return out[0], out[1]

    def _sparse_attention(self, q: Tensor, k: Tensor, v: Tensor, keep: int) -> Tensor:
        b, heads, nq, d = q.shape
        nkv = k.shape[2]
        kt = k.transpose(-2, -1)                      # (B,h,d,Nkv)
        chunk = self.attention_chunk_size or nq
        outs = []
        scale = 1.0 / math.sqrt(d)
        neg_inf = torch.finfo(q.dtype).min
        for start in range(0, nq, chunk):
            stop = min(start + chunk, nq)
            logits = torch.matmul(q[:, :, start:stop], kt) * scale   # (B,h,n,Nkv)
            if keep < nkv:
                idx = torch.topk(logits, k=keep, dim=-1, largest=True, sorted=False).indices
                sel = torch.zeros_like(logits, dtype=torch.bool)
                sel.scatter_(-1, idx, True)
                logits = logits.masked_fill(~sel, neg_inf)
            outs.append(torch.matmul(torch.softmax(logits, dim=-1), v))
        return torch.cat(outs, dim=2)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        if x_q.ndim != 4 or x_kv.ndim != 4:
            raise ValueError("两个输入都必须是 [B,C,H,W]")
        if x_q.shape[1] != self.channels_q:
            raise ValueError(f"Q 侧期望 {self.channels_q} 通道,得到 {x_q.shape[1]}")
        if x_kv.shape[1] != self.channels_kv:
            raise ValueError(f"K/V 侧期望 {self.channels_kv} 通道,得到 {x_kv.shape[1]}")

        b, _, hq, wq = x_q.shape
        eta = self.eta.to(device=x_q.device, dtype=x_q.dtype)

        # 三分支在 skip_{i+1} 上算
        f1 = F.relu(self.bn_pool(self.pool(x_kv)))
        f2 = F.relu(self.bn_deform(self.deform(x_kv)))
        f3 = F.relu(self.bn_dilated(self.dilated(x_kv)))
        fagg = eta[0] * f1 + eta[1] * f2 + eta[2] * f3

        q = self.q_proj(self._flatten_map(x_q))
        kv = self.kv_norm(self.kv_proj(self._flatten_map(fagg)))
        nq = q.shape[1]
        nkv = kv.shape[1]
        q = q.view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(b, nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, nkv, self.num_heads, self.head_dim).transpose(1, 2)

        keep1, keep2 = self._resolve_topk(nkv)
        o1 = self._sparse_attention(q, k, v, keep1)
        o2 = self._sparse_attention(q, k, v, keep2)
        mix = self.mix_weights.to(device=x_q.device, dtype=o1.dtype)
        o = mix[0] * o1 + mix[1] * o2
        o = o.transpose(1, 2).contiguous().view(b, nq, self.attention_dim)
        o = self.out_proj(o).transpose(1, 2).reshape(b, self.channels_q, hq, wq)

        return x_q + o          # ← 残差(原版模块内部没有,这一版在外面加)
