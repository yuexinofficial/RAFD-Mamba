"""PyTorch reproduction of the MSACA block described in CFG-MambaNet.

The paper calls the block *multi-scale adaptive context aggregation* (MSACA).
This implementation uses the name ``TriScaleDualSparseContextAggregator``
(TDSCA) to make clear which parts of the design are implemented: three
parallel context branches followed by two row-wise Top-k attention paths.

The paper specifies the equations, but leaves several engineering details
unspecified (pooling kernel, dilation, head count, Top-k proportions and the
alpha/beta values).  They are therefore constructor arguments rather than
hidden constants.  Defaults are documented in ``README.md``.

Input and output use the usual PyTorch ``[B, C, H, W]`` layout, while the
paper writes feature maps as ``[H, W, C]``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Number = Union[int, float]


def _pair(value: Number | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"expected a pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _odd_kernel(value: Number | Tuple[int, int], name: str) -> Tuple[int, int]:
    pair = _pair(value)
    if pair[0] <= 0 or pair[1] <= 0 or pair[0] % 2 == 0 or pair[1] % 2 == 0:
        raise ValueError(f"{name} must contain positive odd values, got {pair}")
    return pair


def _as_float_tuple(values: Sequence[Number], length: int, name: str) -> Tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return tuple(float(v) for v in values)


class ModulatedDeformConv2d(nn.Module):
    """A small DCNv2-style layer used by the second MSACA branch.

    ``torchvision.ops.deform_conv2d`` is used when available.  A differentiable
    ``grid_sample`` implementation is kept as a fallback so that the module
    remains usable in a minimal PyTorch installation.  The fallback supports
    the stride-one, same-padding configuration used by TDSCA.

    The offset and modulation-mask predictors are deliberately part of this
    layer: this follows Eq. (16)--(17), where ``Delta`` and ``M`` are learned
    from ``F_in`` rather than supplied by the caller.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Number | Tuple[int, int] = 3,
        *,
        padding: Optional[Number | Tuple[int, int]] = None,
        dilation: Number | Tuple[int, int] = 1,
        bias: bool = True,
        use_torchvision: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _odd_kernel(kernel_size, "kernel_size")
        self.dilation = _pair(dilation)
        if any(d <= 0 for d in self.dilation):
            raise ValueError("dilation must be positive")
        if padding is None:
            padding = tuple(d * (k // 2) for d, k in zip(self.dilation, self.kernel_size))
        self.padding = _pair(padding)
        self.stride = (1, 1)

        kh, kw = self.kernel_size
        n_points = kh * kw
        # A 3x3 predictor is sufficient and keeps the offset/mask heads light.
        self.offset_conv = nn.Conv2d(in_channels, 2 * n_points, 3, padding=1)
        self.mask_conv = nn.Conv2d(in_channels, n_points, 3, padding=1)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # Start from regular (zero-offset) sampling.  A zero mask logit is the
        # standard DCNv2 neutral initialization and gives sigmoid(M)=0.5.
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

        self._tv_deform_conv = None
        if use_torchvision:
            try:
                from torchvision.ops import deform_conv2d  # type: ignore

                self._tv_deform_conv = deform_conv2d
            except Exception:
                # Importing torchvision can fail when its optional C++ ops are
                # not built.  The pure-PyTorch fallback below still works.
                self._tv_deform_conv = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        if self._tv_deform_conv is not None:
            try:
                return self._tv_deform_conv(
                    x,
                    offset,
                    self.weight,
                    self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    mask=mask,
                )
            except (RuntimeError, NotImplementedError):
                # Some builds expose the Python symbol but omit the compiled
                # operator for a particular device/dtype.  Fall back safely.
                pass
        return self._fallback(x, offset, mask)

    def _fallback(self, x: Tensor, offset: Tensor, mask: Tensor) -> Tensor:
        """Differentiable stride-one modulated deformable convolution."""
        if self.stride != (1, 1):  # defensive; TDSCA never changes this
            raise RuntimeError("the fallback only supports stride=1")
        b, _, h, w = x.shape
        kh, kw = self.kernel_size
        # Coordinates are expressed in input-pixel units.  grid_sample with
        # align_corners=False maps a pixel centre p to 2*(p+.5)/size - 1.
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        samples = []
        point = 0
        for ky in range(kh):
            for kx in range(kw):
                dy = offset[:, 2 * point]
                dx = offset[:, 2 * point + 1]
                gy = yy[None] - self.padding[0] + ky * self.dilation[0] + dy
                gx = xx[None] - self.padding[1] + kx * self.dilation[1] + dx
                grid = torch.stack(
                    (
                        (gx + 0.5) * (2.0 / max(w, 1)) - 1.0,
                        (gy + 0.5) * (2.0 / max(h, 1)) - 1.0,
                    ),
                    dim=-1,
                )
                sampled = F.grid_sample(
                    x,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                samples.append(sampled * mask[:, point : point + 1])
                point += 1

        # [B, C_in, K, H, W] times [C_out, C_in, K].
        sampled = torch.stack(samples, dim=2)
        weight = self.weight.reshape(self.out_channels, self.in_channels, -1)
        out = torch.einsum("bckhw,ock->bohw", sampled, weight)
        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out


class TriScaleDualSparseContextAggregator(nn.Module):
    """TDSCA, a faithful and configurable implementation of paper MSACA.

    The computation follows the paper's Eq. (15)--(25):

    * ``F1``: average pooling -> BN -> ReLU;
    * ``F2``: modulated deformable convolution -> BN -> ReLU;
    * ``F3``: dilated 3x3 convolution -> BN -> ReLU;
    * ``Fagg = eta1*F1 + eta2*F2 + eta3*F3``;
    * ``Q`` comes from ``Flatten(Fin)`` and ``K,V`` from ``Flatten(Fagg)``;
    * two row-wise Top-k masks are applied to ``QK^T/sqrt(d_k)``;
    * ``O = alpha*(A1 V) + beta*(A2 V)`` and heads are merged back to a map.

    Parameters not specified in the paper are explicit here.  In particular,
    ``topk_fractions=(0.5, 0.25)`` means retaining ``ceil(N/2)`` and
    ``ceil(N/4)`` keys per query, corresponding to the paper's ``N/k1`` and
    ``N/k2`` notation for an illustrative choice ``k1=2, k2=4``.
    """

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 4,
        pool_kernel_size: int = 3,
        dilation: int = 2,
        deform_kernel_size: int = 3,
        eta: Sequence[Number] = (0.4, 0.3, 0.3),
        topk_fractions: Sequence[Number] = (0.5, 0.25),
        alpha_beta: Sequence[Number] = (0.5, 0.5),
        attention_dim: Optional[int] = None,
        attention_chunk_size: Optional[int] = 1024,
        learnable_alpha_beta: bool = False,
        use_torchvision_deform: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.attention_dim = channels if attention_dim is None else int(attention_dim)
        if self.attention_dim <= 0 or self.attention_dim % self.num_heads != 0:
            raise ValueError("attention_dim must be positive and divisible by num_heads")
        self.head_dim = self.attention_dim // self.num_heads

        if not isinstance(pool_kernel_size, int) or pool_kernel_size <= 0 or pool_kernel_size % 2 == 0:
            raise ValueError("pool_kernel_size must be a positive odd integer")
        if not isinstance(dilation, int) or dilation <= 0:
            raise ValueError("dilation must be positive")
        self.pool_kernel_size = int(pool_kernel_size)
        self.dilation = int(dilation)

        eta_tuple = _as_float_tuple(eta, 3, "eta")
        if any(v < 0 for v in eta_tuple) or not math.isclose(sum(eta_tuple), 1.0, abs_tol=1e-5):
            raise ValueError("eta must be non-negative and sum to one")
        self.register_buffer("eta", torch.tensor(eta_tuple, dtype=torch.float32))

        fractions = _as_float_tuple(topk_fractions, 2, "topk_fractions")
        if any((not math.isfinite(v)) or v <= 0 or v > 1 for v in fractions):
            raise ValueError("topk_fractions must lie in (0, 1]")
        self.topk_fractions = fractions

        mix = _as_float_tuple(alpha_beta, 2, "alpha_beta")
        if any(not math.isfinite(v) for v in mix):
            raise ValueError("alpha_beta must be finite")
        if learnable_alpha_beta and any(v <= 0 for v in mix):
            raise ValueError("learnable alpha_beta values must be positive")
        self.learnable_alpha_beta = bool(learnable_alpha_beta)
        if self.learnable_alpha_beta:
            self.mix_logits = nn.Parameter(torch.log(torch.tensor(mix, dtype=torch.float32)))
        else:
            self.register_buffer("mix_weights", torch.tensor(mix, dtype=torch.float32))

        self.pool = nn.AvgPool2d(
            kernel_size=self.pool_kernel_size,
            stride=1,
            padding=self.pool_kernel_size // 2,
            count_include_pad=False,
        )
        self.deform = ModulatedDeformConv2d(
            channels,
            channels,
            kernel_size=deform_kernel_size,
            use_torchvision=use_torchvision_deform,
        )
        self.dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )
        self.bn_pool = nn.BatchNorm2d(channels)
        self.bn_deform = nn.BatchNorm2d(channels)
        self.bn_dilated = nn.BatchNorm2d(channels)

        # Eq. (20): Q = Linear(Flatten(Fin)).  The lower path in Fig. 7 shows
        # one Linear + LayerNorm before the K/V split; a 2D projection followed
        # by one LayerNorm captures that diagram while retaining distinct K,V.
        self.q_proj = nn.Linear(channels, self.attention_dim)
        self.kv_proj = nn.Linear(channels, 2 * self.attention_dim)
        self.kv_norm = nn.LayerNorm(2 * self.attention_dim)
        self.out_proj = (
            nn.Identity()
            if self.attention_dim == channels
            else nn.Linear(self.attention_dim, channels)
        )
        if attention_chunk_size is not None:
            if not isinstance(attention_chunk_size, int) or attention_chunk_size <= 0:
                raise ValueError("attention_chunk_size must be a positive integer or None")
            attention_chunk_size = int(attention_chunk_size)
        self.attention_chunk_size = attention_chunk_size

    def _mix(self) -> Tensor:
        if self.learnable_alpha_beta:
            return torch.softmax(self.mix_logits, dim=0)
        return self.mix_weights

    @staticmethod
    def _flatten_map(x: Tensor) -> Tensor:
        # [B,C,H,W] -> [B,N,C], matching the paper's Flatten convention.
        return x.flatten(2).transpose(1, 2).contiguous()

    def _resolve_topk(self, n: int) -> Tuple[int, int]:
        counts = []
        for fraction in self.topk_fractions:
            # Fractions are intentionally interpreted as retained proportions;
            # ceil keeps at least one key for small feature maps.
            count = int(math.ceil(fraction * n))
            counts.append(max(1, min(n, count)))
        return counts[0], counts[1]

    def _sparse_attention(self, q: Tensor, k: Tensor, v: Tensor, keep: int) -> Tensor:
        """Compute one masked scaled dot-product attention path.

        ``keep`` keys are selected independently for every query and head.  A
        query chunk limits peak memory but does not change the mathematical
        result, apart from normal floating-point associativity.
        """
        b, heads, n, d = q.shape
        kt = k.transpose(-2, -1)
        chunk = self.attention_chunk_size or n
        outputs = []
        scale = 1.0 / math.sqrt(d)
        neg_inf = torch.finfo(q.dtype).min
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            logits = torch.matmul(q[:, :, start:stop], kt) * scale
            if keep < n:
                indices = torch.topk(logits, k=keep, dim=-1, largest=True, sorted=False).indices
                selected = torch.zeros_like(logits, dtype=torch.bool)
                selected.scatter_(-1, indices, True)
                logits = logits.masked_fill(~selected, neg_inf)
            weights = torch.softmax(logits, dim=-1)
            outputs.append(torch.matmul(weights, v))
        return torch.cat(outputs, dim=2)

    def forward(self, x: Tensor, *, return_aux: bool = False):
        if x.ndim != 4:
            raise ValueError(f"expected input [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {x.shape[1]}")
        if x.shape[2] <= 0 or x.shape[3] <= 0:
            raise ValueError("spatial dimensions must be positive")

        # Eq. (15)--(18).
        f1 = F.relu(self.bn_pool(self.pool(x)))
        f2 = F.relu(self.bn_deform(self.deform(x)))
        f3 = F.relu(self.bn_dilated(self.dilated(x)))
        eta = self.eta.to(device=x.device, dtype=x.dtype)
        fagg = eta[0] * f1 + eta[1] * f2 + eta[2] * f3

        fin_seq = self._flatten_map(x)
        agg_seq = self._flatten_map(fagg)
        b, n, _ = fin_seq.shape

        q = self.q_proj(fin_seq).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_norm(self.kv_proj(agg_seq))
        k, v = kv.chunk(2, dim=-1)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        keep1, keep2 = self._resolve_topk(n)
        o1 = self._sparse_attention(q, k, v, keep1)
        o2 = self._sparse_attention(q, k, v, keep2)
        mix = self._mix().to(device=x.device, dtype=o1.dtype)
        out = mix[0] * o1 + mix[1] * o2
        out = out.transpose(1, 2).contiguous().view(b, n, self.attention_dim)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(b, self.channels, x.shape[2], x.shape[3])

        if not return_aux:
            return out
        # Keep auxiliary tensors useful for inspection without materializing
        # the full N x N attention matrices by default.
        aux: Dict[str, Tensor | Tuple[int, int]] = {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "fagg": fagg,
            "q": q,
            "k": k,
            "v": v,
            "topk_counts": (keep1, keep2),
        }
        return out, aux


# A concise alias helps users map existing paper notation to the renamed
# implementation without making MSACA the primary class name.
TDSCA = TriScaleDualSparseContextAggregator
MSACA = TriScaleDualSparseContextAggregator


__all__ = [
    "ModulatedDeformConv2d",
    "TriScaleDualSparseContextAggregator",
    "TDSCA",
    "MSACA",
]
