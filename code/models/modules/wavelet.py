from __future__ import annotations
import torch

def _to_even(x: torch.Tensor):
    b, h, w, c = x.shape
    hp, wp = ((h + 1) // 2 * 2, (w + 1) // 2 * 2)
    if (hp, wp) != (h, w):
        x = torch.nn.functional.pad(x, (0, 0, 0, wp - w, 0, hp - h))
    return (x, (h, w))

def haar_dwt(x: torch.Tensor):
    x, _ = _to_even(x)
    x00 = x[:, 0::2, 0::2, :]
    x01 = x[:, 0::2, 1::2, :]
    x10 = x[:, 1::2, 0::2, :]
    x11 = x[:, 1::2, 1::2, :]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x01 - x10 - x11) * 0.5
    hl = (x00 - x01 + x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return (ll, lh, hl, hh)

def haar_idwt_high(lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, out_hw) -> torch.Tensor:
    x00 = (lh + hl + hh) * 0.5
    x01 = (lh - hl - hh) * 0.5
    x10 = (-lh + hl - hh) * 0.5
    x11 = (-lh - hl + hh) * 0.5
    b, h, w, c = lh.shape
    out = torch.empty(b, 2 * h, 2 * w, c, dtype=lh.dtype, device=lh.device)
    out[:, 0::2, 0::2, :] = x00
    out[:, 0::2, 1::2, :] = x01
    out[:, 1::2, 0::2, :] = x10
    out[:, 1::2, 1::2, :] = x11
    oh, ow = out_hw
    return out[:, :oh, :ow, :]
