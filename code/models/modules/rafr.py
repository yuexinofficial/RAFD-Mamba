from __future__ import annotations
import torch
from torch import nn
from .wavelet import haar_dwt, haar_idwt_high
GATE_BIAS_INIT = -4.0

class FreqResidual(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        three_c = 3 * self.dim
        self.in_proj = nn.Conv2d(three_c, three_c, 1, groups=3)
        self.dw = nn.Conv2d(three_c, three_c, 3, padding=1, groups=three_c)
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(self.dim, self.dim, 1)
        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        _, lh, hl, hh = haar_dwt(x)
        hf = torch.cat([lh, hl, hh], dim=-1)
        hf = hf.permute(0, 3, 1, 2)
        hf = self.dw(self.in_proj(hf))
        hf = self.act(hf)
        hf = hf.permute(0, 2, 3, 1)
        lh2, hl2, hh2 = hf.chunk(3, dim=-1)
        r = haar_idwt_high(lh2, hl2, hh2, (h, w))
        r = r.permute(0, 3, 1, 2)
        return self.out_proj(r).permute(0, 2, 3, 1)

class ReliabilityAwareFreqResidual(nn.Module):

    def __init__(self, dim: int, gate_bias_init: float=GATE_BIAS_INIT):
        super().__init__()
        self.dim = int(dim)
        self.res = FreqResidual(self.dim)
        self.gate = nn.Linear(2 * self.dim, self.dim)
        self.gate_bias = nn.Parameter(torch.full((self.dim,), float(gate_bias_init)))
        nn.init.zeros_(self.gate.bias)
        self.reset_stats()

    def reset_stats(self):
        self._n = 0.0
        self._lam = 0.0
        self._lam_sq = 0.0
        self._f = 0.0
        self._r = 0.0
        self._i = 0.0

    @torch.no_grad()
    def _accumulate(self, x, r, lam, inject):
        b = x.shape[0]
        n = lambda t: t.detach().reshape(b, -1).norm(dim=1)
        self._n += b
        self._lam += float(lam.detach().mean()) * b
        self._lam_sq += float((lam.detach() ** 2).mean()) * b
        self._f += float(n(x).sum())
        self._r += float(n(r).sum())
        self._i += float(n(inject).sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.res(x)
        lam = torch.sigmoid(self.gate(torch.cat([x, r], dim=-1)) + self.gate_bias)
        inject = lam * r
        self._accumulate(x, r, lam, inject)
        return x + inject

    @torch.no_grad()
    def stats(self) -> dict:
        if self._n == 0:
            nan = float('nan')
            return {'lambda_mean': nan, 'lambda_std': nan, 'R_norm': nan, 'F_norm': nan, 'inj_norm': nan, 'E': nan}
        n = self._n
        lam_mean = self._lam / n
        var = max(self._lam_sq / n - lam_mean ** 2, 0.0)
        f, i = (self._f / n, self._i / n)
        return {'lambda_mean': lam_mean, 'lambda_std': var ** 0.5, 'R_norm': self._r / n, 'F_norm': f, 'inj_norm': i, 'E': self._i / self._f if self._f > 0 else float('nan')}
