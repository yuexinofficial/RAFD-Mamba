from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .wavelet import haar_dwt, haar_idwt_high
GATE_BIAS_INIT = -4.0

class MultiScaleFreqResidual(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        six_c = 6 * self.dim
        self.in_proj = nn.Conv2d(six_c, six_c, 1, groups=6)
        self.dw = nn.Conv2d(six_c, six_c, 3, padding=1, groups=six_c)
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(self.dim, self.dim, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _bands(x):
        ll1, lh1, hl1, hh1 = haar_dwt(x)
        _, lh2, hl2, hh2 = haar_dwt(ll1)
        return (torch.cat([lh1, hl1, hh1], dim=-1), torch.cat([lh2, hl2, hh2], dim=-1), ll1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        hf1, hf2, _ = self._bands(x)
        hf2 = F.interpolate(hf2.permute(0, 3, 1, 2), size=hf1.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        hf = torch.cat([hf1, hf2], dim=-1)
        hf = hf.permute(0, 3, 1, 2)
        hf = self.act(self.dw(self.in_proj(hf)))
        hf = hf.permute(0, 2, 3, 1)
        b1, b2 = hf.chunk(2, dim=-1)
        l1 = b1.chunk(3, dim=-1)
        l2 = b2.chunk(3, dim=-1)
        r1 = haar_idwt_high(l1[0], l1[1], l1[2], (h, w))
        r2 = haar_idwt_high(l2[0], l2[1], l2[2], (h // 2, w // 2))
        r2 = F.interpolate(r2.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        r = r1 + r2
        r = r.permute(0, 3, 1, 2)
        return self.out_proj(r).permute(0, 2, 3, 1)

class MultiScaleFreqGate(nn.Module):

    def __init__(self, dim: int, gate_bias_init: float=GATE_BIAS_INIT, fixed_lambda: float | None=None):
        super().__init__()
        self.dim = int(dim)
        self.res = MultiScaleFreqResidual(self.dim)
        self.fixed_lambda = None if fixed_lambda is None else float(fixed_lambda)
        if self.fixed_lambda is None:
            self.gate = nn.Linear(3 * self.dim, self.dim)
            self.gate_bias = nn.Parameter(torch.full((self.dim,), float(gate_bias_init)))
            nn.init.zeros_(self.gate.bias)
        self.reset_stats()

    def reset_stats(self):
        self._n = 0.0
        self._lam = 0.0
        self._lam_sq = 0.0
        self._rho = 0.0
        self._rho_sq = 0.0
        self._rho_lam = 0.0
        self._f = 0.0
        self._i = 0.0

    @torch.no_grad()
    def _accumulate(self, x, r, lam, rho, inject):
        b = x.shape[0]
        n = lambda t: t.detach().reshape(b, -1).norm(dim=1)
        self._n += b
        self._lam += float(lam.detach().mean()) * b
        self._lam_sq += float((lam.detach() ** 2).mean()) * b
        rc = rho.detach()
        lc = lam.detach().mean(dim=(1, 2))
        if self.fixed_lambda is not None:
            self._rho_lam = float('nan')
        self._rho += float(rc.mean()) * b
        self._rho_sq += float((rc ** 2).mean()) * b
        self._rho_lam += float((rc * lc).mean()) * b
        self._f += float(n(x).sum())
        self._i += float(n(inject).sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        hf1, hf2, _ = MultiScaleFreqResidual._bands(x)

        def _energy(hf):
            bb, hh, ww, _ = hf.shape
            return hf.view(bb, hh, ww, 3, c).abs().mean(dim=(1, 2)).sum(dim=1)
        e1, e2 = (_energy(hf1), _energy(hf2))
        rho = e2 / (e1 + 1e-06)
        r = self.res(x)
        rho4 = rho.view(b, 1, 1, c).expand(-1, h, w, -1)
        if self.fixed_lambda is None:
            lam = torch.sigmoid(self.gate(torch.cat([x, r, rho4], dim=-1)) + self.gate_bias)
        else:
            lam = torch.full_like(r, self.fixed_lambda)
        inject = lam * r
        if not self.training:
            self._accumulate(x, r, lam, rho, inject)
        return x + inject

    @torch.no_grad()
    def stats(self):
        nan = float('nan')
        if self._n == 0:
            return dict(lambda_mean=nan, lambda_std=nan, rho_mean=nan, rho_std=nan, corr_rho_lambda=nan, E=nan)
        n = self._n
        lm = self._lam / n
        var = max(self._lam_sq / n - lm ** 2, 0.0)
        rm = self._rho / n
        rvar = max(self._rho_sq / n - rm ** 2, 0.0)
        cov = self._rho_lam / n - rm * lm
        corr = cov / (var ** 0.5 * rvar ** 0.5) if var > 0 and rvar > 0 else nan
        f, i = (self._f, self._i)
        return dict(lambda_mean=lm, lambda_std=var ** 0.5, rho_mean=rm, rho_std=rvar ** 0.5, corr_rho_lambda=corr, E=i / f if f > 0 else nan)
MSFR = MultiScaleFreqGate
__all__ = ['MultiScaleFreqGate', 'MultiScaleFreqResidual', 'MSFR', 'GATE_BIAS_INIT']
