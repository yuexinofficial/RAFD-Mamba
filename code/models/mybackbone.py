from __future__ import annotations
import os
import sys
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs as C
from models.modules import MSACA, ReliabilityAwareFreqResidual
from models.modules.dmsc import DynamicMultiScaleSparseContext
from models.modules.msaca_dual import DualInputMSACA
from models.modules.hae import MSC2 as HAEModule
from models.modules.lge import SplitLocalGlobalEnhance
from models.modules.msfr import MultiScaleFreqGate
from models.modules.sco import BasicLayer_mlp as SCOLayer, MyNorm as SCONorm
from models.vmunet import VMUNet

class Skip0Conn(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(int(dim), int(dim), 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(s.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

def _empty_or(d, levels, build):
    if not levels:
        return nn.ModuleDict()
    return nn.ModuleDict({str(i): build(i) for i in levels})

class VMUNetRAFR(VMUNet):

    def __init__(self, *args, rafr_levels=(1, 2, 3), rafr_gate_bias=-4.0, msaca_levels=(), msaca_heads=4, msaca_dual_levels=(), dmsc_levels=(), dmsc_heads=4, dmsc_lambda=True, sco_levels=(), sco_shift_size=1, msfr_levels=(), msfr_gate_bias=-4.0, skip0_conn=False, lge_levels=(), msfr_fixed_lambda=None, hae_levels=(), **kwargs):
        super().__init__(*args, **kwargs)
        dims = [int(v) for v in self.vmunet.dims]
        _d = lambda i: dims[i] if i < len(dims) else dims[-1]
        self.rafr_levels = tuple((int(i) for i in rafr_levels))
        self.msaca_levels = tuple((int(i) for i in msaca_levels))
        self.msaca_dual_levels = tuple((int(i) for i in msaca_dual_levels))
        self.dmsc_levels = tuple((int(i) for i in dmsc_levels))
        self.msfr_levels = tuple((int(i) for i in msfr_levels))
        self.skip0_conn_on = bool(skip0_conn)
        self.lge_levels = tuple((int(i) for i in lge_levels))
        self.hae_levels = tuple((int(i) for i in hae_levels))
        self.sco_levels = tuple((int(i) for i in sco_levels))
        self.sco_shift_size = int(sco_shift_size)
        _clash = set(self.sco_levels) & (set(self.rafr_levels) | set(self.msaca_levels) | set(self.msaca_dual_levels) | set(self.msfr_levels))
        if _clash:
            raise ValueError(f'SCO 与 A/MSACA 撞在同一层 {sorted(_clash)}:SCO 把 skip 劈成 dims/2,而 RAFR/MSACA 按 dims 构造')
        self._as_list = []
        with torch.random.fork_rng(devices=[]):
            self.skip_rafr = _empty_or(None, self.rafr_levels, lambda i: ReliabilityAwareFreqResidual(_d(i), rafr_gate_bias))
            self.skip_msaca = _empty_or(None, self.msaca_levels, lambda i: MSACA(channels=_d(i), num_heads=msaca_heads))
            self.skip_msaca_dual = _empty_or(None, self.msaca_dual_levels, lambda i: DualInputMSACA(channels_q=_d(i), channels_kv=_d(i + 1), num_heads=msaca_heads))
            self.dmsc = _empty_or(None, self.dmsc_levels, lambda i: DynamicMultiScaleSparseContext(_d(i), num_heads=dmsc_heads, use_lambda=dmsc_lambda))
            self.dmsc_lambda = bool(dmsc_lambda)
            self.skip_msfr = _empty_or(None, self.msfr_levels, lambda i: MultiScaleFreqGate(_d(i), msfr_gate_bias, fixed_lambda=msfr_fixed_lambda))
            self.skip0_conn = Skip0Conn(dims[0]) if self.skip0_conn_on else None
            self.lge = _empty_or(None, self.lge_levels, lambda i: SplitLocalGlobalEnhance(_d(i)))
            self.hae = _empty_or(None, self.hae_levels, lambda i: HAEModule(_d(i)))
            _dpr_rate = 0.0
            try:
                _dpr_rate = float(self.vmunet.layers[-1].blocks[-1].drop_path.drop_prob)
            except Exception:
                pass
            _depths = [len(l.blocks) for l in self.vmunet.layers]
            _dpr = torch.linspace(0, _dpr_rate, sum(_depths)).tolist()
            self.sco = _empty_or(None, self.sco_levels, lambda i: SCOLayer(dim=_d(i) // 2, input_resolution=(0, 0), depth=_depths[i], shift_size=self.sco_shift_size, mlp_ratio=1, drop_path=_dpr[sum(_depths[:i]):sum(_depths[:i + 1])], norm_layer=SCONorm, downsample=None))

    def _encoder_with_dmsc(self, x):
        v = self.vmunet
        skip_list, as_list = ([], [])
        x = v.patch_embed(x)
        if v.ape:
            x = x + v.absolute_pos_embed
        x = v.pos_drop(x)
        for i, layer in enumerate(v.layers):
            key = str(i)
            if key in self.sco:
                x1, x2 = torch.chunk(x, 2, dim=3)
                skip_list.append(x1)
                as_list.append(self.sco[key](x2))
            else:
                skip_list.append(x)
                as_list.append(None)
            if key in self.hae:
                x = self.hae[key](x)
            if key in self.lge:
                x = self.lge[key](x)
            if key in self.dmsc:
                x = self.dmsc[key](x)
            x = layer(x)
        out_key = str(len(v.layers))
        if out_key in self.skip_msfr:
            x = self.skip_msfr[out_key](x)
        if out_key in self.skip_rafr:
            x = self.skip_rafr[out_key](x)
        return (x, skip_list, as_list)

    def forward_features(self, x):
        _out = '4' in self.skip_msfr or '4' in self.skip_rafr
        if len(self.dmsc) or len(self.sco) or len(self.lge) or len(self.hae) or _out:
            x, skip_list, as_list = self._encoder_with_dmsc(x)
            self._as_list = as_list
        else:
            x, skip_list = self.vmunet.forward_features(x)
            self._as_list = []
        if len(self.skip_rafr) or len(self.skip_msaca) or len(self.skip_msaca_dual) or len(self.skip_msfr):
            skip_list = list(skip_list)
            _kv_src = list(skip_list) + [x]
            _ns = len(skip_list)
            for i, mod in self.skip_rafr.items():
                if int(i) < _ns:
                    skip_list[int(i)] = mod(skip_list[int(i)])
            for i, mod in self.skip_msfr.items():
                if int(i) < _ns:
                    skip_list[int(i)] = mod(skip_list[int(i)])
            for i, mod in self.skip_msaca.items():
                _s = skip_list[int(i)]
                _s = mod(_s.permute(0, 3, 1, 2))
                skip_list[int(i)] = _s.permute(0, 2, 3, 1)
            for i, mod in self.skip_msaca_dual.items():
                k = int(i)
                kv = _kv_src[k + 1]
                skip_list[k] = mod(skip_list[k].permute(0, 3, 1, 2), kv.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return (x, skip_list)

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.vmunet.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                s = skip_list[-inx]
                a = self._as_list[-inx] if self._as_list else None
                if a is not None:
                    s = torch.cat([s, a], dim=3)
                x = layer_up(x + s)
        if self.skip0_conn is not None and len(skip_list):
            x = x + self.skip0_conn(skip_list[0])
        return x

    def forward_logits(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.vmunet.forward_final(x)
        return x

class MyBackboneRAFRSeg(nn.Module):
    ARCHITECTURE = 'vmunet_rafr'

    def __init__(self, num_classes=1, input_channels=3, depths=None, depths_decoder=None, drop_path_rate=None, load_ckpt_path=None, img_size=None, rafr_levels=None, rafr_gate_bias=-4.0, msaca_levels=None, msaca_heads=4, msaca_dual_levels=None, dmsc_levels=None, dmsc_heads=4, dmsc_lambda=True, sco_levels=None, sco_shift_size=1, msfr_levels=None, msfr_gate_bias=-4.0, skip0_conn=False, lge_levels=None, msfr_fixed_lambda=None, hae_levels=None, arch_name=None, **_ignored):
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError('本项目只接二值分割 (num_classes=1)')
        if arch_name:
            self.ARCHITECTURE = str(arch_name)
        self.num_classes = int(num_classes)
        self.img_size = int(img_size if img_size is not None else C.IMG_SIZE)
        p = C.ARCH_PROTOCOLS['mybackbone']
        self.depths = tuple(depths if depths is not None else p['depths'])
        self.depths_decoder = tuple(depths_decoder if depths_decoder is not None else p['depths_decoder'])
        self.drop_path_rate = float(drop_path_rate if drop_path_rate is not None else p['drop_path_rate'])
        self.load_ckpt_path = load_ckpt_path
        self.norm = p['norm']
        self.output_mode = p['output_mode']
        self.rafr_levels = tuple(rafr_levels if rafr_levels is not None else ())
        self.msaca_levels = tuple(msaca_levels if msaca_levels is not None else ())
        self.msaca_dual_levels = tuple(msaca_dual_levels if msaca_dual_levels is not None else ())
        self.dmsc_levels = tuple(dmsc_levels if dmsc_levels is not None else ())
        self.dmsc_lambda = bool(dmsc_lambda)
        self.sco_levels = tuple(sco_levels if sco_levels is not None else ())
        self.sco_shift_size = int(sco_shift_size)
        self.msfr_levels = tuple(msfr_levels if msfr_levels is not None else ())
        self.skip0_conn = bool(skip0_conn)
        self.lge_levels = tuple(lge_levels if lge_levels is not None else ())
        self.hae_levels = tuple(hae_levels if hae_levels is not None else ())
        self.backbone = VMUNetRAFR(input_channels=int(input_channels), num_classes=int(num_classes), depths=list(self.depths), depths_decoder=list(self.depths_decoder), drop_path_rate=self.drop_path_rate, load_ckpt_path=load_ckpt_path, rafr_levels=self.rafr_levels, rafr_gate_bias=rafr_gate_bias, msaca_levels=self.msaca_levels, msaca_heads=msaca_heads, msaca_dual_levels=self.msaca_dual_levels, dmsc_levels=self.dmsc_levels, dmsc_heads=dmsc_heads, dmsc_lambda=self.dmsc_lambda, sco_levels=self.sco_levels, sco_shift_size=self.sco_shift_size, msfr_levels=self.msfr_levels, msfr_gate_bias=msfr_gate_bias, skip0_conn=self.skip0_conn, lge_levels=self.lge_levels, msfr_fixed_lambda=msfr_fixed_lambda, hae_levels=self.hae_levels)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.backbone.forward_logits(x)
        zero = torch.zeros_like(logits[:, :1])
        return torch.cat([zero, logits], dim=1)

    def load_from(self):
        p = self.load_ckpt_path
        if p is None:
            raise ValueError('未指定 load_ckpt_path')
        if not os.path.exists(p):
            raise FileNotFoundError(f'预训练权重不存在: {p}')
        self.backbone.load_from()

    def reset_msaca_stats(self):
        return None

    def reset_rafr_stats(self):
        for m in self.backbone.skip_rafr.values():
            m.reset_stats()

    def rafr_stats(self):
        return {k: m.stats() for k, m in self.backbone.skip_rafr.items()}

    def reset_msfr_stats(self):
        for m in self.backbone.skip_msfr.values():
            m.reset_stats()

    def msfr_stats(self):
        return {k: m.stats() for k, m in self.backbone.skip_msfr.items()}

    def reset_dmsc_stats(self):
        for m in self.backbone.dmsc.values():
            m.reset_stats()

    def dmsc_stats(self):
        return {k: m.stats() for k, m in self.backbone.dmsc.items()}

    def export_config(self):
        return dict(architecture=self.ARCHITECTURE, num_classes=self.num_classes, img_size=self.img_size, depths=list(self.depths), depths_decoder=list(self.depths_decoder), drop_path_rate=self.drop_path_rate, norm=self.norm, output_mode=self.output_mode, rafr_levels=list(self.rafr_levels), msaca_levels=list(self.msaca_levels), msaca_dual_levels=list(self.msaca_dual_levels), dmsc_levels=list(self.dmsc_levels), dmsc_lambda=self.dmsc_lambda, sco_levels=list(self.sco_levels), sco_shift_size=self.sco_shift_size, msfr_levels=list(self.msfr_levels), skip0_conn=self.skip0_conn, lge_levels=list(self.lge_levels), hae_levels=list(self.hae_levels))
