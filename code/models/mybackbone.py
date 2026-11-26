from __future__ import annotations
import os
import sys
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs as C
from models.modules import ReliabilityAwareFreqResidual
from models.modules.dmsc import DynamicMultiScaleSparseContext
from models.vmunet import VMUNet

class VMUNetRAFR(VMUNet):

    def __init__(self, *args, rafr_levels=(1, 2, 3, 4), rafr_gate_bias=-4.0, dmsc_levels=(2, 3), dmsc_heads=4, dmsc_lambda=True, **kwargs):
        super().__init__(*args, **kwargs)
        dims = [int(v) for v in self.vmunet.dims]
        _d = lambda i: dims[i] if i < len(dims) else dims[-1]
        self.rafr_levels = tuple((int(i) for i in rafr_levels))
        self.dmsc_levels = tuple((int(i) for i in dmsc_levels))
        with torch.random.fork_rng(devices=[]):
            self.skip_rafr = nn.ModuleDict({str(i): ReliabilityAwareFreqResidual(_d(i), rafr_gate_bias) for i in self.rafr_levels})
            self.dmsc = nn.ModuleDict({str(i): DynamicMultiScaleSparseContext(_d(i), num_heads=dmsc_heads, use_lambda=dmsc_lambda) for i in self.dmsc_levels})

    def _encoder_with_modules(self, x):
        v = self.vmunet
        skip_list = []
        x = v.patch_embed(x)
        if v.ape:
            x = x + v.absolute_pos_embed
        x = v.pos_drop(x)
        for i, layer in enumerate(v.layers):
            key = str(i)
            skip_list.append(x)
            if key in self.dmsc:
                x = self.dmsc[key](x)
            x = layer(x)
        out_key = str(len(v.layers))
        if out_key in self.skip_rafr:
            x = self.skip_rafr[out_key](x)
        return (x, skip_list)

    def forward_features(self, x):
        x, skip_list = self._encoder_with_modules(x)
        skip_list = list(skip_list)
        for i, mod in self.skip_rafr.items():
            if int(i) < len(skip_list):
                skip_list[int(i)] = mod(skip_list[int(i)])
        return (x, skip_list)

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.vmunet.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
        return x

    def forward_logits(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.vmunet.forward_final(x)
        return x

class MyBackboneRAFRSeg(nn.Module):
    ARCHITECTURE = 'vmunet_rafr'

    def __init__(self, num_classes=1, input_channels=3, depths=None, depths_decoder=None, drop_path_rate=None, load_ckpt_path=None, img_size=None, rafr_levels=None, rafr_gate_bias=-4.0, dmsc_levels=None, dmsc_heads=4, dmsc_lambda=True, arch_name=None, **_ignored):
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError('二值分割 (num_classes=1)')
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
        self.dmsc_levels = tuple(dmsc_levels if dmsc_levels is not None else ())
        self.dmsc_lambda = bool(dmsc_lambda)
        self.backbone = VMUNetRAFR(input_channels=int(input_channels), num_classes=int(num_classes), depths=list(self.depths), depths_decoder=list(self.depths_decoder), drop_path_rate=self.drop_path_rate, load_ckpt_path=load_ckpt_path, rafr_levels=self.rafr_levels, rafr_gate_bias=rafr_gate_bias, dmsc_levels=self.dmsc_levels, dmsc_heads=dmsc_heads, dmsc_lambda=self.dmsc_lambda)

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

    def reset_rafr_stats(self):
        for m in self.backbone.skip_rafr.values():
            m.reset_stats()

    def rafr_stats(self):
        return {k: m.stats() for k, m in self.backbone.skip_rafr.items()}

    def reset_dmsc_stats(self):
        for m in self.backbone.dmsc.values():
            m.reset_stats()

    def dmsc_stats(self):
        return {k: m.stats() for k, m in self.backbone.dmsc.items()}

    def export_config(self):
        return dict(architecture=self.ARCHITECTURE, num_classes=self.num_classes, img_size=self.img_size, depths=list(self.depths), depths_decoder=list(self.depths_decoder), drop_path_rate=self.drop_path_rate, norm=self.norm, output_mode=self.output_mode, rafr_levels=list(self.rafr_levels), dmsc_levels=list(self.dmsc_levels), dmsc_lambda=self.dmsc_lambda)
