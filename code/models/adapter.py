from __future__ import annotations
import os
import sys
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs as C
from models.vmunet import VMUNet

class VMUNetSeg(nn.Module):
    ARCHITECTURE = 'vmunet'

    def __init__(self, num_classes=1, input_channels=3, depths=None, depths_decoder=None, drop_path_rate=None, load_ckpt_path=None, img_size=None, **_ignored):
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError('本项目只接二值分割 (num_classes=1)')
        self.num_classes = int(num_classes)
        self.img_size = int(img_size if img_size is not None else C.IMG_SIZE)
        self.depths = tuple(depths if depths is not None else C.ARCH_PROTOCOLS['mybackbone']['depths'])
        self.depths_decoder = tuple(depths_decoder if depths_decoder is not None else C.ARCH_PROTOCOLS['mybackbone']['depths_decoder'])
        self.drop_path_rate = float(drop_path_rate if drop_path_rate is not None else C.ARCH_PROTOCOLS['mybackbone']['drop_path_rate'])
        self.load_ckpt_path = load_ckpt_path
        self.norm = C.ARCH_PROTOCOLS['vmunet']['norm']
        self.output_mode = C.ARCH_PROTOCOLS['vmunet']['output_mode']
        self.backbone = VMUNet(input_channels=int(input_channels), num_classes=int(num_classes), depths=list(self.depths), depths_decoder=list(self.depths_decoder), drop_path_rate=self.drop_path_rate, load_ckpt_path=load_ckpt_path)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.backbone.vmunet(x)
        zero = torch.zeros_like(logits[:, :1])
        return torch.cat([zero, logits], dim=1)

    def load_from(self):
        p = self.load_ckpt_path
        if p is None:
            raise ValueError('未指定 load_ckpt_path')
        if not os.path.exists(p):
            raise FileNotFoundError(f'预训练权重不存在: {p}')
        self.backbone.load_from()

    def export_config(self):
        return dict(architecture=self.ARCHITECTURE, num_classes=self.num_classes, img_size=self.img_size, depths=list(self.depths), depths_decoder=list(self.depths_decoder), drop_path_rate=self.drop_path_rate, norm=self.norm, output_mode=self.output_mode)

def build_model(architecture='vmunet', pretrained=None, **overrides):
    proto = C.proto_for_arch(architecture)
    if proto is None:
        raise KeyError(f'未知 architecture: {architecture};可用协议 {list(C.ARCH_PROTOCOLS)}')
    kw = dict(num_classes=int(proto.get('num_classes', 1)), input_channels=3, img_size=C.IMG_SIZE, load_ckpt_path=pretrained)
    kw.update(depths=proto['depths'], depths_decoder=proto['depths_decoder'], drop_path_rate=proto['drop_path_rate'])
    if 'rafr_levels' in proto:
        kw['rafr_levels'] = proto['rafr_levels']
        kw['rafr_gate_bias'] = proto.get('rafr_gate_bias', -4.0)
    if 'dmsc_levels' in proto:
        kw['dmsc_levels'] = proto['dmsc_levels']
        kw['dmsc_heads'] = proto.get('dmsc_heads', 4)
        kw['dmsc_lambda'] = proto.get('dmsc_lambda', True)
    kw.update({k: v for k, v in overrides.items() if v is not None})
    arch = proto.get('model', architecture)
    if arch == 'vmunet':
        return VMUNetSeg(**kw)
    if arch in ('vmunet_rafr', 'vmunet_dmsc', 'vmunet_rafr_dmsc', 'vmunet_rafr4', 'vmunet_rafr4_dmsc'):
        from models.mybackbone import MyBackboneRAFRSeg
        kw['arch_name'] = arch
        return MyBackboneRAFRSeg(**kw)
    raise KeyError(architecture)
