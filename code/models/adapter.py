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

    def __init__(self, num_classes: int=1, input_channels: int=3, depths=None, depths_decoder=None, drop_path_rate=None, load_ckpt_path: str=None, img_size: int=None, **_ignored):
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError('本项目只接二值分割 (num_classes=1)')
        self.num_classes = int(num_classes)
        self.img_size = int(img_size if img_size is not None else C.IMG_SIZE)
        self.depths = tuple(depths if depths is not None else C.MODEL['depths'])
        self.depths_decoder = tuple(depths_decoder if depths_decoder is not None else C.MODEL['depths_decoder'])
        self.drop_path_rate = float(drop_path_rate if drop_path_rate is not None else C.MODEL['drop_path_rate'])
        self.load_ckpt_path = load_ckpt_path
        self.norm = C.ARCH_PROTOCOLS['vmunet']['norm']
        self.output_mode = C.ARCH_PROTOCOLS['vmunet']['output_mode']
        self.backbone = VMUNet(input_channels=int(input_channels), num_classes=int(num_classes), depths=list(self.depths), depths_decoder=list(self.depths_decoder), drop_path_rate=self.drop_path_rate, load_ckpt_path=load_ckpt_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.backbone.vmunet(x)
        if self.num_classes == 1:
            zero = torch.zeros_like(logits[:, :1])
            return torch.cat([zero, logits], dim=1)
        return logits

    def load_from(self):
        p = self.load_ckpt_path
        if p is None:
            raise ValueError('未指定 load_ckpt_path')
        if not os.path.exists(p):
            raise FileNotFoundError(f'预训练权重不存在: {p}')
        self.backbone.load_from()

    def export_config(self) -> dict:
        return dict(architecture=self.ARCHITECTURE, num_classes=self.num_classes, img_size=self.img_size, depths=list(self.depths), depths_decoder=list(self.depths_decoder), drop_path_rate=self.drop_path_rate, norm=self.norm, output_mode=self.output_mode)

def build_model(architecture='vmunet', pretrained=None, **overrides):
    proto = C.proto_for_arch(architecture)
    if proto is None:
        raise KeyError(f'未知 architecture: {architecture};可用协议 {list(C.ARCH_PROTOCOLS)} 或架构 {sorted({p.get('model', k) for k, p in C.ARCH_PROTOCOLS.items()})}')
    kw = dict(num_classes=int(proto.get('num_classes', 1)), input_channels=3, img_size=C.IMG_SIZE, load_ckpt_path=pretrained)
    if 'depths' in proto:
        kw.update(depths=proto['depths'], depths_decoder=proto['depths_decoder'], drop_path_rate=proto['drop_path_rate'])
    if 'bilinear' in proto:
        kw['bilinear'] = proto['bilinear']
    if 'rafr_levels' in proto:
        kw['rafr_levels'] = proto['rafr_levels']
        kw['rafr_gate_bias'] = proto.get('rafr_gate_bias', -4.0)
    if 'msaca_levels' in proto:
        kw['msaca_levels'] = proto['msaca_levels']
        kw['msaca_heads'] = proto.get('msaca_heads', 4)
    if 'version' in proto:
        kw['version'] = proto['version']
    if 'msaca_dual_levels' in proto:
        kw['msaca_dual_levels'] = proto['msaca_dual_levels']
        kw['msaca_heads'] = proto.get('msaca_heads', 4)
    if 'dmsc_levels' in proto:
        kw['dmsc_levels'] = proto['dmsc_levels']
        kw['dmsc_heads'] = proto.get('dmsc_heads', 4)
        kw['dmsc_lambda'] = proto.get('dmsc_lambda', True)
    if 'sco_levels' in proto:
        kw['sco_levels'] = proto['sco_levels']
        kw['sco_shift_size'] = proto.get('sco_shift_size', 1)
    if 'msfr_levels' in proto:
        kw['msfr_levels'] = proto['msfr_levels']
        kw['msfr_gate_bias'] = proto.get('msfr_gate_bias', -4.0)
        kw['msfr_fixed_lambda'] = proto.get('msfr_fixed_lambda', None)
        kw['skip0_conn'] = proto.get('skip0_conn', False)
    if 'lge_levels' in proto:
        kw['lge_levels'] = proto['lge_levels']
    if 'hae_levels' in proto:
        kw['hae_levels'] = proto['hae_levels']
    if 'variant' in proto:
        kw['variant'] = proto['variant']
    kw.update({k: v for k, v in overrides.items() if v is not None})
    arch = proto.get('model', architecture)
    if arch == 'vmunet':
        return VMUNetSeg(**kw)
    if arch == 'eccomamba':
        from models.eccomamba_adapter import EccoMambaSeg
        return EccoMambaSeg(**kw)
    if arch in ('vmunet_rafr', 'vmunet_msaca', 'vmunet_msacadual', 'vmunet_dmsc', 'vmunet_rafr_dmsc', 'vmunet_dmsc_nolam', 'vmunet_dmsc_sco1', 'vmunet_dmsc_sco3', 'vmunet_msfr', 'vmunet_msfr_dmsc', 'vmunet_lge', 'vmunet_msfr_lge', 'vmunet_msfrfix_dmsc', 'vmunet_rafr4', 'vmunet_rafr4_dmsc', 'vmunet_rafr_dmsc4', 'vmunet_rafr_lge', 'vmunet_rafr_dmsc123', 'vmunet_rafr4_dmsc123', 'vmunet_rafr4_hae'):
        from models.mybackbone import MyBackboneRAFRSeg
        kw['arch_name'] = arch
        return MyBackboneRAFRSeg(**kw)
    if arch == 'dacnet':
        from models.dacnet_adapter import DACNetSeg
        return DACNetSeg(**kw)
    if arch in ('convnextv2_unet', 'convnextv2_unet_rafr', 'convnextv2_unet_dmsc', 'convnextv2_unet_rafr_dmsc', 'convnextv2_unet_nano'):
        from models.convnextv2_unet import ConvNeXtV2UNetSeg
        if arch == 'convnextv2_unet_nano':
            kw['arch_name'] = arch
        return ConvNeXtV2UNetSeg(**kw)
    if arch == 'att_unet':
        from models.attunet_adapter import AttUNetSeg
        kw['feature_scale'] = proto.get('feature_scale', 4)
        kw.pop('depths', None)
        kw.pop('depths_decoder', None)
        kw.pop('drop_path_rate', None)
        return AttUNetSeg(**kw)
    if arch == 'swinunet':
        from models.swinunet_adapter import SwinUnetSeg
        kw.pop('depths', None)
        kw.pop('depths_decoder', None)
        kw.pop('drop_path_rate', None)
        return SwinUnetSeg(**kw)
    if arch == 'swin_umamba':
        from models.swin_umamba_adapter import SwinUMambaSeg
        kw.pop('depths', None)
        kw.pop('depths_decoder', None)
        kw.pop('drop_path_rate', None)
        return SwinUMambaSeg(**kw)
    if arch == 'nnunet_plain':
        from models.nnunet_plain_adapter import NNUNetPlainSeg
        kw.pop('depths', None)
        kw.pop('depths_decoder', None)
        kw.pop('drop_path_rate', None)
        return NNUNetPlainSeg(**kw)
    if arch == 'cfformer':
        from models.cfformer_adapter import CFFormerSeg
        return CFFormerSeg(**kw)
    if arch == 'pytorch_unet':
        from models.pytorch_unet_adapter import PytorchUNetSeg
        return PytorchUNetSeg(**{k: v for k, v in kw.items() if k != 'depths'})
    raise KeyError(architecture)
