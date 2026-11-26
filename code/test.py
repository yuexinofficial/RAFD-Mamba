import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import configs as C
from data.thyroid_dataset import build_dataset
from metrics import aggregate, binarize_logits, image_metrics
from models.adapter import build_model
CKPT_ORDER = ['best_model.pth', 'final_model.pth', 'last_model.pth']

def resolve_ckpts(path, include_diagnostics=False):
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    if include_diagnostics:
        out = [os.path.join(path, n) for n in CKPT_ORDER if os.path.exists(os.path.join(path, n))]
        return out or [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.pth')][:2]
    best = os.path.join(path, CKPT_ORDER[0])
    if os.path.exists(best):
        return [best]
    fb = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.pth')]
    if not fb:
        raise FileNotFoundError(f'{path} 下没有任何 .pth')
    print(f'  ⚠️  {path} 没有 best_model.pth,退而评 {os.path.basename(fb[0])}')
    return fb[:1]

def load_model(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = None
    for k in ('state_dict', 'model_state_dict', 'model'):
        if k in payload:
            sd = payload[k]
            break
    if sd is None:
        sd = payload
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    cfg = payload.get('model_config')
    if cfg is None:
        raise RuntimeError(f'{ckpt_path} 缺少 model_config —— 不是本项目产出的 ckpt')
    arch = cfg.get('architecture')
    _proto = C.proto_for_arch(arch)
    if _proto is None:
        raise RuntimeError(f'未知 architecture: {arch}')
    if 'norm' not in cfg:
        fallback = (_proto or {}).get('norm')
        if fallback is None:
            raise RuntimeError(f'{ckpt_path} 的 model_config 缺少 norm,且架构 {arch} 的协议里也没有,无法确定归一化尺度,拒绝继续')
        print(f'  ⚠️  {os.path.basename(ckpt_path)} 的 model_config 无 norm 字段(早期 run),按架构 {arch} 的协议取 norm={fallback}')
        cfg = dict(cfg)
        cfg['norm'] = fallback
    if 'output_mode' not in cfg:
        fallback = (_proto or {}).get('output_mode', 'sigmoid_binary')
        print(f'  ⚠️  {os.path.basename(ckpt_path)} 无 output_mode 字段(早期 run),按架构 {arch} 取 {fallback}')
        cfg = dict(cfg)
        cfg['output_mode'] = fallback
    _kw = dict(depths=cfg.get('depths'), depths_decoder=cfg.get('depths_decoder'), drop_path_rate=cfg.get('drop_path_rate'))
    if 'num_classes' in cfg:
        _kw['num_classes'] = cfg['num_classes']
    if 'version' in cfg:
        _kw['version'] = cfg['version']
    if 'rafr_levels' in cfg:
        _kw['rafr_levels'] = cfg['rafr_levels']
    if 'dmsc_levels' in cfg:
        _kw['dmsc_levels'] = cfg['dmsc_levels']
    if 'dmsc_lambda' in cfg:
        _kw['dmsc_lambda'] = cfg['dmsc_lambda']
    model = build_model(arch, pretrained=None, **_kw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f'{ckpt_path} 有多余权重键: {unexpected[:5]}')
    real_missing = [k for k in missing if 'absolute_pos_embed' not in k]
    if real_missing:
        raise RuntimeError(f'{ckpt_path} 缺权重键: {real_missing[:5]}')
    model.to(device).eval()
    meta = dict(epoch=payload.get('epoch'), best_epoch=payload.get('best_epoch'), best_dice=payload.get('best_dice'), final_val_dice=payload.get('final_val_dice'), architecture=arch, norm=cfg['norm'], output_mode=cfg['output_mode'])
    return (model, meta)

@torch.no_grad()
def run_one(model, loader, device, output_mode='sigmoid_binary'):
    per_case = []
    for img, lbl in loader:
        img = img.to(device, non_blocking=True)
        out = model(img)
        pred = binarize_logits(out, output_mode).detach().cpu().numpy()
        gt = lbl.numpy()
        for i in range(pred.shape[0]):
            per_case.append(image_metrics(pred[i], gt[i], keys=C.TEST_METRICS))
    return (aggregate(per_case, C.TEST_METRICS), per_case)
W = 10

def fmt(v):
    return f'{'nan':>{W}}' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:>{W}.4f}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='run 目录或单个 .pth')
    ap.add_argument('--dataset', default='TN3K', choices=list(C.DATASETS))
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    ap.add_argument('--pre-root', default=C.PRE_ROOT)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default=None, help='汇总表输出路径(.txt)')
    ap.add_argument('--include-diagnostics', action='store_true', help='额外评 final/last 作稳定性诊断。🔴 默认关闭 —— 每多评一次 test 就多一次"看过 test",正式结果只认 Test@best,不要常态化开启')
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    device = torch.device(f'cuda:{a.gpu}')
    ckpts = resolve_ckpts(a.ckpt, include_diagnostics=a.include_diagnostics)
    _mode = 'best + final/last(稳定性诊断)' if a.include_diagnostics else 'best only'
    print(f'数据集 {a.dataset}/{a.split}   ckpt 数={len(ckpts)}   评测口径: {_mode}')
    print(f'口径: HD95/ASD 于 {C.IMG_SIZE}x{C.IMG_SIZE} pixel;退化惩罚={C.DEGENERATE_PENALTY:.2f};指标实现与 val 同一份')
    head = f'{'ckpt':<20}' + ''.join((f'{k:>{W}}' for k in C.TEST_METRICS)) + f'{'n':>8}{'skip':>7}{'ep':>6}'
    lines = [head, '-' * len(head)]
    all_res = []
    norm_used = set()
    for cp in ckpts:
        model, meta = load_model(cp, device)
        ds = build_dataset(a.dataset, a.split, pre_root=a.pre_root, augment=False, norm=meta['norm'])
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)
        norm_used.add(meta['norm'])
        print(f'  {os.path.basename(cp)}: arch={meta['architecture']} norm={meta['norm']} output_mode={meta['output_mode']} n={len(ds)}')
        agg, per_case = run_one(model, dl, device, meta['output_mode'])
        name = os.path.basename(cp)
        lines.append(f'{name:<20}' + ''.join((fmt(agg[k]) for k in C.TEST_METRICS)) + f'{agg['n']:>8}{agg['n_skipped']:>7}{str(meta.get('epoch')):>6}')
        all_res.append(dict(ckpt=name, path=cp, meta=meta, **{k: agg[k] for k in C.TEST_METRICS}, **{f'{k}_std': agg[f'{k}_std'] for k in C.TEST_METRICS}, n=agg['n'], n_skipped=agg['n_skipped'], per_case=per_case))
        del model
        torch.cuda.empty_cache()
    lines.append('-' * len(head))
    lines.append(f'（各列依次为 {', '.join(C.TEST_METRICS)}）')
    if len(norm_used) > 1:
        lines.append(f'⚠️ 本次多个 ckpt 的归一化不同 {norm_used},已各自按其自描述处理')
    if len(all_res) > 1:
        lines.append('')
        if a.split == 'test':
            lines.append('  ⚠️ 以下为稳定性诊断,**不作评分**;且已多看过 test —— 不要用于选模型')
        else:
            lines.append(f'  （稳定性诊断,不作评分;split={a.split},多看几次无妨)')
        b = all_res[0][C.BEST_METRIC]
        for r in all_res[1:]:
            lines.append(f'  best → {r['ckpt']} 变化: {r[C.BEST_METRIC] - b:+.4f}')
    out = '\n'.join(lines)
    print('\n' + out)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(out + '\n')
        with open(a.out.replace('.txt', '_percase.json'), 'w', encoding='utf-8') as f:
            json.dump(all_res, f, ensure_ascii=False)
        print(f'\n写出: {a.out}\n      {a.out.replace('.txt', '_percase.json')}')
if __name__ == '__main__':
    main()
