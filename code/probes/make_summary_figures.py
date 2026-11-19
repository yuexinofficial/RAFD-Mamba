import glob
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
MODELS = [('RGF-Net', 'mybackbone/runs/mybackbone_rafr4_dmsc_s123/20260926_135443', 'output/mybackbone_rafr4_dmsc_tnui_s42/2026*', 'mybackbone/runs/mybackbone_rafr4_dmsc_s123/20260926_135443', 56.18), ('EccoMamba', 'output/eccomamba_userproto_s42/20260924_124907', 'output/eccomamba_userproto_tnui_s42/2026*', 'output/eccomamba_userproto_s42/20260924_124907', 63.47), ('ConvNeXtV2-UNet', 'output/convnextv2_unet_tn3k_s42/20260926_032645', 'output/convnextv2_unet_tnui_s42/2026*', 'output/convnextv2_unet_tn3k_s42/20260926_032645', 34.86), ('CFFormer', 'output/cfformer_tn3k_s42/20260925_003052', 'output/cfformer_tnui_s42/2026*', 'output/cfformer_tn3k_s42/20260925_003052', 99.57), ('Swin-Unet', 'output/swinunet_tn3k_s42/2026*', 'output/swinunet_tnui_s42/2026*', 'output/swinunet_tn3k_s42/2026*', 41.39), ('nnU-Net', 'output/nnunet_plain_tn3k_s42/2026*', 'output/nnunet_plain_tnui_s42/2026*', 'output/nnunet_plain_tn3k_s42/2026*', 9.0), ('VM-UNet', 'output/vmunet_baseline_tn3k_fold0_s42/20260924_053648', 'output/vmunet_tnui_s42/2026*', 'output/vmunet_baseline_tn3k_fold0_s42/20260924_053648', 27.43), ('U-Net', 'output/pytorch_unet_tn3k_fold0_s42/20260924_061243', 'output/pytorch_unet_tnui_s42/2026*', 'output/pytorch_unet_tn3k_fold0_s42/20260924_061243', 31.04), ('DAC-Net', 'output/dacnet500_tn3k_s42/20260926_084434', 'output/dacnet_tnui_s42/2026*', 'output/dacnet500_tn3k_s42/20260926_084434', 1.28), ('Attention U-Net', 'output/att_unet_tn3k_s42/2026*', 'output/att_unet_tnui_s42/2026*', 'output/att_unet_tn3k_s42/2026*', 2.88)]
TAG = ['test', 'test', 'DDTI']

def load(pat, tag):
    g = sorted(glob.glob(os.path.join(HERE, pat)))
    if not g:
        return None
    f = glob.glob(os.path.join(g[0], f'test_{tag}_percase.json'))
    if not f:
        return None
    pc = json.load(open(f[0]))[0]['per_case']
    out = {}
    for k in ['dice', 'iou', 'hd95', 'asd', 'recall', 'precision']:
        v = np.array([c[k] for c in pc if c.get(k) is not None], float)
        out[k] = v.mean()
    return out

def main():
    data = {}
    for nm, t3, tn, dd, prm in MODELS:
        data[nm] = {'params': prm}
        for i, pat in enumerate([t3, tn, dd]):
            r = load(pat, TAG[i])
            if r:
                data[nm][f'DS{i}'] = r
    missing = [n for n, v in data.items() if len(v) < 4]
    if missing:
        print('!! 缺数据:', missing)
    names = [n for n in data if 'DS0' in data[n]]
    names.sort(key=lambda n: -data[n]['DS0']['dice'])
    dice = [data[n]['DS0']['dice'] for n in names]
    hd95 = [data[n]['DS0']['hd95'] for n in names]
    x = np.arange(len(names))
    colors = ['#c0392b' if n == 'RGF-Net' else '#7f8c8d' for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.6))
    for ax, vals, lab, better in [(axes[0], dice, 'Dice $\\uparrow$', 'max'), (axes[1], hd95, 'HD95 (pixel) $\\downarrow$', 'min')]:
        ax.bar(x, vals, color=colors, width=0.68, edgecolor='white', linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (0.004 if better == 'max' else 0.5), f'{v:.4f}' if better == 'max' else f'{v:.2f}', ha='center', va='bottom', fontsize=7.0)
        best = max(vals) if better == 'max' else min(vals)
        ax.axhline(best, color='#c0392b', ls=':', lw=0.9, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=32, ha='right', fontsize=8)
        ax.set_ylabel(lab, fontsize=9.5)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)
        ax.tick_params(labelsize=8)
    axes[0].set_ylim(min(dice) - 0.03, max(dice) + 0.022)
    axes[1].set_ylim(0, max(hd95) * 1.16)
    axes[0].set_title('(a) TN3K 测试集 Dice(按 Dice 降序)', fontsize=9.8)
    axes[1].set_title('(b) TN3K 测试集 HD95', fontsize=9.8)
    plt.tight_layout()
    out_a = os.path.join(HERE, 'figures', 'summary_TN3K_bars.png')
    fig.savefig(out_a, dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(out_a.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print('saved', out_a)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.0))
    names_b = [n for n in data if n != 'RGF-Net' and all((f'DS{k}' in data[n] for k in range(3)))]
    names_b.sort(key=lambda n: -data[n]['DS0']['dice'])
    ref = data['RGF-Net']
    tg = {'RGF-Net': '#c0392b', 'EccoMamba': '#e67e22', 'ConvNeXtV2-UNet': '#8e44ad', 'CFFormer': '#2980b9', 'Swin-Unet': '#16a085', 'nnU-Net': '#7f8c8d', 'VM-UNet': '#34495e', 'U-Net': '#95a5a6', 'DAC-Net': '#bdc3c7', 'Attention U-Net': '#d5dbdb'}
    y = np.arange(len(names_b))
    h = 0.26
    ds_lab = ['TN3K', 'TNUI-2021', 'DDTI(外部)']
    for k in range(3):
        for ax, met, better in [(axes[0], 'dice', 1), (axes[1], 'hd95', -1)]:
            d = [better * (data[n][f'DS{k}'][met] - ref[f'DS{k}'][met]) for n in names_b]
            ax.barh(y + (k - 1) * h, d, height=h, color=plt.cm.viridis(k / 2.6), edgecolor='white', linewidth=0.4, label=ds_lab[k] if met == 'dice' else None)
        axes[1].set_xlabel('相对 RGF-Net 的 HD95 改善 (pixel,$\\downarrow$ 为正=更好)', fontsize=8.8)
    axes[0].set_xlabel('相对 RGF-Net 的 Dice 差值($\\uparrow$ 为正=更好)', fontsize=8.8)
    for ax, ttl in zip(axes, ['(a) 三套测试集上的 Dice 差值', '(b) 三套测试集上的 HD95 改善']):
        ax.set_yticks(y)
        ax.set_yticklabels(names_b, fontsize=8)
        ax.axvline(0, color='#c0392b', lw=1.1)
        ax.grid(axis='x', alpha=0.25, lw=0.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)
        ax.tick_params(labelsize=8)
        ax.set_title(ttl, fontsize=9.6)
        ax.invert_yaxis()
    axes[0].legend(fontsize=7.4, frameon=False, loc='lower right')
    axes[1].text(0.02, 0.05, '条形越长表示相对 RGF-Net 的改善越大', transform=axes[1].transAxes, fontsize=7.2, ha='left', color='#444444')
    plt.tight_layout()
    out_b = os.path.join(HERE, 'figures', 'summary_delta_gain.png')
    fig.savefig(out_b, dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(out_b.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print('saved', out_b)
    print('\n=== 取数核对 ===')
    for n in names:
        d = data[n]
        print(f'{n:17} TN3K {d['DS0']['dice']:.4f}/{d['DS0']['hd95']:.2f}  TNUI {d['DS1']['dice']:.4f}/{d['DS1']['hd95']:.2f}  DDTI {d['DS2']['dice']:.4f}/{d['DS2']['hd95']:.2f}')
if __name__ == '__main__':
    main()
