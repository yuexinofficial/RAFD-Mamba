import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
RUN = 'mybackbone/runs/mybackbone_rafr4_dmsc_s123/20260926_135443'
C_LAM = '#c0392b'
C_E = '#2e86de'

def load(run):
    rows = [json.loads(l) for l in open(os.path.join(run, 'metrics.jsonl')) if l.strip()]
    ep = np.array([r['epoch'] for r in rows])
    rafr = {k: dict(lam=[], E=[]) for k in ['1', '2', '3', '4']}
    dmsc = {k: dict(lam=[], E=[], eta1=[], eta2=[], eta3=[], alpha=[], beta=[]) for k in ['2', '3']}
    for r in rows:
        for k in rafr:
            rafr[k]['lam'].append(r['rafr'][k]['lambda_mean'])
            rafr[k]['E'].append(r['rafr'][k]['E'])
        for k in dmsc:
            d = r['dmsc'][k]
            dmsc[k]['lam'].append(d['lambda_mean'])
            dmsc[k]['E'].append(d['E'])
            for kk in ['eta1', 'eta2', 'eta3', 'alpha', 'beta']:
                dmsc[k][kk].append(d[kk])
    for k in rafr:
        for kk in rafr[k]:
            rafr[k][kk] = np.array(rafr[k][kk], float)
    for k in dmsc:
        for kk in dmsc[k]:
            dmsc[k][kk] = np.array(dmsc[k][kk], float)
    vd = np.array([r['val_dice'] for r in rows], float)
    return (ep, rafr, dmsc, vd)

def smooth(x, w=9):
    if w <= 1:
        return x
    a = 2.0 / (w + 1.0)
    out = np.empty(len(x), float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
    ax.tick_params(labelsize=8.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--out', default='figures/gate_telemetry.png')
    ap.add_argument('--dpi', type=int, default=400)
    a = ap.parse_args()
    ep, rafr, dmsc, vd = load(a.run)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))
    ax = axes[0]
    cols = ['#f1c40f', '#e67e22', '#c0392b', '#7b241c']
    lv = ['1', '2', '3', '4']
    dims = {'1': '192', '2': '384', '3': '768', '4': '768 (trunk)'}
    for c, k in zip(cols, lv):
        ax.plot(ep, smooth(rafr[k]['lam']), color=c, lw=1.7, label=f'skip{k} ($C$={dims[k]})')
        ax.plot(ep, rafr[k]['lam'], color=c, lw=0.5, alpha=0.3)
    ax.axhline(0.018, color='#555555', ls=':', lw=1.0)
    ax.text(60, 0.033, 'initial $\\lambda_0$=σ(−4)≈0.018', fontsize=7.6, color='#555555')
    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Reliability gate $\\lambda$', fontsize=9.5)
    ax.set_title('(a) RAFR: frequency gate opens during training', fontsize=9.8)
    ax.legend(fontsize=7.8, frameon=False, loc='upper left', ncol=1)
    ax.set_ylim(0, 0.62)
    style(ax)
    ax2 = ax.twinx()
    for c, k in zip(cols, lv):
        ax2.plot(ep, smooth(rafr[k]['E'], 15), color=c, lw=1.1, ls='--', alpha=0.85)
    ax2.set_ylabel('注入强度 $E=\\|\\lambda R\\|/\\|F\\|$', fontsize=9.0, color='#333333')
    ax2.tick_params(labelsize=8.0)
    ax2.set_ylim(0, 0.3)
    ax.text(0.97, 0.3, 'solid: $\\lambda$ / dashed: $E$', transform=ax.transAxes, fontsize=7.6, ha='right', color='#333333')
    ax = axes[1]
    cols_b = {'2': '#8e44ad', '3': '#2e86de'}
    res = {'2': '16×16×384', '3': '8×8×768'}
    for k in ['2', '3']:
        ax.plot(ep, smooth(dmsc[k]['lam']), color=cols_b[k], lw=1.8, label=f'stage{k} ({res[k]})')
        ax.plot(ep, dmsc[k]['lam'], color=cols_b[k], lw=0.5, alpha=0.3)
    ax.axhline(0.018, color='#555555', ls=':', lw=1.0)
    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Shuttable gate $\\lambda$', fontsize=9.5)
    ax.set_title('(b) DMSC: context gate opens during training', fontsize=9.8)
    ax.legend(fontsize=8.0, frameon=False, loc='upper left')
    ax.set_ylim(0, 0.16)
    style(ax)
    ax2 = ax.twinx()
    for k in ['2', '3']:
        ax2.plot(ep, smooth(dmsc[k]['E'], 15), color=cols_b[k], lw=1.1, ls='--', alpha=0.85)
    ax2.set_ylabel('注入强度 $E=\\|\\lambda W_o[O_s]\\|/\\|X\\|$', fontsize=9.0)
    ax2.tick_params(labelsize=8.0)
    ax2.set_ylim(0, 0.16)
    ax = axes[2]
    ax.plot(ep, smooth(dmsc['2']['eta2'], 21), color='#e67e22', lw=1.9, label='stage2 $\\eta_2$ dilated DWConv')
    ax.plot(ep, smooth(dmsc['3']['eta1'], 21), color='#27ae60', lw=1.9, label='stage3 $\\eta_1$ local DWConv')
    ax.plot(ep, smooth(dmsc['2']['beta'], 21), color='#8e44ad', lw=1.9, ls='--', label='stage2 $\\beta$ (Top-$k$ $N$/4)')
    ax.plot(ep, smooth(dmsc['3']['beta'], 21), color='#2e86de', lw=1.9, ls='--', label='stage3 $\\beta$ (Top-$k$ $N$/4)')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Gate weight', fontsize=9.5)
    ax.set_title('(c) Convergence of dynamic scale and sparse gates', fontsize=9.8)
    ax.legend(fontsize=7.6, frameon=False, loc='center right')
    style(ax)
    ax.axhline(0.5, color='#888888', ls=':', lw=0.9)
    ax.text(6, 0.45, 'equal weights for two Top-$k$ (0.5)', fontsize=7.4, color='#888888')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(a.out.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print('=== 末轮(ep200)门控状态 ===')
    for k in lv:
        print(f'  RAFR skip{k}: λ={rafr[k]['lam'][-1]:.4f}  E={rafr[k]['E'][-1]:.4f}')
    for k in ['2', '3']:
        d = dmsc[k]
        print(f'  DMSC stage{k}: λ={d['lam'][-1]:.4f}  E={d['E'][-1]:.4f}  η=({d['eta1'][-1]:.3f},{d['eta2'][-1]:.3f},{d['eta3'][-1]:.3f})  α={d['alpha'][-1]:.4f} β={d['beta'][-1]:.4f}')
    print('saved', a.out)
if __name__ == '__main__':
    main()
