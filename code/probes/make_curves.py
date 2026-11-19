import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import configs as C
COLORS = ['#e74c3c', '#2e86de', '#27ae60', '#8e44ad', '#e67e22', '#16a085', '#c0392b']
ABLATION = [('Base', 'mybackbone/runs/mybackbone_v1_s42/20260924_071240'), ('+RAFR', 'mybackbone/runs/mybackbone_rafr4_s123/20260926_100045'), ('+DMSC', 'mybackbone/runs/mybackbone_dmsc_s42/20260925_022711'), ('+RAFR+DMSC (Ours)', 'mybackbone/runs/mybackbone_rafr4_dmsc_s123/20260926_135443')]

def ema(x, w):
    if w <= 1:
        return np.asarray(x, float)
    a = 2.0 / (w + 1.0)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def load(d, metric):
    f = glob_metrics(d)
    L = [json.loads(l) for l in open(f) if l.strip()]
    ep = np.array([x['epoch'] for x in L])
    y = np.array([x[metric] for x in L], dtype=float)
    return (ep, y)

def glob_metrics(d):
    import glob
    g = glob.glob(os.path.join(d, 'metrics.jsonl'))
    if not g:
        g = sorted(glob.glob(os.path.join(d, '2026*', 'metrics.jsonl')))
    if not g:
        raise FileNotFoundError(f'{d} 下没有 metrics.jsonl')
    return g[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='')
    ap.add_argument('--ablation', action='store_true', help='画消融四格')
    ap.add_argument('--metric', default='val_dice', choices=['val_dice', 'train_loss'])
    ap.add_argument('--smooth', type=int, default=9, help='EMA 窗口(0/1=不平滑)')
    ap.add_argument('--out', default='curves.png')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--logy', action='store_true', help='train_loss 用对数纵轴')
    ap.add_argument('--dual', action='store_true', help='双纵轴:train_loss(左) + val_dice(右)。**只对单个模型有意义**')
    a = ap.parse_args()
    ms = ABLATION if a.ablation else [(s.split(':')[0], s.split(':')[1]) for s in a.models.split()]
    if a.dual:
        assert len(ms) == 1, '--dual 只能配单个模型'
        name, d = ms[0]
        ep, loss = load(d, 'train_loss')
        _, vd = load(d, 'val_dice')
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        c1, c2 = ('#c0392b', '#2e86de')
        ax.plot(ep, loss, color=c1, lw=0.7, alpha=0.22)
        ax.plot(ep, ema(loss, a.smooth), color=c1, lw=1.9, label='Training loss')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Training loss', fontsize=11, color=c1)
        ax.tick_params(axis='y', colors=c1)
        if a.logy:
            ax.set_yscale('log')
        bx = ax.twinx()
        bx.plot(ep, vd, color=c2, lw=0.7, alpha=0.22)
        bx.plot(ep, ema(vd, a.smooth), color=c2, lw=1.9, label='Validation Dice')
        bx.set_ylabel('Validation Dice', fontsize=11, color=c2)
        bx.tick_params(axis='y', colors=c2)
        bi = int(np.argmax(vd))
        bx.scatter([ep[bi]], [vd[bi]], s=32, facecolor='white', edgecolor=c2, linewidths=1.6, zorder=5)
        bx.annotate(f'best {vd[bi]:.4f} @ep{ep[bi]}', (ep[bi], vd[bi]), textcoords='offset points', xytext=(6, -12), fontsize=8.5, color=c2)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = bx.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9.5, frameon=False, loc='center right')
        ax.grid(alpha=0.25, lw=0.5)
        for sp in list(ax.spines.values()) + list(bx.spines.values()):
            sp.set_linewidth(0.7)
        ax.set_title(f'{name}   (thin = raw, thick = EMA-{a.smooth})', fontsize=9.5, pad=6)
        plt.tight_layout()
        plt.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
        print(f'  {name}: train_loss {loss[0]:.4f} → {loss[-1]:.4f}   val_dice best {vd[bi]:.4f} @ep{ep[bi]}')
        print(f'  写出 {a.out}')
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for k, (name, d) in enumerate(ms):
        ep, y = load(d, a.metric)
        c = COLORS[k % len(COLORS)]
        ax.plot(ep, y, color=c, lw=0.8, alpha=0.28, zorder=2)
        ax.plot(ep, ema(y, a.smooth), color=c, lw=1.9, label=name, zorder=3)
        bi = int(np.argmax(y)) if a.metric == 'val_dice' else int(np.argmin(y))
        ax.scatter([ep[bi]], [y[bi]], s=26, facecolor='white', edgecolor=c, linewidths=1.6, zorder=4)
        print(f'  {name:<20} best {a.metric}={y[bi]:.4f} @ep{ep[bi]}   (n={len(ep)})')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Dice' if a.metric == 'val_dice' else 'Training loss', fontsize=11)
    if a.logy and a.metric == 'train_loss':
        ax.set_yscale('log')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9.5, frameon=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
    if a.smooth > 1:
        ax.set_title(f'(thin = raw, thick = EMA-{a.smooth})', fontsize=8.5, pad=6)
    plt.tight_layout()
    plt.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
    print(f'  写出 {a.out}')
if __name__ == '__main__':
    main()
