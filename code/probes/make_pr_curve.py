import argparse, glob, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import configs as C
from data.thyroid_dataset import build_dataset
from test import load_model
NBIN = 100000

def collect_probs(ckpt, ds, dev, batch=16):
    model, meta = load_model(os.path.join(ckpt, 'best_model.pth'), dev)
    model.eval()
    mode = meta['output_mode']
    P, G = ([], [])
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            imgs, lbls = ([], [])
            for j in range(i, min(i + batch, len(ds))):
                x, y = ds[j]
                imgs.append(x)
                lbls.append(y)
            xb = torch.stack(imgs).to(dev)
            out = model(xb)
            if mode == 'softmax_binary':
                prob = torch.softmax(out, dim=1)[:, 1]
            else:
                prob = torch.sigmoid(out[:, 1])
            P.append(prob.flatten().cpu().numpy().astype(np.float32))
            G.append(torch.stack(lbls).flatten().numpy().astype(np.uint8))
    del model
    torch.cuda.empty_cache()
    return (np.concatenate(P), np.concatenate(G))

def pr_curve(prob, gt, nbin=NBIN):
    edges = np.linspace(0.0, 1.0, nbin + 1)
    tot, _ = np.histogram(prob, bins=edges)
    pos, _ = np.histogram(prob, bins=edges, weights=gt.astype(np.float64))
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum((tot - pos)[::-1])[::-1]
    P = gt.sum()
    R = tp / max(P, 1)
    Pr = tp / np.maximum(tp + fp, 1)
    R_asc, P_asc = (R[::-1], Pr[::-1])
    keep = np.concatenate([[True], np.diff(R_asc) > 0])
    R2 = np.concatenate([[0.0], R_asc[keep]])
    P2 = np.concatenate([[1.0], P_asc[keep]])
    ap_hist = float(np.sum(np.diff(R2) * P2[1:]))
    try:
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(gt, prob))
    except Exception:
        ap = ap_hist
    pr_curve.last_hist = ap_hist
    idx = np.searchsorted(edges[:-1], 0.5, side='right') - 1
    return (R_asc, P_asc, ap, (R[idx], Pr[idx]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='', help='留空用 --ckpt 单个模型')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default='pr.png')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--csv', default='', help='把曲线数据写成 csv')
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = torch.device(f'cuda:{a.gpu}')
    ms = []
    if a.models:
        for s in a.models.split():
            ms.append((s.split(':')[0], s.split(':')[1]))
    else:
        ms.append(('Ours', a.ckpt))
    ds = build_dataset('TN3K', 'test', pre_root=C.PRE_ROOT, augment=False, norm='minmax_255')
    COLORS = ['#e74c3c', '#2e86de', '#27ae60', '#8e44ad', '#e67e22', '#16a085']
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    rows = []
    for k, (name, ck) in enumerate(ms):
        prob, gt = collect_probs(ck, ds, dev)
        r, p, auc, at05 = pr_curve(prob, gt)
        print(f'  {name:<18} AP={auc:.4f}   @thr0.5: recall={at05[0]:.4f} precision={at05[1]:.4f}')
        rr = np.concatenate([[0.0], r])
        pp = np.concatenate([[1.0], p])
        ax.plot(rr, pp, color=COLORS[k % len(COLORS)], lw=1.8, label=f'{name}  {auc:.4f}')
        rows.append((name, r, p, auc))
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, columnspacing=1.6, handlelength=1.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
    axin = ax.inset_axes([0.16, 0.1, 0.56, 0.52])
    for k, (name, r, p_, auc) in enumerate(rows):
        axin.plot(r, p_, color=COLORS[k % len(COLORS)], lw=1.5)
    axin.set_xlim(0.8, 1.0)
    axin.set_ylim(0.6, 1.005)
    axin.grid(alpha=0.25, lw=0.4)
    axin.tick_params(labelsize=7.5)
    axin.set_title('zoom: recall > 0.8', fontsize=8.5, pad=3)
    for sp in axin.spines.values():
        sp.set_linewidth(0.6)
    plt.tight_layout()
    plt.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
    print(f'  写出 {a.out}')
    if a.csv:
        import csv
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model', 'recall', 'precision'])
            for name, r, p, _ in rows:
                keep = np.concatenate([[True], np.diff(r) > 1e-09])
                rr, pp = (r[keep], p[keep])
                idx = np.unique(np.linspace(0, len(rr) - 1, min(2000, len(rr))).astype(int))
                for rv, pv in zip(rr[idx], pp[idx]):
                    w.writerow([name, f'{rv:.6f}', f'{pv:.6f}'])
        print(f'  写出 {a.csv}')
if __name__ == '__main__':
    main()
