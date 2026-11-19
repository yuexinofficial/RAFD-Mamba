import argparse, os, sys
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n-img', type=int, default=150, help='用多少张测试图')
    ap.add_argument('--per-class', type=int, default=50, help='每张图前景/背景各采多少个位置')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default='tsne.png')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.cuda.set_device(a.gpu)
    dev = torch.device(f'cuda:{a.gpu}')
    model, meta = load_model(os.path.join(a.ckpt, 'best_model.pth'), dev)
    model.eval()
    bb = model.backbone
    ds = build_dataset('TN3K', 'test', pre_root=C.PRE_ROOT, augment=False, norm=meta['norm'])
    n_img = min(a.n_img, len(ds))
    feats, labels = ([], [])
    with torch.no_grad():
        for i in range(n_img):
            img, lbl = ds[i]
            x, skip = bb.forward_features(img.unsqueeze(0).to(dev))
            up = bb.forward_features_up(x, skip)
            f = up[0].reshape(-1, up.shape[-1]).cpu().numpy()
            H, W = (up.shape[1], up.shape[2])
            gt = (lbl.numpy() > 0).astype(np.float32)
            g = torch.from_numpy(gt)[None, None]
            g = torch.nn.functional.adaptive_max_pool2d(g, (H, W))[0, 0].numpy().reshape(-1) > 0
            fg_idx, bg_idx = (np.nonzero(g)[0], np.nonzero(~g)[0])
            if len(fg_idx) == 0 or len(bg_idx) == 0:
                continue
            k = a.per_class
            sel_fg = rng.choice(fg_idx, size=min(k, len(fg_idx)), replace=False)
            sel_bg = rng.choice(bg_idx, size=min(k, len(bg_idx)), replace=False)
            feats.append(f[sel_fg])
            labels.append(np.ones(len(sel_fg), np.int8))
            feats.append(f[sel_bg])
            labels.append(np.zeros(len(sel_bg), np.int8))
    X = np.concatenate(feats)
    y = np.concatenate(labels)
    print(f'  采样点数={len(X)} (前景 {int(y.sum())} / 背景 {int((1 - y).sum())})  特征维度={X.shape[1]}')
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    Z = TSNE(n_components=2, method='exact', perplexity=30, init='pca', learning_rate='auto', random_state=a.seed).fit_transform(X)
    sil = silhouette_score(X, y, sample_size=min(5000, len(X)), random_state=a.seed)
    print(f'  t-SNE 完成;silhouette score(原始特征空间) = {sil:.4f}')
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for lab, color, name in ((0, '#2e86de', 'Background'), (1, '#e74c3c', 'Foreground')):
        m = y == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=3.0, c=color, alpha=0.55, linewidths=0, label=name, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('t-SNE dim 1', fontsize=9)
    ax.set_ylabel('t-SNE dim 2', fontsize=9)
    ax.legend(markerscale=4, fontsize=10, frameon=False, loc='best')
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    plt.tight_layout()
    plt.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
    print(f'  写出 {a.out}')
if __name__ == '__main__':
    main()
