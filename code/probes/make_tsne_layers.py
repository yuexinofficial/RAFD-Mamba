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
    ap.add_argument('--n-img', type=int, default=25, help='exact t-SNE 是 O(n²),别开太大')
    ap.add_argument('--per-class', type=int, default=50)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default='tsne_layers.png')
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
    STAGES = [('Encoder  /4  96ch', lambda x, sk: sk[0]), ('Encoder bottleneck  /32  768ch', lambda x, sk: x), ('Decoder  /4  96ch', lambda x, sk: bb.forward_features_up(x, sk))]
    data = {i: [[], []] for i in range(len(STAGES))}
    with torch.no_grad():
        for i in range(min(a.n_img, len(ds))):
            img, lbl = ds[i]
            x, sk = bb.forward_features(img.unsqueeze(0).to(dev))
            for si, (_, fn) in enumerate(STAGES):
                f = fn(x, sk)
                H, W = (f.shape[1], f.shape[2])
                v = f[0].reshape(-1, f.shape[-1]).cpu().numpy()
                g = torch.nn.functional.adaptive_max_pool2d(torch.from_numpy((lbl.numpy() > 0).astype('float32'))[None, None], (H, W))[0, 0].numpy().reshape(-1) > 0
                fg, bg = (np.nonzero(g)[0], np.nonzero(~g)[0])
                if len(fg) == 0 or len(bg) == 0:
                    continue
                k = a.per_class
                sf = rng.choice(fg, min(k, len(fg)), replace=False)
                sb = rng.choice(bg, min(k, len(bg)), replace=False)
                data[si][0] += [v[sf], v[sb]]
                data[si][1] += [np.ones(len(sf)), np.zeros(len(sb))]
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    fig, axes = plt.subplots(1, len(STAGES), figsize=(4.6 * len(STAGES), 4.4))
    for si, (name, _) in enumerate(STAGES):
        X = np.concatenate(data[si][0])
        y = np.concatenate(data[si][1])
        sil = silhouette_score(X, y, sample_size=min(4000, len(X)), random_state=a.seed)
        Z = TSNE(n_components=2, method='exact', perplexity=30, init='pca', learning_rate='auto', random_state=a.seed).fit_transform(X)
        ax = axes[si]
        for lab, col, nm in ((0, '#2e86de', 'Background'), (1, '#e74c3c', 'Foreground')):
            m = y == lab
            ax.scatter(Z[m, 0], Z[m, 1], s=2.6, c=col, alpha=0.55, linewidths=0, label=nm, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{name}\nsilhouette = {sil:.3f}', fontsize=10.5)
        if si == 0:
            ax.legend(markerscale=4.5, fontsize=9.5, frameon=False, loc='best')
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        print(f'  {name:<34} n={len(X):<6} silhouette={sil:.4f}')
    plt.tight_layout()
    plt.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
    print(f'  写出 {a.out}')
if __name__ == '__main__':
    main()
