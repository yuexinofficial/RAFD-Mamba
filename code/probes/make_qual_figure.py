import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import configs as C
from data.thyroid_dataset import build_dataset
from metrics import binarize_logits
from test import load_model
IMG = '/public/home/lm12345/CYX/datasets/preprocessed_thyroid/TN3K/test/images'
MSK = '/public/home/lm12345/CYX/datasets/preprocessed_thyroid/TN3K/test/masks'

def contour(mask, color, ax, lw=1.5):
    if mask.sum() == 0:
        return
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw, antialiased=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--mode', default='best', choices=['best', 'balanced', 'given'])
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--idx', default='', help='--mode given 时的下标,逗号分隔')
    ap.add_argument('--gpu', type=int, default=7)
    ap.add_argument('--out', default='qualitative.png')
    ap.add_argument('--dpi', type=int, default=300)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = torch.device(f'cuda:{a.gpu}')
    pj = os.path.join(a.ckpt, 'test_test_percase.json')
    dice = np.array([c['dice'] for c in json.load(open(pj))[0]['per_case']])
    order = np.argsort(-dice)
    if a.mode == 'best':
        sel = order[:a.n]
    elif a.mode == 'balanced':
        k = max(1, a.n // 3)
        med = len(dice) // 2
        sel = np.concatenate([order[:k], order[med - k // 2:med - k // 2 + k], order[-k:]])
    else:
        sel = np.array([int(x) for x in a.idx.split(',') if x.strip()])
    names = sorted((f[:-4] for f in os.listdir(IMG) if f.endswith('.png')))
    model, meta = load_model(os.path.join(a.ckpt, 'best_model.pth'), dev)
    ds = build_dataset('TN3K', 'test', pre_root=C.PRE_ROOT, augment=False, norm=meta['norm'])
    ncol = 4
    fig, axes = plt.subplots(len(sel), ncol, figsize=(3.0 * ncol, 3.0 * len(sel)))
    if len(sel) == 1:
        axes = axes[None, :]
    with torch.no_grad():
        for r, i in enumerate(sel):
            name = names[i]
            gray = np.array(Image.open(os.path.join(IMG, name + '.png')).convert('L'))
            gt = np.array(Image.open(os.path.join(MSK, name + '.png'))) > 0
            x, _ = ds[i]
            out = model(x.unsqueeze(0).to(dev))
            pred = binarize_logits(out, meta['output_mode'])[0].cpu().numpy()
            for c in range(ncol):
                axes[r, c].imshow(gray, cmap='gray', vmin=0, vmax=255)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                for sp in axes[r, c].spines.values():
                    sp.set_linewidth(0.6)
            contour(gt, '#2ecc71', axes[r, 1])
            contour(pred, '#e74c3c', axes[r, 2])
            contour(gt, '#2ecc71', axes[r, 3])
            contour(pred, '#e74c3c', axes[r, 3])
            axes[r, 0].set_ylabel(f'#{r + 1}  Dice={dice[i]:.4f}', fontsize=10, labelpad=6)
            if r == 0:
                for c, t in enumerate(['Ultrasound image', 'Ground truth', 'Ours', 'Overlay']):
                    axes[0, c].set_title(t, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
    print(f'写出 {a.out}')
    print(' 选中:', ', '.join((f'{names[i]}({dice[i]:.4f})' for i in sel)))
if __name__ == '__main__':
    main()
