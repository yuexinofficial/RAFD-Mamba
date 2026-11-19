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
DEFAULT_OPP = [('VM-UNet', 'output/vmunet_baseline_tn3k_fold0_s42/20260924_053648'), ('CFFormer', 'output/cfformer_tn3k_s42/20260925_003052'), ('CX-Tiny', 'output/convnextv2_unet_tn3k_s42/20260926_032645')]
OURS_COLOR = '#e74c3c'
OPP_COLOR = '#3498db'

def contour(mask, color, ax, lw=1.5, alpha=1.0):
    if mask.sum() == 0:
        return
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw, alpha=alpha, antialiased=True)

def dice_of(mask, gt):
    inter = np.logical_and(mask, gt).sum()
    s = mask.sum() + gt.sum()
    return 1.0 if s == 0 else 2.0 * inter / s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default='compare.png')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--opponents', default='', help='留空用默认三方')
    ap.add_argument('--must-beat', default='', help='额外要求:本模型 Dice **高于**这些对手(逗号分隔,名字须在 --opponents 里)。用于把「我们输给某个强基线」的案例排除掉。')
    ap.add_argument('--min-ours', type=float, default=0.9, help='只从「本模型 Dice ≥ 此值」的案例里挑(默认 0.90)。🔴 不加这个下限,按 Δ 排序会优先挑到「对手崩溃」的案例,画面上我们自己也只中等,且对手全输出空会被质疑挑图。设 0 = 不加限制(即旧行为)。')
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = torch.device(f'cuda:{a.gpu}')
    opps = DEFAULT_OPP if not a.opponents else [(s.split(':')[0], s.split(':')[1]) for s in a.opponents.split()]

    def percase(d):
        return np.array([c['dice'] for c in json.load(open(os.path.join(d, 'test_test_percase.json')))[0]['per_case']])
    ours = percase(a.ckpt)
    opp_d = {n: percase(p) for n, p in opps}
    opp_mean = np.mean([opp_d[n] for n, _ in opps], axis=0)
    score = ours - opp_mean
    pool = np.where(ours >= a.min_ours)[0] if a.min_ours > 0 else np.arange(len(ours))
    if a.must_beat:
        for nm in [x.strip() for x in a.must_beat.split(',') if x.strip()]:
            if nm not in opp_d:
                raise KeyError(f'--must-beat 里的 {nm!r} 不在 --opponents 里;可用 {list(opp_d)}')
            keep = ours > opp_d[nm]
            print(f'  要求 ours > {nm}: 剔掉 {int((~keep[pool]).sum())} 张')
            pool = pool[keep[pool]]
    bad = np.zeros(len(ours), bool)
    for n in opp_d:
        bad |= opp_d[n] < 0.3
    pool = pool[~bad[pool]]
    sel = pool[np.argsort(-score[pool])][:a.n]
    print(f'  候选池(ours≥{a.min_ours}' + (f' 且 beat {a.must_beat}' if a.must_beat else '') + f'): {len(pool)} 张;Δ>0 的 {int((score[pool] > 0).sum())} 张')
    names = sorted((f[:-4] for f in os.listdir(IMG) if f.endswith('.png')))
    cols = [('Ultrasound', None), ('Ground truth', None), ('Ours', a.ckpt)] + [(n, p) for n, p in opps]
    ncol = len(cols)
    models = {}
    for n, p in [('Ours', a.ckpt)] + opps:
        m, meta = load_model(os.path.join(p, 'best_model.pth'), dev)
        models[n] = (m, meta)
    ds = build_dataset('TN3K', 'test', pre_root=C.PRE_ROOT, augment=False, norm=models['Ours'][1]['norm'])
    fig, axes = plt.subplots(a.n, ncol, figsize=(1.55 * ncol, 2.05 * a.n))
    if a.n == 1:
        axes = axes[None, :]
    with torch.no_grad():
        for r, i in enumerate(sel):
            gray = np.array(Image.open(os.path.join(IMG, names[i] + '.png')).convert('L'))
            gt = np.array(Image.open(os.path.join(MSK, names[i] + '.png'))) > 0
            x, _ = ds[i]
            xb = x.unsqueeze(0).to(dev)
            preds = {}
            for n, (m, meta) in models.items():
                preds[n] = binarize_logits(m(xb), meta['output_mode'])[0].cpu().numpy()
            for c, (title, _) in enumerate(cols):
                ax = axes[r, c]
                if title == 'Ultrasound':
                    ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
                elif title == 'Ground truth':
                    ax.imshow(gt.astype(float), cmap='gray', vmin=0, vmax=1)
                else:
                    ax.imshow(preds[title].astype(float), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.6)
                if title == 'Ours':
                    ax.text(0.03, 0.97, f'{dice_of(preds['Ours'], gt):.3f}', transform=ax.transAxes, va='top', ha='left', fontsize=9, color='white', fontweight='bold', bbox=dict(fc=OURS_COLOR, ec='none', alpha=0.85, pad=1.5))
                elif title not in ('Ultrasound', 'Ground truth'):
                    ax.text(0.03, 0.97, f'{dice_of(preds[title], gt):.3f}', transform=ax.transAxes, va='top', ha='left', fontsize=9, color='white', fontweight='bold', bbox=dict(fc=OPP_COLOR, ec='none', alpha=0.85, pad=1.5))
                if r == 0:
                    ax.set_title({'Ultrasound': 'Image', 'Ground truth': 'GT', 'ConvNeXtV2-UNet': 'ConvNeXtV2'}.get(title, title), fontsize=16, pad=3)
            axes[r, 0].set_ylabel(f'#{r + 1}', fontsize=16)
            print(f'  #{r + 1} {names[i]}  ours={ours[i]:.4f}  opp={opp_mean[i]:.4f}  Δ={score[i]:+.4f}')
    plt.tight_layout(rect=[0, 0.0, 1, 1])
    plt.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
    print(f'写出 {a.out}')
if __name__ == '__main__':
    main()
