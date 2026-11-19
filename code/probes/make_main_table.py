import argparse
import glob
import json
import os
import re
import sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'probes'))
from paper_stats import MODELS, arr, percase
METRICS = ['dice', 'iou', 'hd95', 'asd', 'recall', 'precision']
DEC = {'dice': 4, 'iou': 4, 'recall': 4, 'precision': 4, 'hd95': 2, 'asd': 2}
STAR_METRICS = ['dice', 'iou', 'hd95']
DATASETS = [('TN3K', 0, 'test', 'TN3K ($n$=614)', 10), ('TNUI-2021', 1, 'test', 'TNUI ($n$=139)', 10), ('DDTI', 2, 'DDTI', 'DDTI ($n$=127)', 10)]
REF = 'Ours (RAFR+DMSC)'
DISPLAY = {'Ours (RAFR+DMSC)': 'RGF-Net', 'Swin-Unet+': 'Swin-Unet', 'nnU-Net+': 'nnU-Net', 'Attention U-Net+': 'Attention U-Net', 'EccoMamba*': 'EccoMamba', 'DAC-Net(500)': 'DAC-Net'}
LOWER_BETTER = {'hd95', 'asd'}

def resolve(pat):
    g = sorted(glob.glob(os.path.join(HERE, pat)))
    return g[0] if g else None

def boot_half(x, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return (hi - lo) / 2

def bh(pv):
    m = len(pv)
    order = np.argsort(pv)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        v = min(prev, pv[i] * m / (rank + 1))
        adj[i] = v
        prev = v
    return adj

def star(p):
    return '' if p != p else '$^{***}$' if p < 0.001 else '$^{**}$' if p < 0.01 else '$^{*}$' if p < 0.05 else ''

def fmt(v, hw, met, st=''):
    d = DEC[met]
    return f'{v:.{d}f}$\\pm${hw:.{d}f}{st}'

def collect(idx, tag):
    data = {}
    for nm, t3, tn, dd in MODELS:
        d = resolve([t3, tn, dd][idx])
        if not d:
            print('!! 缺:', nm, file=sys.stderr)
            continue
        pc = percase(d, tag)
        if pc is None:
            print('!! 无 percase:', nm, file=sys.stderr)
            continue
        data[nm] = {m: arr(pc, m) for m in METRICS}
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='回写 paper.tex')
    a = ap.parse_args()
    blocks = []
    for name, idx, tag, label, nrow in DATASETS:
        data = collect(idx, tag)
        if REF not in data:
            raise SystemExit(f'{name}: 缺少参考模型 {REF}')
        order = sorted(data.items(), key=lambda kv: -kv[1]['dice'].mean())
        names = [n for n, _ in order if n != REF]
        pv = {m: [] for m in STAR_METRICS}
        for n in names:
            for m in STAR_METRICS:
                try:
                    _, p = stats.wilcoxon(data[REF][m], data[n][m])
                except ValueError:
                    p = np.nan
                pv[m].append(p)
        adj = {m: bh(np.array(pv[m])) for m in STAR_METRICS}
        lines = []
        for r, (nm, vals) in enumerate(order):
            nm_tex = DISPLAY.get(nm, nm)
            cells = ['\\best{' + fmt(vals[m].mean(), boot_half(vals[m]), m) + '}' if nm == REF else fmt(vals[m].mean(), boot_half(vals[m]), m) for m in METRICS]
            if nm != REF:
                k = names.index(nm)
                for j, m in enumerate(METRICS):
                    if m in STAR_METRICS:
                        cells[j] = fmt(vals[m].mean(), boot_half(vals[m]), m, star(adj[m][k]))
            for j, m in enumerate(METRICS):
                allv = [dd[m].mean() for _, dd in order]
                target = min(allv) if m in LOWER_BETTER else max(allv)
                if abs(vals[m].mean() - target) < 1e-12 and 'best' not in cells[j]:
                    cells[j] = '\\best{' + cells[j] + '}'
            if r == 0:
                head = f'\\multirow{{{nrow}}}{{*}}{{\\makecell{{{label}}}}}\n & {nm_tex} & ' + ' & '.join(cells) + '\\\\'
            else:
                head = f' & {nm_tex} & ' + ' & '.join(cells) + '\\\\'
            lines.append(head)
        blocks.append('\n'.join(lines))
        print(f'===== {name} =====')
        print('\n'.join(lines))
        print()
    if a.write:
        tex = os.path.join(HERE, 'paper', 'paper.tex')
        s = open(tex, encoding='utf-8').read()
        new_body = '\n\\midrule\n'.join(blocks)
        pat = re.compile('(\\\\begin\\{tabular\\}\\{p\\{22pt\\}p\\{66pt\\}cccccc\\}\\n\\\\toprule\\n.*?\\\\midrule\\n)(.*?)(\\n\\\\bottomrule)', re.S)
        m = pat.search(s)
        if not m:
            raise SystemExit('未在 paper.tex 中定位到表 1 的 tabular 主体')
        s = s[:m.start(2)] + new_body + s[m.end(2):]
        s = s.replace('\\item[] $^{*}$EccoMamba 采用本文统一训练协议(控制臂)。表中方法均在同一数据划分与同一评测口径下重训。', '\\item[] 每格为“均值$\\pm$半宽”,半宽为对测试图 bootstrap 重采样($B$=10000)的 95\\% 置信区间半宽;上标星号为该模型与 RGF-Net 的逐图配对 Wilcoxon 符号秩检验(BH-FDR 校正)结果,$^{***}P<0.001$、$^{**}P<0.01$、$^{*}P<0.05$,未标注表示差异不显著(仅对 Dice/IoU/HD95 标星);加粗为该指标最优值。表中方法均在同一数据划分与同一评测口径下重训。')
        open(tex, 'w', encoding='utf-8').write(s)
        print('已回写', tex)
if __name__ == '__main__':
    main()
