import os
import re
import sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'probes'))
import make_main_table as M
OUT = os.path.join(HERE, 'paper', 'tables')
BS = chr(92)

def clean_label(lab):
    return lab

def row_end():
    return BS + BS

def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    bi_has_prev = [False]
    for name, idx, tag, label, nrow in M.DATASETS:
        data = M.collect(idx, tag)
        order = sorted(data.items(), key=lambda kv: -kv[1]['dice'].mean())
        names = [n for n, _ in order if n != M.REF]
        pv = {m: [] for m in M.STAR_METRICS}
        for n in names:
            for m in M.STAR_METRICS:
                try:
                    _, p = stats.wilcoxon(data[M.REF][m], data[n][m])
                except ValueError:
                    p = float('nan')
                pv[m].append(p)
        adj = {m: M.bh(np.array(pv[m])) for m in M.STAR_METRICS}
        if bi_has_prev[0]:
            lines.append(BS + 'midrule')
        bi_has_prev[0] = True
        for r, (nm, vals) in enumerate(order):
            nm_tex = M.DISPLAY.get(nm, nm)
            cells = [M.fmt(vals[m].mean(), M.boot_half(vals[m]), m) for m in M.METRICS]
            if nm != M.REF:
                k = names.index(nm)
                for j, m in enumerate(M.METRICS):
                    if m in M.STAR_METRICS:
                        cells[j] = M.fmt(vals[m].mean(), M.boot_half(vals[m]), m, M.star(adj[m][k]))
            for j, m in enumerate(M.METRICS):
                allv = [dd[m].mean() for _, dd in order]
                target = min(allv) if m in M.LOWER_BETTER else max(allv)
                if abs(vals[m].mean() - target) < 1e-12:
                    cells[j] = BS + 'best{' + cells[j] + '}'
            cells_tex = ' & '.join(cells)
            if r == 0:
                lines.append(BS + 'multirow{%d}{*}{' % nrow + BS + 'makecell[c]{' + clean_label(label) + '}}')
                lines.append(' & ' + nm_tex + ' & ' + cells_tex + row_end())
            else:
                lines.append(' & ' + nm_tex + ' & ' + cells_tex + row_end())
        print(f'{name}: {len(order)} 行')
    lines.append(BS + 'bottomrule')
    p = os.path.join(OUT, 'main_body.tex')
    open(p, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('写出', p)
if __name__ == '__main__':
    main()
