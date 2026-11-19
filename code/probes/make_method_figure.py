import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'
C_ENC, C_DEC = ('#dbe7f5', '#e2f0e2')
C_A, C_B = ('#fbe3cf', '#e6dcf5')
EC_A, EC_B = ('#b9651a', '#5b3f8c')

def box(ax, x, y, w, h, text, fc, ec='#333333', fs=8.0, lw=1.0, r=0.03, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.004,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=z + 1, linespacing=1.4)
    return (x, y, w, h)

def arrow(ax, xy1, xy2, color='#333333', lw=1.1, rad=0.0, ls='-', z=4, style='-|>'):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=10, lw=lw, color=color, zorder=z, linestyle=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=0.5, shrinkB=0.5))

def circle(ax, x, y, txt, fc, ec, r=0.16, fs=7.4):
    ax.add_patch(mp.Circle((x, y), r, fc=fc, ec=ec, lw=1.0, zorder=6))
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, color=ec, zorder=7)

def blank(ax, xlim=(0, 10), ylim=(0, 6.2)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')

def panel_a(ax):
    blank(ax)
    ax.text(0.05, 6.1, '(a) 整体框架', fontsize=10.5, va='top')
    ew, eh = (1.3, 0.8)
    ex, gap = (1.1, 1.1)
    enc = [('Stage 1\n96×64×64', 4.95), ('Stage 2\n192×32×32', 3.95), ('Stage 3\n384×16×16', 2.95), ('Stage 4\n768×8×8', 1.95)]
    box(ax, ex, 5.75, ew, 0.42, 'Patch Embed', C_ENC, fs=7.4)
    for lb, y in enc:
        box(ax, ex, y, ew, eh, lb, C_ENC, fs=7.2)
    box(ax, ex, 0.55, ew, 0.5, '输入\n256×256', '#ffffff', fs=7.4)
    arrow(ax, (ex + ew / 2, 1.0), (ex + ew / 2, 1.95))
    for i in range(len(enc) - 1):
        arrow(ax, (ex + ew / 2, enc[i][1]), (ex + ew / 2, enc[i + 1][1] + eh))
    arrow(ax, (ex + ew / 2, 5.75), (ex + ew / 2, 4.95 + eh))
    bx, bw = (3.0, 3.9)
    box(ax, bx, 0.55, bw, 0.5, 'VSS Block 瓶颈层 (8×8×768)', '#ffffff', fs=8.0)
    arrow(ax, (bx + 0.3, 0.8), (bx - 0.12, 0.8))
    arrow(ax, (ex + ew / 2, 1.95), (ex + ew / 2, 0.8), rad=0.0)
    arrow(ax, (ex + ew / 2, 0.8), (bx, 0.8))
    dx = ex + ew + gap
    dec = [('Up4\n96×64×64', 1.95), ('Up3\n192×32×32', 2.95), ('Up2\n384×16×16', 3.95), ('Up1\n768×8×8', 4.95)]
    for lb, y in dec:
        box(ax, dx, y, ew, eh, lb, C_DEC, fs=7.2)
    box(ax, dx, 5.75, ew, 0.42, 'Seg Head', C_DEC, fs=7.4)
    box(ax, dx, 0.55, ew, 0.5, '预测掩膜', '#ffffff', fs=7.4, z=5)
    arrow(ax, (bx + bw, 0.8), (dx + ew / 2, 0.8))
    arrow(ax, (dx + ew / 2, 0.8), (dx + ew / 2, 1.95))
    for i in range(len(dec) - 1):
        arrow(ax, (dx + ew / 2, dec[i][1] + eh), (dx + ew / 2, dec[i + 1][1]))
    arrow(ax, (dx + ew / 2, 4.95 + eh), (dx + ew / 2, 5.75))
    pairs = [(1.95 + eh / 2, 4.95 + eh / 2), (2.95 + eh / 2, 3.95 + eh / 2), (3.95 + eh / 2, 2.95 + eh / 2), (4.95 + eh / 2, 1.95 + eh / 2)]
    for i, (ys, yd) in enumerate(pairs):
        arrow(ax, (ex + ew, ys), (dx, yd), color='#c0392b', lw=1.0, ls='--')
        if i < 3:
            xm, ym = ((ex + ew + dx) / 2, (ys + yd) / 2)
            off = {0: 0.42, 1: -0.42, 2: 0.86}[i]
            circle(ax, xm, ym + off, 'A', C_A, EC_A)
    ax.text(5.0, 5.34, 'A = RAFR(频率残差),挂 skip 1/2/3 + 主干输出 4', fontsize=7.4, color=EC_A, ha='center', va='center')
    for lb, y in [enc[1], enc[2]]:
        ax.add_patch(mp.Circle((ex + ew / 2, y - 0.02), 0.16, fc=C_B, ec=EC_B, lw=1.0, zorder=6))
        ax.text(ex + ew / 2, y - 0.02, 'B', ha='center', va='center', fontsize=7.4, color=EC_B, zorder=7)
    ax.annotate('DMSC(模块 B)\n挂 stage 2/3 入口主干', xy=(ex + ew / 2 + 0.17, 3.93), xytext=(0.1, 2.55), fontsize=7.4, color=EC_B, ha='left', arrowprops=dict(arrowstyle='-', color=EC_B, lw=0.8))
    ax.add_patch(mp.Rectangle((0.05, 0.02), 0.22, 0.16, fc=C_A, ec=EC_A))
    ax.text(0.33, 0.1, 'A:频率残差(可关闭)', fontsize=7.4, va='center')
    ax.add_patch(mp.Rectangle((2.55, 0.02), 0.22, 0.16, fc=C_B, ec=EC_B))
    ax.text(2.83, 0.1, 'B:多尺度稀疏上下文(可关闭)', fontsize=7.4, va='center')
    ax.plot([5.55, 5.85], [0.1, 0.1], ls='--', color='#c0392b', lw=1.0)
    ax.text(5.95, 0.1, 'skip 连接', fontsize=7.4, va='center')

def panel_b(ax):
    blank(ax)
    ax.text(0.05, 6.1, '(b) 模块 A:可靠性自适应频率残差', fontsize=10.5, va='top')
    box(ax, 0.35, 4.85, 1.75, 0.62, '输入特征 F\n(H×W×C)', '#ffffff', fs=8.0)
    box(ax, 3.05, 4.85, 2.0, 0.62, 'Haar 小波分解\n(丢弃低频 LL)', '#ffffff', fs=8.0)
    arrow(ax, (2.1, 5.16), (3.05, 5.16))
    box(ax, 3.05, 3.75, 2.0, 0.62, '高频先验 $P$ = [$LH, HL, HH$]', '#ffffff', fs=8.0)
    arrow(ax, (4.05, 4.85), (4.05, 4.37))
    box(ax, 5.75, 3.35, 3.05, 1.62, '可学习频率残差 R(F, P)\n\n1×1 分组卷积 (g=3,子带独立)\n→ 3×3 深度可分离卷积 → GELU\n→ 逆 Haar 重构 → 1×1 投影\n(输出投影零初始化)', '#ffffff', fs=7.6)
    arrow(ax, (5.05, 4.06), (5.75, 4.06))
    box(ax, 2.85, 1.3, 3.55, 1.0, '可靠性门 $\\lambda = \\sigma(\\,W\\cdot[F \\,\\|\\, R] + b\\,)$\n逐通道、逐位置;b 初值 −4\n⇒ λ₀ ≈ 0.018', '#fdf0e2', fs=7.8)
    arrow(ax, (1.22, 4.85), (1.22, 1.8), color='#777777', lw=0.9)
    arrow(ax, (1.22, 1.8), (2.85, 1.8), color='#777777', lw=0.9)
    arrow(ax, (7.27, 3.35), (7.27, 1.8), color='#777777', lw=0.9)
    arrow(ax, (7.27, 1.8), (6.4, 1.8), color='#777777', lw=0.9)
    box(ax, 7.0, 1.3, 1.8, 0.62, '$\\lambda \\cdot R$', '#fdf0e2', fs=9.5)
    arrow(ax, (6.4, 1.61), (7.0, 1.61))
    box(ax, 4.2, 0.15, 3.1, 0.62, '$F_{out} = F + \\lambda \\cdot R$', '#e8f4ea', fs=9.5)
    arrow(ax, (7.9, 1.3), (7.9, 0.46), lw=1.1)
    arrow(ax, (7.9, 0.46), (7.3, 0.46), lw=1.1)
    arrow(ax, (0.35, 5.0), (0.12, 5.0), lw=1.1)
    arrow(ax, (0.12, 5.0), (0.12, 0.46), lw=1.1)
    arrow(ax, (0.12, 0.46), (4.2, 0.46), lw=1.1)
    ax.text(0.2, 3.1, '恒等路径 ($F$)', fontsize=7.4, rotation=90, va='center')
    ax.text(4.25, 2.72, '$\\lambda \to 0 \\;\\Rightarrow\\; F_{out} \\equiv F$   (可证明退回骨干网络)', fontsize=8.4, color='#1e6b34')

def panel_c(ax):
    blank(ax)
    ax.text(0.05, 6.1, '(c) 模块 B:动态多尺度稀疏上下文', fontsize=10.5, va='top')
    box(ax, 0.35, 2.85, 1.15, 0.66, '输入 X\n(H×W×C)', '#ffffff', fs=8.0)
    brs = [('DWConv 3×3\n局部纹理', 4.55), ('空洞 DWConv 3×3 (d=2)\n较大感受野', 3.55), ('深度可分离可变形卷积\n不规则形态', 2.55)]
    for txt, y in brs:
        box(ax, 2.3, y, 2.5, 0.78, txt, '#ffffff', fs=7.2)
        arrow(ax, (1.5, 3.18), (2.3, y + 0.39), lw=0.9)
    box(ax, 2.3, 1.3, 2.5, 0.66, '[$\\eta_1,\\eta_2,\\eta_3$] = Softmax(MLP(GAP(X)))\n动态尺度门(逐样本)', '#ece3f8', fs=7.2)
    arrow(ax, (1.5, 2.85), (1.5, 1.63), lw=0.9)
    arrow(ax, (1.5, 1.63), (2.3, 1.63), lw=0.9)
    for _, y in brs:
        arrow(ax, (4.8, 1.96), (4.8, y), lw=0.8, color=EC_B)
    box(ax, 5.25, 3.15, 1.7, 0.66, '$F_m = \\Sigma\\, \\eta_i F_i$', '#ece3f8', fs=9.0)
    for _, y in brs:
        arrow(ax, (4.8, y), (5.25, 3.48), lw=0.9)
    box(ax, 7.3, 3.55, 2.45, 0.86, '稀疏注意力:Q←X, K,V←F_m\n双粒度 Top-k (N/2、N/4)\n→ $O_1$, $O_2$', '#ffffff', fs=7.2)
    arrow(ax, (6.95, 3.48), (7.3, 3.98), lw=1.0)
    box(ax, 7.3, 2.7, 2.45, 0.6, '[α,β] = Softmax(MLP(GAP(X)))\n动态稀疏门(逐样本)', '#ece3f8', fs=7.2)
    box(ax, 7.3, 1.85, 2.45, 0.6, '$O_s = \\alpha O_1 + \\beta O_2$', '#ffffff', fs=9.0)
    arrow(ax, (8.52, 3.55), (8.52, 3.3), lw=0.9)
    arrow(ax, (8.52, 2.7), (8.52, 2.45), lw=0.9)
    box(ax, 7.3, 0.85, 2.45, 0.68, '$Y = X + \\lambda \\cdot W_o[O_s]$', '#e8f4ea', fs=9.5)
    arrow(ax, (8.52, 1.85), (8.52, 1.53), lw=1.0)
    box(ax, 4.55, 0.85, 2.45, 0.68, '$\\lambda = \\sigma$( gate([$X \\,\\|\\, W_o[O_s]$]) + b )\n$W_o$ 零初始化,b 初值 −4', '#fdf0e2', fs=7.2)
    arrow(ax, (7.0, 1.19), (7.3, 1.19), lw=1.0)
    box(ax, 0.35, 0.85, 3.6, 0.68, 'X 直连(恒等路径)   $\\lambda \\to 0 \\Rightarrow Y \\equiv X$', '#ffffff', fs=7.8)
    arrow(ax, (1.5, 2.85), (1.5, 1.53), lw=1.1)
    arrow(ax, (1.5, 1.53), (4.55, 1.53), lw=1.1)
    arrow(ax, (8.52, 0.85), (8.52, 0.3), lw=1.0)
    ax.text(8.62, 0.16, '输出 Y', fontsize=8.0, va='center')
    ax.text(4.6, 0.42, 'λ 与模块 A 同构:模块初始逐位恒等,由网络决定是否启用', fontsize=7.4, color='#1e6b34')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/method_overview.png')
    ap.add_argument('--dpi', type=int, default=400)
    a = ap.parse_args()
    fig = plt.figure(figsize=(13.6, 9.0))
    gs = fig.add_gridspec(3, 1, hspace=0.04)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))
    panel_c(fig.add_subplot(gs[2]))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(a.out.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print('saved', a.out)
if __name__ == '__main__':
    main()
