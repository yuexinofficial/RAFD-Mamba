import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configs as C
try:
    from medpy.metric import binary as _mpy
except ImportError:
    _mpy = None
OUTPUT_MODES = ('sigmoid_binary', 'softmax_binary')

def binarize_logits(logits, output_mode='sigmoid_binary'):
    import torch
    if output_mode not in OUTPUT_MODES:
        raise KeyError(f'未知 output_mode: {output_mode};可用 {OUTPUT_MODES}')
    t = logits
    if output_mode == 'softmax_binary':
        if t.dim() < 4:
            raise ValueError('softmax_binary 需要 (B,C,H,W) 的 logits')
        return t.argmax(dim=1) == 1
    if t.dim() == 4:
        t = t[:, 1] if t.shape[1] == 2 else t[:, 0]
    return t > 0.0

def image_metrics(pred, gt, keys=C.TEST_METRICS, degenerate_penalty=None):
    p = np.asarray(pred) > 0
    g = np.asarray(gt) > 0
    ps, gs = (int(p.sum()), int(g.sum()))
    if ps == 0 and gs == 0:
        return None
    acc = float((p == g).sum()) / int(p.size)
    if ps == 0 or gs == 0:
        d = float(C.DEGENERATE_PENALTY if degenerate_penalty is None else degenerate_penalty)
        out = dict(dice=0.0, iou=0.0, recall=0.0, precision=0.0, accuracy=acc, hd95=d, asd=d)
        return {k: out[k] for k in keys}
    inter = int(np.logical_and(p, g).sum())
    out = dict(dice=2.0 * inter / (ps + gs), iou=inter / (ps + gs - inter), recall=inter / gs, precision=inter / ps, accuracy=acc)
    if 'hd95' in keys or 'asd' in keys:
        d = float(C.DEGENERATE_PENALTY if degenerate_penalty is None else degenerate_penalty)
        if _mpy is None:
            out['hd95'] = out['asd'] = d
        else:
            pu, gu = (p.astype(np.uint8), g.astype(np.uint8))
            try:
                out['hd95'] = float(_mpy.hd95(pu, gu))
            except Exception:
                out['hd95'] = d
            try:
                out['asd'] = float(_mpy.asd(pu, gu))
            except Exception:
                out['asd'] = out['hd95']
    return {k: out[k] for k in keys}

def aggregate(per_image, keys=None):
    keys = keys or C.TEST_METRICS
    valid = [d for d in per_image if d is not None]
    res = dict(n=len(valid), n_skipped=len(per_image) - len(valid))
    for k in keys:
        v = np.array([d[k] for d in valid], dtype=np.float64)
        res[k] = float(v.mean()) if len(v) else float('nan')
        res[k + '_std'] = float(v.std()) if len(v) else float('nan')
    return res

def _self_test():
    import torch
    H = W = 8
    g = np.zeros((H, W), np.uint8)
    g[2:6, 2:6] = 1
    ok = True

    def t(name, cond):
        nonlocal ok
        print(('  PASS  ' if cond else '  FAIL  ') + name)
        ok = ok and cond
    print('[T1] 完全正确')
    r = image_metrics(g, g)
    t(f'dice=1 iou=1 hd95=0 asd=0 (得 dice={r['dice']:.3f} iou={r['iou']:.3f} hd95={r['hd95']:.3f} asd={r['asd']:.3f})', abs(r['dice'] - 1) < 1e-09 and abs(r['iou'] - 1) < 1e-09 and (r['hd95'] == 0) and (r['asd'] == 0))
    print('[T2] 完全不相交')
    p2 = np.zeros((H, W), np.uint8)
    p2[0:2, 0:2] = 1
    r = image_metrics(p2, g)
    t(f'dice=0 iou=0 recall=0 precision=0 (得 {r['dice']:.3f}/{r['iou']:.3f}/{r['recall']:.3f}/{r['precision']:.3f})', r['dice'] == 0 and r['iou'] == 0 and (r['recall'] == 0) and (r['precision'] == 0))
    print('[T3] 预测为空、GT 非空 → 重叠指标 0,HD95/ASD 记对角线')
    r = image_metrics(np.zeros((H, W), np.uint8), g)
    t(f'hd95=asd=对角线({r['hd95']:.2f}) 且 dice=0', r['dice'] == 0 and abs(r['hd95'] - C.DEGENERATE_PENALTY) < 1e-06 and (abs(r['asd'] - C.DEGENERATE_PENALTY) < 1e-06))
    print('[T4] 双方均空 → 返回 None(不进任何分母)')
    t('两者全空返回 None', image_metrics(np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)) is None)
    print('[T5] 聚合:None 被计入 n_skipped,不污染均值')
    agg = aggregate([image_metrics(g, g), None, image_metrics(g, g)])
    t(f'n=2 n_skipped=1 dice=1.0 (得 n={agg['n']} skip={agg['n_skipped']} dice={agg['dice']:.3f})', agg['n'] == 2 and agg['n_skipped'] == 1 and (abs(agg['dice'] - 1) < 1e-09))
    print('[T6] 二值化取前景 logit>0,不用 argmax')
    logits = torch.full((1, 2, H, W), -5.0)
    logits[:, 1, 2:6, 2:6] = 5.0
    b = binarize_logits(logits)[0].numpy()
    t('前景通道 >0 的区域被判为前景', bool((b == g.astype(bool)).all()))
    one = torch.full((1, 1, H, W), 3.0)
    t('单通道输出全正 → 全被判前景(若用 argmax 会全判背景)', bool(binarize_logits(one).all()))
    print('[T7] output_mode 显式生效 —— 构造 bg=10/fg=2 的"真背景logit"情形')
    _chk = torch.tensor([[[[10.0]], [[2.0]]]])
    t(f'测试张量形状为 (1,2,1,1)(不是 (2,1,1,1))', tuple(_chk.shape) == (1, 2, 1, 1))
    fake = torch.tensor([[[[10.0]], [[2.0]]]])
    sig = binarize_logits(fake, 'sigmoid_binary')
    sm = binarize_logits(fake, 'softmax_binary')
    t(f'零占位口径(ch1>0) 判前景 = {bool(sig.flatten()[0])}', bool(sig.flatten()[0]) is True)
    t(f'真背景口径(argmax) 判背景 = {not bool(sm.flatten()[0])}', bool(sm.flatten()[0]) is False)
    t('两种模式在此情形下结论相反 ⇒ output_mode 不是摆设', bool(sig.flatten()[0]) != bool(sm.flatten()[0]))
    zero_ph = torch.tensor([[[[0.0]], [[2.0]]]])
    t('通道0恒为0时两模式逐像素等价', bool((binarize_logits(zero_ph, 'sigmoid_binary') == binarize_logits(zero_ph, 'softmax_binary')).all()))
    try:
        binarize_logits(fake, 'unknown_mode')
        t('未知模式应报错', False)
    except KeyError:
        t('未知模式报错', True)
    print('\n' + '=' * 56)
    print('指标自检: ' + ('全部通过' if ok else '存在失败'))
    return ok
if __name__ == '__main__':
    sys.exit(0 if _self_test() else 1)
