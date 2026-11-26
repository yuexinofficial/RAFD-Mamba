import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configs as C
SCHEMA_VERSION = 1

def _tn3k_splits():
    raw = os.path.join(C.RAW_ROOT, 'TN3K')
    tv = sorted(os.listdir(os.path.join(raw, 'trainval-image')))
    fold = json.load(open(os.path.join(raw, 'tn3k-trainval-fold0.json')))
    out = {'train': [tv[i] for i in fold['train']], 'val': [tv[i] for i in fold['val']], 'test': sorted(os.listdir(os.path.join(raw, 'test-image')))}
    assert len(set(fold['train']) & set(fold['val'])) == 0, 'fold0 train/val 有交集'
    assert len(set(fold['train']) | set(fold['val'])) == len(tv), 'fold0 未覆盖全部 trainval'
    return out

def _ddti_splits():
    raw = os.path.join(C.RAW_ROOT, 'DDTI')
    out = {}
    for s in ('train', 'val', 'test'):
        names = [l.strip() for l in open(os.path.join(raw, 'splits', f'{s}.txt')) if l.strip()]
        out[s] = [f'{n}.png' for n in names]
    return out

def _tnui_splits():
    raw = os.path.join(C.RAW_ROOT, 'TNUI2021', 'thyroid_data')
    return {s: sorted(os.listdir(os.path.join(raw, s, 'images'))) for s in ('train', 'val', 'test')}
SPLIT_FUNCS = {'TN3K': _tn3k_splits, 'DDTI': _ddti_splits, 'TNUI2021': _tnui_splits}

def _paths(dataset, split, name):
    if dataset == 'TN3K':
        base = os.path.join(C.RAW_ROOT, 'TN3K')
        sub = 'test' if split == 'test' else 'trainval'
        return (os.path.join(base, f'{sub}-image', name), os.path.join(base, f'{sub}-mask', name))
    if dataset == 'DDTI':
        base = os.path.join(C.RAW_ROOT, 'DDTI')
        return (os.path.join(base, 'img', name), os.path.join(base, 'label', name))
    if dataset == 'TNUI2021':
        base = os.path.join(C.RAW_ROOT, 'TNUI2021', 'thyroid_data', split)
        return (os.path.join(base, 'images', name), os.path.join(base, 'masks', name))
    raise KeyError(dataset)

def convert_one(dataset, split, name, out_dir):
    ipath, mpath = _paths(dataset, split, name)
    if not (os.path.exists(ipath) and os.path.exists(mpath)):
        return (None, f'缺文件: {ipath} | {mpath}')
    img = Image.open(ipath).convert('L')
    msk = Image.open(mpath).convert('L')
    thr = C.BINARIZE_THRESHOLD[dataset]
    m = np.asarray(msk) > thr
    msk_bin = Image.fromarray((m * 255).astype(np.uint8), mode='L')
    size = (C.IMG_SIZE, C.IMG_SIZE)
    img_r = img.resize(size, Image.Resampling.BILINEAR)
    msk_r = msk_bin.resize(size, Image.Resampling.NEAREST)
    stem = os.path.splitext(name)[0]
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)
    img_r.save(os.path.join(out_dir, 'images', f'{stem}.png'))
    msk_r.save(os.path.join(out_dir, 'masks', f'{stem}.png'))
    return (stem, None)

def run(dataset, force=False):
    splits = SPLIT_FUNCS[dataset]()
    root = os.path.join(C.PRE_ROOT, dataset)
    if os.path.exists(os.path.join(root, 'split_manifest.json')) and (not force):
        print(f'[skip] {dataset} 已存在 manifest(加 --force 重做)')
        return
    manifest = dict(schema_version=SCHEMA_VERSION, dataset=dataset, source=os.path.join(C.RAW_ROOT, C.DATASETS[dataset]['raw_dir']), split_origin=C.DATASETS[dataset]['split_origin'], split_note=C.DATASETS[dataset]['split_note'], built_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), resize=dict(size=[C.IMG_SIZE, C.IMG_SIZE], image_interp=C.RESIZE_IMAGE_INTERP, mask_interp=C.RESIZE_MASK_INTERP), binarize_threshold=C.BINARIZE_THRESHOLD[dataset], channels='grayscale (L) -> replicated to 3ch at load time', normalization=dict(kind='per_image_minmax', scale=255.0, order='uint8 -> per-image min-max to [0,255]'), crop='none', splits={})
    for split, names in splits.items():
        out_dir = os.path.join(root, split)
        n_ok, errs, fg = (0, [], [])
        for name in names:
            stem, err = convert_one(dataset, split, name, out_dir)
            if err:
                errs.append(f'{name}: {err}')
                continue
            n_ok += 1
            m = np.asarray(Image.open(os.path.join(out_dir, 'masks', f'{stem}.png')))
            fg.append(float((m > 127).mean()))
        fg = np.array(fg) if fg else np.array([np.nan])
        manifest['splits'][split] = dict(n=n_ok, requested=len(names), fg_ratio_mean=round(float(np.nanmean(fg)), 5), fg_ratio_min=round(float(np.nanmin(fg)), 5), fg_ratio_max=round(float(np.nanmax(fg)), 5), n_empty_mask=int((fg < 1e-09).sum()))
        print(f'  {dataset}/{split}: {n_ok}/{len(names)}  前景均值={np.nanmean(fg):.4f}  空mask={int((fg < 1e-09).sum())}  错误={len(errs)}')
        if errs:
            manifest['splits'][split]['errors'] = errs[:20]
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'split_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'  -> manifest: {os.path.join(root, 'split_manifest.json')}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='TN3K', choices=list(C.DATASETS) + ['all'])
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    names = list(C.DATASETS) if a.dataset == 'all' else [a.dataset]
    for d in names:
        print(f'=== {d} ===')
        run(d, force=a.force)
if __name__ == '__main__':
    main()
