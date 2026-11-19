import os
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs as C

def normalize_like_author(img, scale=255.0):
    lo, hi = (float(img.min()), float(img.max()))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return (img.astype(np.float32) - lo) / (hi - lo) * scale

def build_transform(augment: bool, seed: int):
    if not augment:
        return None
    import albumentations as A
    import cv2
    a = C.AUGMENT
    return A.Compose([A.HorizontalFlip(p=a['hflip_p']), A.Affine(scale=a['scale'], rotate=(-a['rotate_deg'], a['rotate_deg']), interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101, p=1.0), A.RandomBrightnessContrast(brightness_limit=a['bc_limit'], contrast_limit=a['bc_limit'], p=a['bc_p']), A.GaussNoise(std_range=(0.02, 0.08), p=a['gauss_p'])], seed=seed, strict=True)

class ThyroidDataset(Dataset):

    def __init__(self, dataset, split, pre_root=C.PRE_ROOT, augment=False, seed=42, norm=None):
        self.dataset, self.split = (dataset, split)
        if norm is None:
            raise ValueError('必须显式指定 norm 模式(见 configs.NORM_MODES),避免不同架构误用同一尺度')
        if norm not in C.NORM_MODES:
            raise KeyError(f'未知 norm: {norm};可用 {list(C.NORM_MODES)}')
        self.norm_spec = C.NORM_MODES[norm]
        self.seed = seed
        self.augment = augment
        root = os.path.join(pre_root, dataset, split)
        self.img_dir = os.path.join(root, 'images')
        self.msk_dir = os.path.join(root, 'masks')
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f'{self.img_dir} 不存在 —— 先跑 python preprocess.py --dataset {dataset}')
        self.names = sorted((os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith('.png')))
        if not self.names:
            raise RuntimeError(f'{self.img_dir} 为空')
        self.transform = build_transform(augment, self.seed * 100003 + 0)

    def set_epoch(self, epoch):
        if self.augment:
            self.transform = build_transform(True, self.seed * 100003 + epoch)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        n = self.names[idx]
        img = np.asarray(Image.open(os.path.join(self.img_dir, f'{n}.png')).convert('L'))
        msk = np.asarray(Image.open(os.path.join(self.msk_dir, f'{n}.png')).convert('L'))
        if self.transform is not None:
            out = self.transform(image=img, mask=msk)
            img, msk = (out['image'], out['mask'])
        if self.norm_spec['kind'] == 'per_image_minmax':
            img = normalize_like_author(img, self.norm_spec['scale'])
        elif self.norm_spec['kind'] == 'div255':
            img = img.astype(np.float32) / 255.0
        else:
            raise KeyError(f'未知 norm kind: {self.norm_spec['kind']}')
        img = np.repeat(img[:, :, None], 3, axis=2)
        msk = msk > 127
        image = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        label = torch.from_numpy(np.ascontiguousarray(msk)).long()
        return (image, label)

def build_dataset(dataset, split, pre_root=C.PRE_ROOT, augment=False, seed=42, norm=None):
    return ThyroidDataset(dataset, split, pre_root=pre_root, augment=augment, seed=seed, norm=norm)
if __name__ == '__main__':
    for ds in C.DATASETS:
        for sp in ('train', 'val', 'test'):
            try:
                d = build_dataset(ds, sp, augment=sp == 'train', norm='minmax_255')
            except FileNotFoundError:
                print(f'{ds}/{sp}: 未预处理,跳过')
                continue
            x, y = d[0]
            fg = np.mean([float(d[i][1].float().mean()) for i in range(0, len(d), max(1, len(d) // 50))])
            print(f'{ds:9s}/{sp:5s} n={len(d):5d} image={tuple(x.shape)} {x.dtype} [{x.min():.2f},{x.max():.2f}] label={tuple(y.shape)} {y.dtype} uniq={torch.unique(y).tolist()} 前景≈{fg:.4f}')
