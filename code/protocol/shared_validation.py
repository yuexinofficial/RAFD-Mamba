import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs as C
from metrics import aggregate, binarize_logits, image_metrics

@torch.no_grad()
def evaluate(model, loader, device, keys=None, max_batches=0):
    keys = keys or C.VAL_METRICS
    om = getattr(model, 'output_mode', 'sigmoid_binary')
    was_training = model.training
    model.eval()
    per_image = []
    for bi, (img, lbl) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        img = img.to(device, non_blocking=True)
        out = model(img)
        pred = binarize_logits(out, om)
        pred = pred.detach().cpu().numpy()
        gt = lbl.numpy()
        for i in range(pred.shape[0]):
            per_image.append(image_metrics(pred[i], gt[i], keys=keys))
    if was_training:
        model.train()
    return (aggregate(per_image, keys), per_image)
if __name__ == '__main__':
    '自检两件事:(1) 完美模型必须拿到 dice=1/hd95=0;(2) 评估器与 metrics 逐图一致。'
    from torch.utils.data import DataLoader
    H = W = C.IMG_SIZE

    class Synth(torch.utils.data.Dataset):

        def __len__(self):
            return 12

        def __getitem__(self, i):
            g = np.zeros((H, W), np.float32)
            r = np.random.default_rng(i)
            y, x = r.integers(40, 180, 2)
            h, w = r.integers(30, 70, 2)
            g[y:y + h, x:x + w] = 1.0
            logits = np.stack([np.zeros_like(g), g * 10.0 - 5.0])
            return (torch.from_numpy(logits).float(), torch.from_numpy(g).long())

    class Identity(torch.nn.Module):

        def forward(self, x):
            return x
    dl = DataLoader(Synth(), batch_size=4, shuffle=False, num_workers=0)
    agg, per = evaluate(Identity(), dl, torch.device('cpu'))
    ok = abs(agg['dice'] - 1.0) < 1e-09 and abs(agg['iou'] - 1.0) < 1e-09 and (agg['hd95'] == 0.0)
    print(f'  {('PASS' if ok else 'FAIL')}  完美模型: n={agg['n']} dice={agg['dice']:.6f} iou={agg['iou']:.6f} hd95={agg['hd95']:.4f}')
    same = True
    for i, (x, y) in enumerate(dl):
        for j in range(x.shape[0]):
            ref = image_metrics(x[j, 1].numpy() > 0, y[j].numpy(), keys=C.VAL_METRICS)
            got = per[i * 4 + j]
            if (ref is None) != (got is None):
                same = False
            elif ref is not None and any((abs(ref[k] - got[k]) > 1e-12 for k in C.VAL_METRICS)):
                same = False
    print(f'  {('PASS' if same else 'FAIL')}  评估器与 metrics.image_metrics 逐图逐指标一致')
    sys.exit(0 if ok and same else 1)
