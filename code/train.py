import argparse
import io
import contextlib
import json
import os
import random
import sys
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import configs as C
from data.thyroid_dataset import build_dataset
from losses import build_loss
from models.adapter import build_model
from protocol.shared_validation import evaluate
from schedulers import build_scheduler

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='TN3K', choices=list(C.DATASETS))
    p.add_argument('--pre-root', default=C.PRE_ROOT)
    p.add_argument('--architecture', default='vmunet', choices=list(C.ARCH_PROTOCOLS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exp', required=True)
    p.add_argument('--output-root', default=os.path.join(HERE, 'output'))
    p.add_argument('--gpu', type=int, default=0, help='可见设备序号;启动脚本用 CUDA_VISIBLE_DEVICES 限定范围')
    p.add_argument('--no-pretrained', dest='use_pretrained', action='store_false', default=True)
    p.add_argument('--max-train-batches', type=int, default=0, help='冒烟用,0=全量')
    p.add_argument('--max-val-batches', type=int, default=0)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=None)
    return p

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('需要 CUDA(VMamba 的 selective_scan 无 CPU 回退)')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    proto = dict(C.ARCH_PROTOCOLS[args.architecture])
    epochs = args.epochs if args.epochs is not None else proto['epochs']
    bs = args.batch_size if args.batch_size is not None else proto['batch_size']
    lr = args.lr if args.lr is not None else proto['lr']
    wd = args.weight_decay if args.weight_decay is not None else proto['weight_decay']
    official = args.epochs is None and args.batch_size is None and (args.lr is None) and (args.weight_decay is None)
    run_dir = os.path.join(args.output_root, args.exp, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(os.path.join(run_dir, 'log.txt'), 'a', buffering=1)

    def log(msg=''):
        print(msg)
        log_f.write(msg + '\n')
    log(f'=== ThyroidSeg train | {args.exp} ===')
    log(f'run_dir : {run_dir}')
    log(f'device  : cuda:{args.gpu} = {torch.cuda.get_device_name(args.gpu)}')
    log(f'arch    : {args.architecture}  协议来源: {proto['source']}')
    log(f'dataset : {args.dataset}  seed={args.seed}  epochs={epochs}  bs={bs}  AMP=off(FP32)  img_size={C.IMG_SIZE}(全局统一,覆盖作者值)')
    _ns = C.NORM_MODES[proto['norm']]
    _nd = '逐图 min-max → [0,%g]' % _ns['scale'] if _ns['kind'] == 'per_image_minmax' else 'x/255 → [0,1]'
    log(f'归一化  : {proto['norm']} ({_nd})')
    tr_ds = build_dataset(args.dataset, 'train', pre_root=args.pre_root, augment=True, seed=args.seed, norm=proto['norm'])
    va_ds = build_dataset(args.dataset, 'val', pre_root=args.pre_root, augment=False, norm=proto['norm'])
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=max(1, bs // 2), shuffle=False, num_workers=8, pin_memory=True)
    iters_per_epoch = min(len(tr_dl), args.max_train_batches) if args.max_train_batches else len(tr_dl)
    log(f'train n={len(tr_ds)}  val n={len(va_ds)}  iters/epoch={iters_per_epoch}  (test 全程不碰)')
    set_seed(args.seed)
    pretrained = proto['pretrained'] if args.use_pretrained else None
    model = build_model(args.architecture, pretrained=pretrained).to(device)
    if pretrained:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model.load_from()
        log('预训练权重:')
        for line in buf.getvalue().splitlines():
            if 'update' in line or 'finished' in line or '加载' in line or ('Loaded' in line):
                log('  ' + line.strip())
        miss = [p for p, _ in model.named_parameters() if p.startswith(('backbone.msc',))]
    elif proto.get('internal_pretrained'):
        log('预训练权重: 由模型**内部**加载(torchvision ImageNet),无需外部 ckpt')
    else:
        log('预训练权重: 未加载(从零训练)')
    n_param = sum((p.numel() for p in model.parameters() if p.requires_grad))
    log(f'可训练参数: {n_param / 1000000.0:.2f}M')
    trainable = [p for p in model.parameters() if p.requires_grad]
    _opt = str(proto['opt']).lower()
    if _opt == 'adamw':
        optimizer = torch.optim.AdamW(trainable, lr=lr, betas=proto['betas'], eps=proto.get('eps', 1e-08), weight_decay=wd)
        opt_desc = f'AdamW lr={lr} wd={wd} betas={tuple(proto['betas'])}'
    elif _opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(trainable, lr=lr, weight_decay=wd, momentum=proto['momentum'])
        opt_desc = f'RMSprop lr={lr} wd={wd} momentum={proto['momentum']}'
    elif _opt == 'sgd':
        optimizer = torch.optim.SGD(trainable, lr=lr, momentum=proto['momentum'], nesterov=proto.get('nesterov', False), weight_decay=wd)
        opt_desc = f'SGD lr={lr} wd={wd} momentum={proto['momentum']} nesterov={proto.get('nesterov', False)}'
    else:
        raise KeyError(f'未知 optimizer: {proto['opt']}')
    scheduler, step_per = build_scheduler(proto['sched'], optimizer, proto, epochs, iters_per_epoch)
    criterion = build_loss(proto['loss'], **proto['loss_kw'])
    log(f'优化器: {opt_desc}')
    _sch_extra = {'plateau': lambda: f'patience={proto.get('plateau_patience', 5)} mode={proto.get('plateau_mode', 'max')}(用 val 指标步进)'}
    log(f'调度器: {proto['sched']} ({_sch_extra.get(step_per, lambda: f'按 {step_per} 步进')()})')
    log(f'损失  : {proto['loss']} {proto['loss_kw']}  空间={proto['loss_space']}')
    if not official:
        log('⚠️ 协议提示: 传入了诊断用超参覆盖,本次结果不与作者默认配置可比')
    cfg_dump = dict(exp=args.exp, dataset=args.dataset, seed=args.seed, architecture=args.architecture, num_classes=1, img_size=C.IMG_SIZE, amp=False, protocol_is_official=official, protocol_source=proto['source'], protocol=dict(split='TN3K official fold0 (train 2303 / val 576); test 614 未使用', amp=False, best_metric=C.BEST_METRIC, val_metrics=list(C.VAL_METRICS), test_metrics=list(C.TEST_METRICS), hd95_scale=f'{C.IMG_SIZE}x{C.IMG_SIZE} pixel', degenerate_penalty=C.DEGENERATE_PENALTY, normalization=dict(mode=proto['norm'], spec=C.NORM_MODES[proto['norm']], note='按架构取各自作者的做法'), augment=C.AUGMENT), epochs=epochs, batch_size=bs, iters_per_epoch=iters_per_epoch, optimizer=dict(name=proto['opt'], lr=lr, weight_decay=wd, **{k: list(v) if isinstance(v, tuple) else v for k, v in proto.items() if k in ('betas', 'eps', 'momentum')}), scheduler=dict(name=proto['sched'], step_per=step_per, **{k: v for k, v in proto.items() if k in ('t_max', 'eta_min', 'power', 'warmup_epochs', 'min_lr')}), loss=dict(name=proto['loss'], space=proto['loss_space'], **proto['loss_kw']), pretrained=pretrained, params_trainable=n_param, model_config=model.export_config(), started_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg_dump, f, ensure_ascii=False, indent=2)
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')

    def save_ckpt(name, epoch, best_epoch, best_dice, final_val_dice):
        torch.save(dict(epoch=epoch, best_epoch=best_epoch, best_dice=best_dice, final_val_dice=final_val_dice, state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(), args=vars(args), num_classes=1, architecture=args.architecture, model_config=model.export_config()), os.path.join(run_dir, name))
    best_dice, best_epoch, cur = (-1.0, -1, float('nan'))
    for epoch in range(1, epochs + 1):
        tr_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        tot, seen = (0.0, 0)
        for bi, (img, lbl) in enumerate(tr_dl):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            img = img.to(device, non_blocking=True)
            out = model(img)
            if proto.get('target_mode') == 'multiclass':
                lbl = lbl.to(device, non_blocking=True).long()
                loss = criterion(out, lbl)
            else:
                lbl = lbl.to(device, non_blocking=True).float().unsqueeze(1)
                if proto['loss_space'] == 'probability':
                    loss = criterion(torch.sigmoid(out[:, 1:2]), lbl)
                else:
                    loss = criterion(out[:, 1:2], lbl)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step_per == 'iter':
                scheduler.step()
            tot += loss.item() * img.size(0)
            seen += img.size(0)
        if step_per == 'epoch':
            scheduler.step()
        tr_loss = tot / max(1, seen)
        has_rafr = hasattr(model, 'rafr_stats') and hasattr(model, 'reset_rafr_stats')
        has_dmsc = hasattr(model, 'dmsc_stats') and hasattr(model, 'reset_dmsc_stats')
        if has_rafr:
            model.reset_rafr_stats()
        if has_dmsc:
            model.reset_dmsc_stats()
        val, _ = evaluate(model, va_dl, device, max_batches=args.max_val_batches)
        rafr = model.rafr_stats() if has_rafr else {}
        dmsc = model.dmsc_stats() if has_dmsc else {}
        cur = val[C.BEST_METRIC]
        if step_per == 'plateau':
            scheduler.step(cur)
        dt = time.time() - t0
        log(f'ep {epoch:3d}/{epochs} | lr {optimizer.param_groups[0]['lr']:.2e} | loss {tr_loss:.4f} | val_dice {val['dice']:.4f} | iou {val['iou']:.4f} | hd95 {val['hd95']:.3f} | {dt:.0f}s')
        if rafr:
            log('    RAFR ' + '  '.join((f'skip{k}: lam={v['lambda_mean']:.4f} E={v['E']:.2e}' for k, v in sorted(rafr.items()))))
        if dmsc:

            def _dev(v):
                return max(abs(v['eta1'] - 1 / 3), abs(v['eta2'] - 1 / 3), abs(v['eta3'] - 1 / 3))
            log('    DMSC ' + '  '.join((f'stage{k}: lam={v['lambda_mean']:.4f} E={v['E']:.2e} eta=({v['eta1']:.3f},{v['eta2']:.3f},{v['eta3']:.3f}) dev={_dev(v):.4f} ab=({v['alpha']:.3f},{v['beta']:.3f})' for k, v in sorted(dmsc.items()))))
        if cur > best_dice:
            best_dice, best_epoch = (cur, epoch)
            save_ckpt('best_model.pth', epoch, best_epoch, best_dice, cur)
        save_ckpt('final_model.pth', epoch, best_epoch, best_dice, cur)
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'], train_loss=tr_loss, val_dice=val['dice'], val_iou=val['iou'], val_hd95=val['hd95'], val_n=val['n'], val_n_skipped=val['n_skipped'], rafr={k: {kk: vv for kk, vv in v.items()} for k, v in rafr.items()}, dmsc={k: {kk: vv for kk, vv in v.items()} for k, v in dmsc.items()}, seconds=round(dt, 2)), ensure_ascii=False) + '\n')
    save_ckpt('last_model.pth', epochs, best_epoch, best_dice, cur)
    log(f'DONE. best val_dice={best_dice:.4f} @ ep{best_epoch}')
    log('ARM_COMPLETE')
    log_f.close()
    open(os.path.join(run_dir, 'ARM_COMPLETE'), 'w').close()
if __name__ == '__main__':
    main()
