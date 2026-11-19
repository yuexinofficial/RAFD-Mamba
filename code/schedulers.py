import torch

class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_epochs=10, total_epochs=130, iters_per_epoch=1, power=0.9, base_lr=0.0003, min_lr=1e-06, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = total_epochs * iters_per_epoch
        self.power = power
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter = self.last_epoch
        if current_iter < self.warmup_iters:
            warmup_factor = current_iter / self.warmup_iters
            factor = max(0.001, warmup_factor)
            return [self.base_lr * factor for _ in self.optimizer.param_groups]
        progress = (current_iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
        factor = (1 - progress) ** self.power
        return [max(self.base_lr * factor, self.min_lr) for _ in self.optimizer.param_groups]

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, warmup_lr, min_lr, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.base_lr = float(base_lr)
        self.warmup_lr = float(warmup_lr)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        import math
        e = self.last_epoch
        if e < self.warmup_epochs:
            if self.warmup_epochs <= 0:
                lr = self.base_lr
            else:
                lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * e / self.warmup_epochs
        else:
            prog = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            prog = min(1.0, max(0.0, prog))
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * prog))
        return [lr for _ in self.optimizer.param_groups]

def build_scheduler(name, optimizer, proto, epochs, iters_per_epoch):
    if name == 'cosine':
        return (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=proto['t_max'], eta_min=proto['eta_min']), 'epoch')
    if name == 'warmup_poly':
        return (WarmupPolyLR(optimizer, warmup_epochs=proto['warmup_epochs'], total_epochs=epochs, iters_per_epoch=iters_per_epoch, power=proto['power'], base_lr=proto['lr'], min_lr=proto['min_lr']), 'iter')
    if name == 'warmup_cosine':
        return (WarmupCosineLR(optimizer, warmup_epochs=proto['warmup_epochs'], total_epochs=epochs, base_lr=proto['lr'], warmup_lr=proto['warmup_lr'], min_lr=proto['min_lr']), 'epoch')
    if name == 'step':
        return (torch.optim.lr_scheduler.StepLR(optimizer, step_size=proto['step_size'], gamma=proto['gamma']), 'epoch')
    if name == 'plateau':
        return (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=proto.get('plateau_mode', 'max'), patience=proto.get('plateau_patience', 5)), 'plateau')
    raise KeyError(f'未知 scheduler: {name}')
