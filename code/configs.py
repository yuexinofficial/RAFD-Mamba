RAW_ROOT = '/public/home/lm12345/CYX/datasets/Thyroid'
PRE_ROOT = '/public/home/lm12345/CYX/datasets/preprocessed_thyroid'
VMAMBA_PRETRAINED = '/public/home/lm12345/CYX/VMUNetSeg/models/vmamba_small_e238_ema.pth'
VMAMBA_TINY_PRETRAINED = '/public/home/lm12345/CYX/VMUNetSeg/models/vmamba_tiny_e292.pth'
IMG_SIZE = 256
RESIZE_IMAGE_INTERP = 'bilinear'
RESIZE_MASK_INTERP = 'nearest'
BINARIZE_THRESHOLD = {'TN3K': 127, 'DDTI': 0, 'TNUI2021': 0}
NORM_SCALE = 255.0
NORM_MODE = 'per_image_minmax_to_255'
NORM_STATS_UNUSED = {'TN3K': {'train': (66.9262, 51.2534), 'val': (66.7244, 50.8139), 'test': (57.9583, 50.3984)}, 'TNUI2021': {'train': (66.3133, 72.548), 'val': (67.053, 72.7203), 'test': (65.4795, 71.7961)}}
DATASETS = {'TN3K': dict(raw_dir='TN3K', classes=1, split_origin='official_5fold_fold0', split_note='官方 tn3k-trainval-fold0.json:train 2303 / val 576;test 614 为官方测试集。五折 val 两两不相交且每样本恰好当一次 val。TN3K 未发布病人 ID,只能图像级划分,非病人不相交。'), 'DDTI': dict(raw_dir='DDTI', classes=1, split_origin='third_party_us30k_random', split_note='划分来自 US30K 的随机切分,DDTI 本身无官方划分;且该版本已重采样为 256x256,非 DDTI 原始分辨率。🔴 计数更正(2026-09-25):实际为 **447/63/127 = 637**,三者不重叠且并集正好等于 img/ 的 637 张。此前笔记里的 446/62/126 是 `wc -l` 的欠计 —— splits/*.txt 末行无换行符,`wc -l` 少数一行。🔴 存在 **train↔test 内容泄漏 1 组**:train/test295、test71、test181、test325 与 test/test294 五张图像 md5 完全相同(mask 亦同,除 test181),即同一张图在 train 出现 4 次、test 出现 1 次。污染 1/127 = 0.8% 的测试样本,对均值的量级约 5e-4,不影响结论但需如实记录。'), 'TNUI2021': dict(raw_dir='TNUI2021', classes=1, split_origin='third_party_github_mirror', split_note='来自第三方 GitHub 镜像 zxg3017/TNUI-2021-(966/276/139),非官方发布入口。🔴 该镜像自带划分存在跨 split 内容泄漏:1381 个文件仅 1371 张唯一图像,train↔val 3 组(train/0032=val/0070、train/101008=val/101308、train/10652=val/101286)、val↔test 1 组(val/101180=test/10200)。无 train↔test 泄漏。用作 TN3K→TNUI 外部测试集时不影响结论;若日后要在 TNUI 上训练或用其 val 选模型,必须先清理。')}
IMG_SIZE = 256
VMAMBA_PRETRAINED = '/public/home/lm12345/CYX/VMUNetSeg/models/vmamba_small_e238_ema.pth'
VMAMBA_TINY_PRETRAINED = '/public/home/lm12345/CYX/VMUNetSeg/models/vmamba_tiny_e292.pth'
ARCH_PROTOCOLS = {
    'vmunet': dict(source='官方 VM-UNet configs/config_setting.py(ISIC18 分支;与 TN3K 同为二值任务)', depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1], drop_path_rate=0.2, pretrained=VMAMBA_PRETRAINED, epochs=300, batch_size=32, opt='AdamW', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, sched='cosine', t_max=50, eta_min=1e-05, loss='bce_dice', loss_kw=dict(wb=1, wd=1), loss_space='probability', norm='minmax_255', output_mode='sigmoid_binary'),
    'mybackbone': dict(model='vmunet', source='用户自建骨干:VMamba-Tiny(2292)+VSS解码器(2922)+普通skip;协议由用户自定', depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2], drop_path_rate=0.2, pretrained=VMAMBA_TINY_PRETRAINED, epochs=200, batch_size=32, opt='adamw', lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, sched='warmup_poly', power=0.9, warmup_epochs=10, min_lr=1e-06, loss='ecco_bce_dice', loss_kw=dict(wb=0.3, wd=0.7), loss_space='probability', norm='minmax_255', output_mode='sigmoid_binary'),
    'mybackbone_rafr4': dict(model='vmunet_rafr4', source='A1(RAFR)挂到 [1,2,3,4]:三条 skip + 主干;隔离"主干"这一项', depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2], drop_path_rate=0.2, pretrained=VMAMBA_TINY_PRETRAINED, rafr_levels=[1, 2, 3, 4], rafr_gate_bias=-4.0, epochs=200, batch_size=32, opt='adamw', lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, sched='warmup_poly', power=0.9, warmup_epochs=10, min_lr=1e-06, loss='ecco_bce_dice', loss_kw=dict(wb=0.3, wd=0.7), loss_space='probability', norm='minmax_255', output_mode='sigmoid_binary'),
    'mybackbone_dmsc': dict(model='vmunet_dmsc', source='自建骨干 + DMSC(模块 B,编码器 stage2/3 入口);协议与 mybackbone 相同', depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2], drop_path_rate=0.2, pretrained=VMAMBA_TINY_PRETRAINED, dmsc_levels=[2, 3], dmsc_heads=4, epochs=200, batch_size=32, opt='adamw', lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, sched='warmup_poly', power=0.9, warmup_epochs=10, min_lr=1e-06, loss='ecco_bce_dice', loss_kw=dict(wb=0.3, wd=0.7), loss_space='probability', norm='minmax_255', output_mode='sigmoid_binary'),
    'mybackbone_rafr4_dmsc': dict(model='vmunet_rafr4_dmsc', source='A1([1,2,3,4]) + B1(DMSC [2,3]);与 A_v2+B1 对照,看增益来自多尺度还是来自主干', depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2], drop_path_rate=0.2, pretrained=VMAMBA_TINY_PRETRAINED, rafr_levels=[1, 2, 3, 4], rafr_gate_bias=-4.0, dmsc_levels=[2, 3], dmsc_heads=4, dmsc_lambda=True, epochs=200, batch_size=32, opt='adamw', lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, sched='warmup_poly', power=0.9, warmup_epochs=10, min_lr=1e-06, loss='ecco_bce_dice', loss_kw=dict(wb=0.3, wd=0.7), loss_space='probability', norm='minmax_255', output_mode='sigmoid_binary'),
}
NORM_MODES = {'minmax_255': dict(kind='per_image_minmax', scale=255.0), 'minmax_01': dict(kind='per_image_minmax', scale=1.0), 'scale_01': dict(kind='div255')}
AUGMENT = dict(hflip_p=0.5, rotate_deg=15.0, scale=(0.9, 1.1), bc_limit=0.1, bc_p=0.3, gauss_p=0.0)
VAL_METRICS = ('dice', 'iou', 'hd95')
TEST_METRICS = ('dice', 'iou', 'hd95', 'asd', 'recall', 'precision')
BEST_METRIC = 'dice'
DEGENERATE_PENALTY = float((IMG_SIZE ** 2 + IMG_SIZE ** 2) ** 0.5)
MAIN_MODEL_PROTOCOL = 'mybackbone_rafr4_dmsc'
MAIN_MODEL_SPEC = dict(architecture='vmunet_rafr4_dmsc', rafr_levels=[1, 2, 3, 4], dmsc_levels=[2, 3], params=56182409, seed=42, result=dict(dice=0.8527, iou=0.7713, hd95=17.6412, asd=5.5047, recall=0.8742, precision=0.8689, best_val=0.8316, best_epoch=144, ckpt='best_model.pth'), run_dir='mybackbone/runs/mybackbone_rafr4_dmsc_s42/20260926_013316')

def proto_for_arch(arch):
    if arch in ARCH_PROTOCOLS:
        return ARCH_PROTOCOLS[arch]
    for p in ARCH_PROTOCOLS.values():
        if p.get('model') == arch:
            return p
    return None
