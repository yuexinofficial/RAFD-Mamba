# RAFD-Mamba — Code and Figures

This package accompanies the manuscript:

> **RAFD-Mamba: A Mamba-Based Network Integrating Reliability-Aware Frequency Residual and Dynamic Multi-Scale Sparse Context for Thyroid Nodule Ultrasound Segmentation**

It contains the training/testing code, the figure-generation scripts, and the final figures used in the paper.

## 1. Layout

```
code/      Python source (comments and docstrings removed; see §5)
figures/   Final figures, named by their number in the manuscript
RESULTS.md Result tables and per-run directories (reference)
```

## 2. Environment

- Python 3.10, PyTorch (CUDA), single NVIDIA GPU
- Key packages: `torch`, `torchvision`, `numpy`, `albumentations`, `opencv-python`,
  `einops`, `timm`, `mamba-ssm`, `scikit-learn`, `matplotlib`, `Pillow`
- The selective-scan CUDA kernels of VMamba are required (`mamba_ssm.ops.selective_scan_interface`).

```bash
pip install -r requirements.txt
```

## 3. Data

- **TN3K** (official fold0): 2303 / 576 / 614 images for train / val / test
- **TNUI-2021** (cleaned public mirror): 966 / 272 / 139 images
- **DDTI** (external test only): 127 images

Preprocessing: grayscale → 3 channels, resize to 256×256, per-image min–max normalization
to [0, 255]; TN3K masks binarized at 127 and DDTI at 0.

The datasets are public and are not redistributed here. Point the code to your local copy
with `--pre-root` (or edit `PRE_ROOT` in `configs.py`).

## 4. Training and testing

```bash
# train the proposed model (TN3K, seed 42)
python train.py --dataset TN3K --architecture mybackbone_rafr4_dmsc \
                --seed 42 --exp rafr4_dmsc --gpu 0

# test on the internal test set
python test.py --ckpt output/rafr4_dmsc/<timestamp>/best_model.pth \
               --dataset TN3K --split test --gpu 0

# external validation on DDTI
python test.py --ckpt output/rafr4_dmsc/<timestamp>/best_model.pth \
               --dataset DDTI --split test --gpu 0
```

The comparison methods follow their authors' public implementations and are not
redistributed here; this package contains the proposed model and the modules it depends on.

Training settings used in the paper: 200 epochs, batch size 32, AdamW
(lr 1e-4, weight decay 1e-2), 10-epoch warm-up followed by polynomial decay
(power 0.9, min lr 1e-6), FP32 without AMP; the best checkpoint is selected by
validation Dice.

## 5. Note on comments

Source files authored by the authors have had all comments and docstrings removed
(`ast`-based stripping; code semantics unchanged). Files that reproduce third-party
modules — VM-UNet (`models/vmunet/`), EccoMamba HAE/SCO (`models/modules/hae.py`,
`sco.py`) and MSACA (`msaca.py`, `msaca_dual.py`) — are kept verbatim with their
original attribution headers, as required by their licenses.

## 6. Figures

| File | Manuscript | Content |
|---|---|---|
| `Figure1_overall.pdf/.png` | Figure 1 | Overall architecture (a) and VSS block (b) |
| `Figure2_RAFR.pdf/.png` | Figure 2 | Reliability-Aware Frequency Residual module |
| `Figure3_DMSC.pdf/.png` | Figure 3 | Dynamic Multi-Scale Sparse Context module |
| `Figure4_dice_hd95.pdf/.png` | Figure 4 | Dice / HD95 bar comparison on TN3K |
| `Figure5_qualitative.png` | Figure 5 | Qualitative comparison (GT / Ours / four baselines) |
| `Figure6a_precision_recall.png` | Figure 6a | Pixel-level precision–recall curve on TN3K |
| `Figure6b_tsne.png` | Figure 6b | t-SNE of decoder features |
| `Figure7_gate_telemetry.pdf/.png` | Figure 7 | Gate telemetry of RAFR and DMSC |
| `Figure8_validation_curves.png` | Figure 8 | Validation Dice curves of the ablation configurations |

Most raster figures are exported at 600 dpi (effective resolution at the printed
width is ≥600 dpi); Figures 1–4 and 7 are also provided as vector PDFs.

## 7. Figure-generation scripts

`code/probes/` contains the scripts that regenerate each figure from checkpoints or
`metrics.jsonl`, e.g.:

```bash
python probes/make_compare_figure_bw.py --ckpt <run_dir> --n 4 --gpu 0 --dpi 600 --out Figure5.png
python probes/make_summary_figures_en.py          # Figures 4 (and supplementary delta panel)
python probes/make_gate_figure_en.py --out Figure7.png
python probes/make_curves.py --ablation --dpi 600 --out Figure8.png
python probes/make_tsne.py --ckpt <run_dir> --n-img 40 --per-class 50 --seed 0 --dpi 600 --out Figure6b.png
```
