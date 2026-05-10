# EEEM068 — Vision Transformer Based Multi-label Knee MRI Classification

**University of Surrey | MSc Artificial Intelligence | EEEM068**

A ViT-S/16 pipeline for multi-label knee MRI classification on the [MRNet benchmark](https://stanfordmlgroup.github.io/competitions/mrnet/), jointly predicting abnormality, ACL tear, and meniscal tear from volumetric MRI studies.

---

## Project Structure

```
EEEM068-MRNet-ViT/
│
├── MRNet_ViT.ipynb            # Main notebook — single-plane classification
├── MRNet_multi_plane.ipynb    # Extra credit — multi-plane fusion (sagittal + coronal + axial)
├── MRNet_Extra_Credit.ipynb   # Extra credit — t-SNE feature-space visualisation & error analysis
├── OOD_images/                # External knee MRI images for OOD inference
├── IEEE_MRNet_Report.pdf      # Project report
└── README.md
```

---

## Notebooks Overview

### `MRNet_ViT.ipynb` — Single-Plane Classifier
The main submission notebook. Covers the full pipeline end-to-end:
- Dataset loading and preprocessing (coronal plane, N=4 slices)
- ViT-S/16 fine-tuning with differential learning rates (freeze all → unfreeze last 2 blocks + head)
- **Focal Loss** (γ=2) replacing class-weighted BCE for better minority-class learning
- Evaluation with accuracy, precision, recall, F1, **IoU**, and AUC-ROC per condition
- Confusion matrix visualisation
- Attention rollout visualisation ([CLS] → patch saliency maps)
- OOD inference on external knee images with stricter per-class thresholds

### `MRNet_multi_plane.ipynb` — Multi-Plane Fusion
Extension notebook. Trains the same ViT-S/16 backbone across all three MRI planes (sagittal, coronal, axial) and averages logits for a fused exam-level prediction.

### `MRNet_Extra_Credit.ipynb` — Feature-Space Analysis
Extra credit notebook. Extracts [CLS] embeddings for all 120 validation exams, applies PCA (50 components) + t-SNE (perplexity=30), and plots 2-D feature-space coloured by condition label to inspect class separability.

---

## Results Summary

### Single-Plane (Focal Loss)

| Condition | Acc. | Prec. | Rec. | F1 | IoU | AUC |
|-----------|------|-------|------|----|-----|-----|
| Abnormal  | 0.792 | 0.891 | 0.834 | 0.861 | 0.756 | 0.798 |
| ACL       | 0.717 | 0.641 | 0.842 | 0.731 | 0.576 | 0.779 |
| Meniscus  | 0.625 | 0.558 | 0.683 | 0.614 | 0.443 | 0.718 |
| **Macro AUC** | | | | | | **0.765** |

### Multi-Plane vs Single-Plane AUC

| Condition | Single-Plane | Multi-Plane | Δ AUC |
|-----------|-------------|-------------|-------|
| Abnormal  | 0.798 | 0.761 | −0.034 |
| ACL       | 0.779 | 0.698 | −0.069 |
| Meniscus  | 0.718 | 0.654 | −0.050 |
| **Macro** | **0.765** | **0.704** | **−0.052** |

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
pip install torch torchvision timm
pip install scikit-learn pandas numpy matplotlib seaborn
pip install Pillow tqdm huggingface_hub
```

Or all at once:

```bash
pip install torch torchvision timm scikit-learn pandas numpy matplotlib seaborn Pillow tqdm huggingface_hub
```

> **Mac users (Python 3.14):** if Jupyter can't find a package you just installed, run inside a notebook cell:
> ```python
> import sys
> !{sys.executable} -m pip install <package>
> ```

### Dataset

1. Download [MRNet v1.0](https://stanfordmlgroup.github.io/competitions/mrnet/) (requires registration)
2. Extract to a local folder — the expected structure is:
```
MRNet-v1.0/
├── train/
│   ├── sagittal/
│   ├── coronal/
│   └── axial/
├── valid/
│   ├── sagittal/
│   ├── coronal/
│   └── axial/
├── train-abnormal.csv
├── train-acl.csv
├── train-meniscus.csv
├── valid-abnormal.csv
├── valid-acl.csv
└── valid-meniscus.csv
```

3. Update `DATA_ROOT` in each notebook to point to your local path:
```python
DATA_ROOT = "/your/path/to/MRNet-v1.0"
```

### OOD Images

Place any external knee MRI images (`.jpg`, `.jpeg`, `.png`) into the `OOD_images/` folder and update `OOD_ROOT`:
```python
OOD_ROOT = "/your/path/to/OOD_images"
```

---

## Key Hyperparameters

| Parameter | Single-Plane | Multi-Plane |
|-----------|-------------|-------------|
| Model | ViT-S/16 (timm) | ViT-S/16 (timm, shared) |
| Plane | coronal | sagittal + coronal + axial |
| Slices (N) | 4 | 2 per plane |
| Batch size | 4 | 4 |
| Epochs | 20 | 20 |
| Learning rate | 2e-5 | 2e-5 |
| Weight decay | 1e-4 | 1e-4 |
| Threshold (τ) | 0.45 | 0.50 |
| Loss | Focal Loss (γ=2) | Focal Loss (γ=2) |
| Seed | 42 | 42 |
| Device | CUDA / MPS / CPU (auto) | CUDA / MPS / CPU (auto) |

---

## Running the Notebooks

Run cells **top to bottom** in order. After any kernel restart, use **Restart & Run All** to avoid stale variable state — this is especially important because class definitions (like `ViTMultiLabel`, `FocalBCEWithLogitsLoss`) must be executed before the model setup cell.

Recommended order within `MRNet_ViT.ipynb`:
1. Config & imports
2. `FocalBCEWithLogitsLoss` class definition
3. `OODImageDataset` class definition
4. `ViTMultiLabel` class definition
5. Dataset & DataLoader setup
6. Model setup + training
7. Evaluation & visualisation cells

---

## Architecture

```
MRNet 3-D Volume
      │
      ▼
Slice Sampling (N evenly-spaced slices)
      │
      ▼
ViT-S/16 Backbone (pretrained ImageNet-1k, timm)
  • 196 patch tokens (16×16 patches, 224×224 input)
  • 12 Transformer blocks, 6 heads, 384-dim
  • Frozen except last 2 blocks + classification head
      │
      ▼
Mean-pool over slice dimension  →  (B, 3)
      │
      ▼
Focal Loss (γ=2) · pos_weight per class
      │
      ▼
[Abnormal | ACL | Meniscus]  (multi-label, sigmoid threshold τ)
```

---

## Interpretability

**Attention Rollout** — implemented in `compute_attention_rollout()`:
- Computes per-head attention at each of the 12 blocks
- Averages heads, adds identity residual, multiplies recursively block-by-block
- Produces a 14×14 [CLS]-to-patch saliency map, up-sampled to 224×224

**t-SNE** (Extra Credit) — [CLS] embeddings for all 120 validation exams projected to 2-D via PCA (50 components) → t-SNE (perplexity=30).

---

## Authors

- Yaseen Ahmed
- Alishala Gloryalishala  
- Devisri Prasad
- Muhammad khalid
- Vishal Reddy

University of Surrey — MSc Artificial Intelligence

---

## References

1. Dosovitskiy et al., *An Image Is Worth 16×16 Words*, ICLR 2021
2. Touvron et al., *DeiT*, ICML 2021
3. Bien et al., *MRNet*, PLoS Medicine 2018
4. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
5. Chefer et al., *Transformer Interpretability Beyond Attention Visualization*, CVPR 2021
