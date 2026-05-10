# Changelog

All notable changes to this project are documented here.

## [1.2.0] - 2025-05-01
### Added
- Extra credit notebook (`MRNet_Extra_Credit.ipynb`) with t-SNE feature-space visualisation
- PCA (50 components) + t-SNE (perplexity=30) applied to [CLS] embeddings from all 120 validation exams
- OOD inference pipeline on external knee MRI images with per-class confidence thresholds
- Error analysis section identifying common failure cases (e.g. borderline meniscal tears)

## [1.1.0] - 2025-04-25
### Added
- Multi-plane fusion notebook (`MRNet_multi_plane.ipynb`)
- Shared ViT-S/16 backbone trained across sagittal, coronal, and axial planes
- Averaged logit fusion strategy for exam-level multi-label prediction
- Comparative AUC evaluation between single-plane and multi-plane models

## [1.0.0] - 2025-04-20
### Added
- Initial ViT-S/16 single-plane classification pipeline (`MRNet_ViT.ipynb`)
- Focal Loss (γ=2) implementation replacing class-weighted BCE for minority-class learning
- Differential learning rate strategy: freeze all layers → unfreeze last 2 blocks + head
- Evaluation suite: accuracy, precision, recall, F1, IoU, and AUC-ROC per condition
- Confusion matrix and AUC-ROC curve visualisations
- Attention rollout visualisation mapping [CLS] token to 14×14 patch saliency
- OOD image folder and inference support with stricter classification thresholds
