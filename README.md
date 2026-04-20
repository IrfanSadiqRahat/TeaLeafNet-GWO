# TeaLeafNet-GWO

> **Towards Precision Agriculture: Tea Leaf Disease Detection using CNNs and Image Processing**
>
> Irfan Sadiq Rahat, Hritwik Ghosh, Suresh Dara, Shashi Kant
>
> ***Scientific Reports*** (Nature Portfolio) · Volume 15, Article 17571 · May 2025
>
> DOI: [10.1038/s41598-025-02378-0](https://doi.org/10.1038/s41598-025-02378-0) · IF: 3.9 · Q1

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports-green)](https://doi.org/10.1038/s41598-025-02378-0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Sylhet%20Tea%20Garden-orange)]()

---

## Abstract

This study introduces a novel deep learning model for precise classification of common tea leaf diseases using advanced image analysis. The model features residual blocks, batch normalization, and multiple optimizers, achieving **99% accuracy** on a Sylhet (Bangladesh) tea garden dataset. The Adam optimizer proved exceptionally effective for this domain.

---

## Model Architecture

```
Input (256×256 RGB)
    │
Zero Padding 2D
    │
Conv2D (64 filters, 7×7) → BatchNorm → ReLU
    │
MaxPool2D
    │
Residual Block × 3  ←── skip connections
    │
GlobalAvgPool2D
    │
Dense (512) → Dropout(0.5)
    │
Dense (num_classes) → Softmax
```

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | **99.0%** |
| Precision | 98.7% |
| Recall | 98.9% |
| F1-Score | 98.8% |

---

## Dataset

Tea leaf images from Sylhet, Bangladesh covering 5 disease classes:
- Anthracnose
- Algal leaf spot
- Bird eye spot
- Brown blight
- Healthy

**Download:** Available on request / Kaggle (see paper for details)

---

## Setup

```bash
git clone https://github.com/IrfanSadiqRahat/TeaLeafNet-GWO.git
cd TeaLeafNet-GWO
pip install -r requirements.txt
python train.py --data_dir data/tea_leaf --epochs 50
python evaluate.py --checkpoint outputs/best_model.pth
```

---

## Citation

```bibtex
@article{rahat2025tealeaf,
  title={Towards precision agriculture tea leaf disease detection using CNNs and image processing},
  author={Rahat, Irfan Sadiq and Ghosh, Hritwik and Dara, Suresh and Kant, Shashi},
  journal={Scientific Reports},
  volume={15},
  pages={17571},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-025-02378-0}
}
```
