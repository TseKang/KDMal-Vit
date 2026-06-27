# KDMal-Vit: Knowledge Distillation with Prompt Learning for Malware Image Classification

[中文版本](README.md)

## Introduction

This project implements malware grayscale image classification based on Vision Transformer (ViT), combining **Prompt Learning** and **Knowledge Distillation** techniques to significantly reduce model inference overhead while maintaining high classification accuracy.

- **Teacher Model**: ViT-Base-Patch16-224
- **Student Model**: DeiT-Tiny-Patch16-224 (learns from teacher via knowledge distillation)
- **Dataset**: [Malimg](https://www.kaggle.com/datasets/manishmalangadan/malimg-malware-dataset) (25 malware families, ~9,339 grayscale images)
- **Framework**: PyTorch + timm

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- torch / torchvision / torchaudio
- timm == 0.5.4
- Pillow == 9.1.1
- scikit-learn
- pandas

## Dataset

Download the Malimg dataset and extract it to the `data/` directory with the following structure:

```
data/
├── train/
│   ├── Adialer.C/
│   ├── Agent.FYI/
│   └── ...
├── val/
│   ├── Adialer.C/
│   └── ...
└── test/
    ├── Adialer.C/
    └── ...
```

## Usage

### 1. Train Teacher Model (Prompt Learning)

```bash
python regenerate_experiment_results.py --data_path ./data --data_set MALIMG --nb_classes 25
```

Generates `checkpoint_prompt_10epoch.pth` after training.

### 2. Knowledge Distillation (Teacher → Student)

```bash
python Distillation.py --data_path ./data --data_set MALIMG --nb_classes 25 --temperature 4.0 --alpha 0.5
```

Distills knowledge from teacher (ViT-Base) to student (DeiT-Tiny). Logs are written to `distillation_log.txt`.

### 3. Evaluation

```bash
python regenerate_experiment_results.py --data_path ./data --is_eval True
```

Evaluation scripts support multiple experiment settings (family / type / binary classification). Results are output to the `evaluation/` directory.

## Project Structure

| File | Description |
|------|-------------|
| `modeling_finetune.py` | ViT model definition based on timm, with integrated Prompt module |
| `prompt.py` | Learnable Prompt embedding layer, generates prompt tokens per class |
| `MALIMG.py` | Malimg dataset loader (supports train/val/test splits) |
| `dataset_folder.py` | Generic image folder dataset class |
| `regenerate_experiment_results.py` | Main training and evaluation script |
| `Distillation.py` | Knowledge distillation training script (ViT → DeiT) |
| `utils.py` | Utility functions (distributed training, metric logging, etc.) |
| `evaluation/` | Evaluation metrics generation scripts for various experiments |
| `data/` | Dataset directory (download required) |

## Method Overview

```
Malware binary → Grayscale image (224×224) → ViT + Prompt → 25-class classification

Knowledge Distillation:
  Teacher (ViT-Base) ──soft labels──→ Student (DeiT-Tiny)
                                       ↑
                        Hard labels (ground truth) joint supervision
```

## License

This project is for academic research purposes only.
