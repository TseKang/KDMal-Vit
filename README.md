# KDMal-Vit: 基于知识蒸馏与 Prompt Learning 的恶意软件图像分类

## 项目简介

本项目基于 Vision Transformer (ViT) 实现恶意软件灰度图分类，结合 **Prompt Learning** 与 **Knowledge Distillation** 技术，在保持较高分类准确率的同时显著降低模型推理开销。

- **教师模型**: ViT-Base-Patch16-224
- **学生模型**: DeiT-Tiny-Patch16-224（通过知识蒸馏从教师模型学习）
- **数据集**: [Malimg](https://www.kaggle.com/datasets/manishmalangadan/malimg-malware-dataset)（25 类恶意软件家族，约 9,339 张灰度图）
- **框架**: PyTorch + timm

## 环境要求

```bash
pip install -r requirements.txt
```

主要依赖：
- torch / torchvision / torchaudio
- timm == 0.5.4
- Pillow == 9.1.1
- scikit-learn
- pandas

## 数据集准备

将 Malimg 数据集下载并解压到 `data/` 目录下，目录结构如下：

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

## 使用说明

### 1. 训练教师模型（Prompt Learning）

```bash
python regenerate_experiment_results.py --data_path ./data --data_set MALIMG --nb_classes 25
```

训练完成后生成 `checkpoint_prompt_10epoch.pth`。

### 2. 知识蒸馏（教师 → 学生）

```bash
python Distillation.py --data_path ./data --data_set MALIMG --nb_classes 25 --temperature 4.0 --alpha 0.5
```

将教师模型 (ViT-Base) 的知识蒸馏到学生模型 (DeiT-Tiny)，日志输出到 `distillation_log.txt`。

### 3. 评估

```bash
python regenerate_experiment_results.py --data_path ./data --is_eval True
```

评估脚本支持多种实验设置（family / type / binary 分类），结果输出到 `evaluation/` 目录。

## 项目结构

| 文件 | 说明 |
|------|------|
| `modeling_finetune.py` | 基于 timm 的 ViT 模型定义，集成 Prompt 模块 |
| `prompt.py` | 可学习的 Prompt 嵌入层，为每个类别生成 prompt token |
| `MALIMG.py` | Malimg 数据集加载器（支持 train/val/test 划分） |
| `dataset_folder.py` | 通用图像文件夹数据集类 |
| `regenerate_experiment_results.py` | 主训练与评估脚本 |
| `Distillation.py` | 知识蒸馏训练脚本（ViT → DeiT） |
| `utils.py` | 工具函数（分布式训练、指标记录等） |
| `evaluation/` | 各类实验评估指标生成脚本 |
| `data/` | 数据集目录（需自行下载） |

## 方法概述

```
恶意软件二进制 → 灰度图 (224×224) → ViT + Prompt → 25 类分类

知识蒸馏:
  Teacher (ViT-Base) ──软标签──→ Student (DeiT-Tiny)
                                    ↑
                        硬标签 (真实标签) 联合监督
```

## License

本项目仅供学术研究使用。
