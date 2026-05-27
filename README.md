# Development and validation of a multimodal deep learning model for lower-limb alignment screening using plantar 3D foot scans

## Abstract
Lower-limb alignment influences how forces are distributed during standing and walking, and abnormal alignment is associated with common conditions such as knee osteoarthritis and foot pain, yet routine assessment often relies on radiographs or specialist examination, which limits scalable early screening. To address this, we collected weight-bearing plantar 3D foot scans from 167 participants with clinical labels for arch type, hindfoot alignment, and frontal-plane knee alignment, and developed a computer model that combines information from plantar scan images and quantitative foot-shape measurements to classify these alignment outcomes. We found that integrating both sources of information achieved the best performance across all three tasks, and the model consistently highlighted clinically relevant plantar regions linked to alignment patterns. These findings indicate that weight-bearing plantar 3D scanning can serve as a non-invasive digital marker of proximal lower-limb alignment and may support scalable screening to enable earlier orthotic or preventive interventions for alignment-related biomechanical risk.

<img width="3033" height="1660" alt="graphical_abstract_v3" src="https://github.com/user-attachments/assets/c219e4ed-5fc1-49f7-bc8c-a5f2b6276206" />


## Overview

This project implements deep learning models that classify foot alignment based on:
- **Plantar pressure images**: Computer vision features extracted from foot pressure scans
- **Morphological measurements**: 16 quantitative measurements of foot anatomy

The system focuses on three foot regions:
- **Arch**: Classification of arch height/alignment
- **Calf**: Classification of calf alignment  
- **Heel**: Classification of heel alignment



## Project Structure

```
FootAlignment/
├── README.md
├── requirements.txt
├── config.py                 # Global configuration
├── dataloaders/
│   ├── dataloader_arch.py    # Data loader for arch region
│   ├── dataloader_calf.py    # Data loader for calf region
│   └── dataloader_heel.py    # Data loader for heel region
├── models/
│   ├── inception_v3_attention.py        # Inception V3 with attention fusion
│   ├── resnet_model.py                  # ResNet with attention fusion
│   ├── ViTWithAttentionFusion.py        # Vision Transformer with attention fusion
│   └── efficientnet_attention_fusion.py # EfficientNet with attention fusion
├── training/
│   ├── train_arch.py         # Training script for arch region
│   ├── train_calf.py         # Training script for calf region
│   └── train_heel.py         # Training script for heel region
├── evaluation/
│   ├── eval_arch.py          # Evaluation script for arch region
│   ├── eval_calf.py          # Evaluation script for calf region
│   └── eval_heel.py          # Evaluation script for heel region
├── utils/
│   ├── ml_class_experiment_utils.py    # Classification utilities
│   └── dl_evaluation_utils.py          # Deep learning evaluation utilities
└── data/
    └── foot_3class.csv       # Input dataset (to be placed here)
```

## Installation

### Requirements
- Python >= 3.8
- CUDA-capable GPU (recommended, CPU will be much slower)

### Setup

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place `foot_3class.csv` in the `data/` directory
   - Place the `Plantar_dataset/` folder containing images in the `archive/` directory
   
   Expected structure:
   ```
   FootAlignment/
   ├── data/
   │   └── foot_3class.csv
   └── archive/
       └── Plantar_dataset/
           ├── sample_1_Orig.png
           ├── sample_2_Orig.png
           └── ...
   ```

## Quick Start

### Training

Train models for all three regions:

```bash
# Train arch model
python training/train_arch.py

```

### Evaluation

Evaluate trained models on the held-out test set:

```bash
# Evaluate arch model
python evaluation/eval_arch.py

```



