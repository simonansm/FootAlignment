# Foot Alignment Classification

A deep learning project for classifying foot alignment into three categories using a fusion of plantar pressure images and morphological measurements.

## Overview

This project implements deep learning models that classify foot alignment based on:
- **Plantar pressure images**: Computer vision features extracted from foot pressure scans
- **Morphological measurements**: 19 quantitative measurements of foot anatomy

The system focuses on three foot regions:
- **Arch**: Classification of arch height/alignment
- **Calf**: Classification of calf alignment  
- **Heel**: Classification of heel alignment

## Data Split

To prevent data leakage, the dataset is split into three non-overlapping sets using stratified random splitting with `random_state=20`:
- **Training set**: 60% of data (600 samples)
- **Validation set**: 20% of data (200 samples)
- **Test set**: 20% of data (200 samples)

The same split indices are used consistently across all regions to ensure proper evaluation. This split is created once during initial data loading and reused for fair model comparison.

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

# Train calf model
python training/train_calf.py

# Train heel model
python training/train_heel.py
```

Each training script will:
- Load and preprocess data
- Apply stratified train/val/test split (60/20/20)
- Train the model with early stopping
- Save the best model to `checkpoints/best_model_inception_{region}.pt`

**Note**: Training takes several hours per region on a GPU. Adjust `NUM_EPOCHS` in `config.py` for faster iteration.

### Evaluation

Evaluate trained models on the held-out test set:

```bash
# Evaluate arch model
python evaluation/eval_arch.py

# Evaluate calf model
python evaluation/eval_calf.py

# Evaluate heel model  
python evaluation/eval_heel.py
```

Each evaluation script will:
- Load the best trained model
- Collect predictions on the full test set
- Run bootstrap resampling (100 iterations, 95% CI) for confidence intervals
- Print detailed per-class and macro-averaged metrics

## Configuration

Edit `config.py` to modify:
- `BATCH_SIZE`: Training batch size (default: 32)
- `NUM_EPOCHS`: Maximum training epochs (default: 1000)
- `LEARNING_RATE`: Learning rate for AdamW optimizer (default: 3e-5)
- `WEIGHT_DECAY`: L2 regularization (default: 1e-3)
- `EARLY_STOPPING_PATIENCE`: Epochs without improvement before stopping (default: 150)
- `N_BOOTSTRAP`: Number of bootstrap resamples for evaluation (default: 100)

## Models

### Available Architectures

All models follow a two-stream fusion architecture:

1. **Inception V3 Attention Fusion** (default)
   - Pre-trained Inception V3 backbone
   - MLP for morphological features
   - Attention-weighted fusion
   - Best balance of accuracy and efficiency

2. **ResNet Attention Fusion**
   - Pre-trained ResNet backbone
   - Alternative backbone option

3. **Vision Transformer (ViT) Attention Fusion**
   - Transformer-based feature extraction
   - Alternative modern architecture

4. **EfficientNet Attention Fusion**
   - Lightweight architecture
   - Improved efficiency vs. accuracy trade-off

### Model Training Details

- **Optimizer**: AdamW with weight decay
- **Loss**: Cross-entropy with class weights (balanced)
- **Data augmentation**: Random horizontal flip, random rotation (10°)
- **Scheduler**: Adaptive learning rate via early stopping
- **Input size**: 224×224 images

## Data Leakage Prevention

This project explicitly prevents data leakage through:

1. **Stratified Random Split**: Using `random_state=20` ensures reproducible splits
   - Same split used for all regions for consistency
   - Stratification preserves class distribution

2. **Separate Sets**: No overlap between train/val/test
   - Training: Used for model optimization
   - Validation: Used for early stopping and hyperparameter selection
   - Test: Used only for final evaluation and reporting

3. **No Information Leakage**: 
   - Preprocessing parameters are computed only on training set
   - No test set statistics used during training
   - Bootstrap evaluation uses only test predictions

## Evaluation Metrics

The evaluation reports the following metrics:

### Macro-Averaged (averaged across all classes)
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall (Sensitivity)**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC (OvR)**: Area under ROC curve (One-vs-Rest)
- **AP (OvR)**: Average Precision curve (One-vs-Rest)

### Per-Class Metrics
- Individual precision, recall, F1, specificity, and AUC for each class
- Confusion matrix
- Classification report

### Bootstrap Confidence Intervals
- 95% confidence intervals on all metrics
- Mean and standard deviation from 100 bootstrap resamples
- More robust uncertainty estimates than single-run evaluation

## Performance

Expected performance on test set (with Inception V3):
- **Arch**: ~75-85% macro F1
- **Calf**: ~75-85% macro F1  
- **Heel**: ~75-85% macro F1

Results will vary based on dataset, hyperparameters, and random seed.

## Troubleshooting

### GPU Memory Issues
- Reduce `BATCH_SIZE` in `config.py`
- Use a different model architecture (EfficientNet is lighter)

### Slow Training
- Reduce `NUM_EPOCHS`
- Increase `BATCH_SIZE` if GPU memory allows
- Ensure GPU is being used: check device printout

### File Not Found Errors
- Verify `data/foot_3class.csv` exists
- Verify `archive/Plantar_dataset/` directory structure
- Check that paths are relative to the `FootAlignment/` root directory

### Model Loading Errors
- Ensure checkpoint files are in `checkpoints/` directory
- Try deleting old checkpoints and retraining

## Future Improvements

- Multi-task learning across all three regions
- Attention visualization (CAM/Grad-CAM)
- Transfer learning from similar datasets
- Ensemble methods
- Uncertainty quantification

## Citation

If you use this code, please cite:
```
@software{footalignment2024,
  title={Foot Alignment Classification},
  author={},
  year={2024},
  url={}
}
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub.
