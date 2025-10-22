# ResNet50 ImageNet Training

A comprehensive, modular PyTorch implementation for training ResNet50 from scratch on ImageNet with the goal of achieving >78% validation accuracy.

## 🎯 Project Goal
Train ResNet50 from scratch on ImageNet to achieve **>78% validation accuracy** using cost-effective cloud resources.

## 🏗️ Architecture

The codebase is designed with modularity in mind:

- **`config.py`** - Configuration management for different platforms and datasets
- **`augmentations.py`** - Data augmentation strategies using Albumentations
- **`models.py`** - ResNet50 and ResNet50-D model implementations
- **`datasets.py`** - Dataset handlers for ImageNet variants
- **`optimizers.py`** - Optimizer and scheduler factories with advanced features
- **`trainer.py`** - Main training loop with mixed precision and advanced features
- **`logger.py`** - Comprehensive logging and visualization
- **`main.py`** - Main orchestration script

## 🚀 Features

### Modern Training Techniques
- **Mixed Precision Training** with AMP for 2x speedup
- **Model EMA** (Exponential Moving Average) for better generalization
- **OneCycleLR** scheduler for faster convergence
- **Mixup/CutMix** augmentation for improved regularization
- **Label Smoothing** for better calibration
- **Gradient Clipping** for stable training

### Advanced Augmentations
- **Albumentations** pipeline with:
  - Random resized crop
  - Color jitter
  - CoarseDropout (like CutOut)
  - Rotation and geometric transforms
  - Normalization

### Multi-Platform Support
- **Local**: Test on your RTX 2070 Max-Q
- **Google Colab**: Validate with T4 GPU
- **AWS EC2**: Full training on g4dn.2xlarge spot instances

### Dataset Flexibility
- **ImageNet-1K**: Full dataset (1000 classes)
- **ImageNet-100**: Subset for faster testing
- **Tiny ImageNet**: Quick prototyping (200 classes, 64x64 images)

### Comprehensive Logging
- **Real-time progress bars** with detailed metrics
- **Training curves** plotted every 10 epochs
- **Detailed epoch logs** with timestamps
- **JSON metrics** export for analysis
- **Checkpoint management** with best model saving

## 📋 Installation

### 1. Clone and Setup
```bash
git clone <repository>
cd resnet50-imagenet
pip install -r requirements.txt
```

### 2. Platform-Specific Setup

#### Local Machine (Windows/Linux)
```bash
# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Google Colab
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install additional packages
!pip install albumentations
```

#### AWS EC2
```bash
# Install AWS CLI and configure
pip install boto3
aws configure
```

## 🎮 Usage

### Quick Start (Local Testing)
```bash
# Test with ImageNet-100 subset on local machine
python main.py --dataset imagenet100 --platform local --epochs 50 --batch-size 128
```

### Google Colab
```bash
# Run on Colab with Tiny ImageNet
python main.py --dataset tiny_imagenet --platform colab --epochs 100 --batch-size 256
```

### AWS EC2 (Full Training)
```bash
# Full ImageNet-1K training
python main.py --dataset imagenet1k --platform ec2 --epochs 200 --batch-size 256 --model resnet50d
```

### Advanced Options
```bash
# With learning rate finder
python main.py --dataset imagenet100 --lr-finder --epochs 100

# Resume from checkpoint
python main.py --resume ./checkpoints/latest_checkpoint.pth

# Evaluation only
python main.py --eval-only ./checkpoints/best_checkpoint.pth
```

## 📊 Expected Results

### Performance Timeline
- **Local Testing** (ImageNet-100): ~65-70% accuracy in 2-3 hours
- **Colab Validation** (Tiny ImageNet): ~60-65% accuracy in 4-6 hours  
- **EC2 Full Training** (ImageNet-1K): **>78% target** in 2-3 days

### Cost Estimation
- **Local**: Free (electricity cost only)
- **Colab**: Free tier available
- **EC2 g4dn.2xlarge spot**: **$15-25 for full training** (optimized)

## 🔧 Configuration

### Key Configuration Options

```python
# Model
model_name = "resnet50"  # or "resnet50d" for better performance
num_classes = 1000

# Training
epochs = 200
batch_size = 256
mixed_precision = True
model_ema = True

# Optimizer
optimizer = "adamw"
learning_rate = 1e-3
scheduler = "onecycle"

# Augmentation
mixup_alpha = 0.2
cutmix_alpha = 1.0
label_smoothing = 0.1
```

### Platform Configurations

#### Local (RTX 2070 Max-Q)
- Batch size: 128-256
- Mixed precision: Enabled
- Dataset: ImageNet-100 for testing

#### Google Colab (T4)
- Batch size: 128-256
- Time limit: ~12 hours
- Auto-save to Google Drive

#### AWS EC2 (g4dn.2xlarge)
- Batch size: 256-512
- Spot instance with checkpointing
- EBS snapshots for optimized data storage

## 📈 Monitoring

### Real-time Monitoring
The training provides detailed real-time monitoring:

```
[START 2025-10-18 06:59:17] Epoch 53/100
Epoch 53: 100% 391/391 [01:10<00:00,  5.54it/s, Loss=1.8894, Acc=65.05%]
Validating: 100% 79/79 [00:04<00:00, 19.12it/s]
Epoch 53/100:
  Train Loss: 1.8894, Train Acc: 65.05%
  Val Loss: 1.6333, Val Acc: 72.79%
  Learning Rate: 0.075640
  Best Val Acc: 72.79% (Epoch 53)
New best model saved with validation accuracy: 72.79%
[END   2025-10-18 07:00:33] Epoch 53/100
--------------------------------------------------
```

### Generated Plots
- Training/validation loss curves
- Accuracy progression (Top-1 and Top-5)
- Learning rate schedule
- Training summary statistics

## 🎯 Achieving 78% Accuracy

### Strategy
1. **Start Local**: Test code with ImageNet-100 subset
2. **Validate on Colab**: Run longer training with subset
3. **Full Training on EC2**: Deploy to cloud for full dataset

### Key Factors for Success
- **Modern training recipe**: AdamW + OneCycleLR + Mixed Precision
- **Strong augmentations**: Mixup/CutMix + Albumentations
- **Model improvements**: Optional ResNet-D for +1-2% boost
- **Proper regularization**: Label smoothing + Model EMA

### Fallback Options
If 78% isn't achieved with standard ResNet50:
- Switch to ResNet50-D (+1-2% accuracy)
- Extend training to 300 epochs
- Add knowledge distillation
- Use test-time augmentation

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python main.py --batch-size 128

# Or enable gradient accumulation (modify config.py)
```

#### Slow Data Loading
```bash
# Increase number of workers
# Modify config.py: num_workers = 8
```

#### Low Accuracy
```bash
# Run learning rate finder
python main.py --lr-finder

# Try ResNet-D variant
python main.py --model resnet50d
```

## 📁 Output Structure

```
logs/run_20251018_120000/
├── training_log.txt              # Detailed training log
├── metrics.json                  # Training metrics in JSON
├── training_curves_latest.png    # Latest training curves
├── training_curves_epoch_XXX.png # Periodic curve snapshots
└── training_summary.json         # Final training summary

checkpoints/
├── latest_checkpoint.pth         # Latest model state
└── best_checkpoint.pth          # Best model based on validation
```

## 🌟 Advanced Features

### Learning Rate Finder
Automatically find optimal learning rate:
```bash
python main.py --lr-finder
```

### Progressive Resizing
Train with smaller images initially for faster convergence:
```python
progressive_resize = True
progressive_epochs = [160, 192, 224]  # Image sizes
```

## 💾 EBS Snapshot Strategy (Cost-Optimized)

Instead of using S3, we use **EBS snapshots** for much better performance and convenience:

### Why EBS Snapshots > S3:
- **Instant availability**: Mount and use immediately (no download time)
- **No transfer costs**: Within same availability zone
- **Incremental backups**: Only pay for changed blocks (~$7.50/month)
- **High-performance**: Direct attachment with 4000 IOPS

### One-Time Data Preparation:
```bash
# 1. Create data preparation instance (t3.medium ~ $0.04/hour)
python aws_utils.py create_prep_instance

# 2. Download and prepare ImageNet (runs offline, ~6 hours)
# SSH into prep instance:
python aws_utils.py download_imagenet
python aws_utils.py organize_imagenet

# 3. Create snapshot (one-time cost, permanent storage)
python aws_utils.py create_snapshot
# Result: Snapshot ID (e.g., snap-1234567890abcdef0)
```

### Training Instance Launch:
```bash
# Launch training instance with data volume from snapshot
python aws_utils.py create_training_instance --snapshot-id snap-1234567890abcdef0 --spot
# Data is instantly available, no download time!
```

## 💰 Realistic Cost Breakdown

### Optimized EC2 Training Cost:
- **Spot instance** (g4dn.2xlarge): $0.25-0.30/hour
- **Training duration**: 48-72 hours (200 epochs)
- **Total compute**: **$12-22**
- **Storage**: 500GB EBS for 3 days = **$4**
- **Snapshot**: 150GB permanent storage = **$7.50/month**

**Total training cost: $15-25** ✅

### Cost Optimization Tips:
1. **Aggressive spot bidding**: Set max price $0.35/hour
2. **Efficient data pipeline**: EBS snapshots eliminate download time
3. **Mixed precision**: 2x faster training
4. **Auto-checkpointing**: Resume from interruptions seamlessly

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License.

---

**Goal**: Achieve >78% ImageNet validation accuracy with ResNet50 from scratch using cost-effective cloud training! 🎯
