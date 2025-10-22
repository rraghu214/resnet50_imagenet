"""
Configuration file for ResNet50 ImageNet training
"""
import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Config:
    # Model Configuration
    model_name: str = "resnet50"  # "resnet50" or "resnet50d"
    num_classes: int = 1000
    
    # Dataset Configuration
    dataset: str = "imagenet1k"  # "imagenet1k", "imagenet100", "tiny_imagenet"
    data_dir: str = "./data"
    
    # Training Configuration
    epochs: int = 200
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    
    # Optimizer Configuration
    optimizer: str = "adamw"  # "adamw", "sgd"
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    momentum: float = 0.9  # for SGD
    
    # Scheduler Configuration
    scheduler: str = "onecycle"  # "onecycle", "cosine", "step"
    max_lr: float = 1e-3
    pct_start: float = 0.3
    div_factor: float = 25
    final_div_factor: float = 10000
    
    # Augmentation Configuration
    img_size: int = 224
    crop_scale: Tuple[float, float] = (0.08, 1.0)
    color_jitter: float = 0.4
    auto_augment: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Training Options
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    label_smoothing: float = 0.1
    model_ema: bool = True
    ema_decay: float = 0.9999
    
    # Progressive Training
    progressive_resize: bool = False
    progressive_epochs: list = None  # [160, 192, 224] sizes
    
    # Logging and Checkpointing
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    log_interval: int = 100
    save_interval: int = 10
    plot_interval: int = 10
    
    # Platform specific
    platform: str = "local"  # "local", "colab", "ec2"
    device: str = "cuda"
    
    # EC2 EBS Configuration (for optimized data loading)
    ebs_snapshot_id: Optional[str] = None
    ebs_volume_size: int = 500  # GB
    ebs_volume_type: str = "gp3"
    ebs_iops: int = 4000
    ebs_throughput: int = 500  # MB/s
    
    def __post_init__(self):
        """Initialize platform-specific configurations"""
        if self.platform == "colab":
            self.data_dir = "/content/drive/MyDrive/datasets"
            self.log_dir = "/content/drive/MyDrive/logs"
            self.save_dir = "/content/drive/MyDrive/checkpoints"
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Progressive resize epochs
        if self.progressive_resize and self.progressive_epochs is None:
            self.progressive_epochs = [160, 192, 224]

# Dataset mappings
DATASET_INFO = {
    "imagenet1k": {
        "num_classes": 1000,
        "train_samples": 1281167,
        "val_samples": 50000,
        "url": None  # Download manually or use torchvision
    },
    "imagenet100k": {
        "num_classes": 100,
        "train_samples": 128116,  # Approx 1/10th
        "val_samples": 5000,
        "url": None  # Create subset
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "train_samples": 100000,
        "val_samples": 10000,
        "url": "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    }
}
