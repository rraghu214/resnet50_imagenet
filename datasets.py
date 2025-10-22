"""
Dataset classes for ImageNet variants
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import json
import zipfile
import requests
from typing import Optional, Tuple, List
import random

class ImageNetDataset:
    """ImageNet dataset handler"""
    
    def __init__(self, config, augmentations):
        self.config = config
        self.augmentations = augmentations
        self.data_dir = config.data_dir
        
    def get_datasets(self):
        """Get train and validation datasets"""
        if self.config.dataset == "imagenet1k":
            return self._get_imagenet1k()
        elif self.config.dataset == "imagenet100k":
            return self._get_imagenet100k()
        elif self.config.dataset == "tiny_imagenet":
            return self._get_tiny_imagenet()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
    
    def _get_imagenet1k(self):
        """Get full ImageNet-1K dataset"""
        train_dir = os.path.join(self.data_dir, "imagenet", "train")
        val_dir = os.path.join(self.data_dir, "imagenet", "val")
        
        if not os.path.exists(train_dir):
            print("ImageNet-1K not found. Please download manually.")
            print("Download from: https://image-net.org/download.php")
            raise FileNotFoundError("ImageNet-1K dataset not found")
        
        train_dataset = datasets.ImageFolder(
            train_dir,
            transform=self.augmentations.get_train_transforms()
        )
        
        val_dataset = datasets.ImageFolder(
            val_dir,
            transform=self.augmentations.get_val_transforms()
        )
        
        return train_dataset, val_dataset
    
    def _get_imagenet100k(self):
        """Get ImageNet-100 subset"""
        # First try to get full ImageNet, then create subset
        try:
            full_train, full_val = self._get_imagenet1k()
        except FileNotFoundError:
            print("Creating ImageNet-100 from torchvision...")
            return self._create_imagenet100_from_torchvision()
        
        # Create subset with first 100 classes
        train_subset = self._create_class_subset(full_train, num_classes=100)
        val_subset = self._create_class_subset(full_val, num_classes=100)
        
        return train_subset, val_subset
    
    def _create_imagenet100_from_torchvision(self):
        """Create ImageNet-100 using torchvision's subset"""
        # This is a placeholder - in practice, you'd download a subset
        # For now, we'll use a synthetic dataset for testing
        print("Using synthetic ImageNet-100 for testing...")
        
        train_dataset = SyntheticImageNet(
            num_samples=50000, 
            num_classes=100,
            transform=self.augmentations.get_train_transforms()
        )
        
        val_dataset = SyntheticImageNet(
            num_samples=5000, 
            num_classes=100,
            transform=self.augmentations.get_val_transforms()
        )
        
        return train_dataset, val_dataset
    
    def _get_tiny_imagenet(self):
        """Get Tiny ImageNet dataset"""
        tiny_dir = os.path.join(self.data_dir, "tiny-imagenet-200")
        
        if not os.path.exists(tiny_dir):
            self._download_tiny_imagenet()
        
        train_dataset = TinyImageNetDataset(
            os.path.join(tiny_dir, "train"),
            transform=self.augmentations.get_train_transforms()
        )
        
        val_dataset = TinyImageNetDataset(
            os.path.join(tiny_dir, "val"),
            transform=self.augmentations.get_val_transforms(),
            is_val=True
        )
        
        return train_dataset, val_dataset
    
    def _download_tiny_imagenet(self):
        """Download Tiny ImageNet dataset"""
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(self.data_dir, "tiny-imagenet-200.zip")
        
        print("Downloading Tiny ImageNet...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        os.remove(zip_path)
        print("Tiny ImageNet downloaded and extracted.")
    
    def _create_class_subset(self, dataset, num_classes):
        """Create subset with specified number of classes"""
        class_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Select first N classes
        selected_classes = sorted(class_indices.keys())[:num_classes]
        subset_indices = []
        
        for class_idx in selected_classes:
            subset_indices.extend(class_indices[class_idx])
        
        return Subset(dataset, subset_indices)

class TinyImageNetDataset(Dataset):
    """Tiny ImageNet dataset implementation"""
    
    def __init__(self, root_dir, transform=None, is_val=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_val = is_val
        
        self.samples = []
        self.class_to_idx = {}
        
        if is_val:
            self._load_val_data()
        else:
            self._load_train_data()
    
    def _load_train_data(self):
        """Load training data"""
        class_dirs = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for i, class_dir in enumerate(sorted(class_dirs)):
            self.class_to_idx[class_dir] = i
            class_path = os.path.join(self.root_dir, class_dir, "images")
            
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, i))
    
    def _load_val_data(self):
        """Load validation data"""
        # Read validation annotations
        val_annotations = os.path.join(self.root_dir, "val_annotations.txt")
        
        if os.path.exists(val_annotations):
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_name = parts[0], parts[1]
                    
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = len(self.class_to_idx)
                    
                    img_path = os.path.join(self.root_dir, "images", img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        return image, label

class SyntheticImageNet(Dataset):
    """Synthetic ImageNet dataset for testing"""
    
    def __init__(self, num_samples, num_classes, transform=None, img_size=224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform
        self.img_size = img_size
        
        # Generate random data
        self.data = []
        for i in range(num_samples):
            label = random.randint(0, num_classes - 1)
            self.data.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image
        image = np.random.randint(0, 256, (self.img_size, self.img_size, 3), dtype=np.uint8)
        label = self.data[idx]
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label

def create_dataloaders(config, augmentations):
    """Create train and validation dataloaders"""
    dataset_handler = ImageNetDataset(config, augmentations)
    train_dataset, val_dataset = dataset_handler.get_datasets()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    return train_loader, val_loader

# EBS and EC2 integration for optimized data loading
def setup_ec2_data_volume(data_dir="/data/imagenet"):
    """Setup data volume on EC2 instance"""
    import subprocess
    
    # Check if volume is already mounted
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"Data volume already mounted at {data_dir}")
        return True
    
    # Mount the data volume (assumes /dev/xvdf is attached)
    try:
        # Create mount point
        os.makedirs(data_dir, exist_ok=True)
        
        # Mount the volume
        subprocess.run(['sudo', 'mount', '/dev/xvdf', data_dir], check=True)
        
        # Verify mount
        if os.path.exists(os.path.join(data_dir, "train")) and os.path.exists(os.path.join(data_dir, "val")):
            print(f"Data volume successfully mounted at {data_dir}")
            return True
        else:
            print(f"Warning: Data volume mounted but ImageNet structure not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error mounting data volume: {e}")
        return False

def verify_imagenet_structure(data_dir):
    """Verify ImageNet directory structure"""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        return False, "Missing train or val directories"
    
    # Check number of classes
    train_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    val_classes = len([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    
    if train_classes != 1000 or val_classes != 1000:
        return False, f"Expected 1000 classes, found {train_classes} train, {val_classes} val"
    
    return True, f"Valid ImageNet structure: {train_classes} classes"

def optimize_ec2_data_loading(config):
    """Optimize data loading for EC2"""
    if config.platform == "ec2":
        # Setup data volume
        if not setup_ec2_data_volume(config.data_dir):
            raise RuntimeError("Failed to setup EC2 data volume")
        
        # Verify structure
        valid, message = verify_imagenet_structure(config.data_dir)
        if not valid:
            raise RuntimeError(f"Invalid ImageNet structure: {message}")
        
        print(f"✅ {message}")
        
        # Optimize for EC2
        config.num_workers = min(8, os.cpu_count())  # Use available CPU cores
        config.pin_memory = True
        config.persistent_workers = True
        
        print(f"Optimized EC2 data loading: {config.num_workers} workers")

# Original functions continue here...
# ...existing code...
def setup_colab_dataset(config):
    """Setup dataset for Google Colab"""
    if config.platform == "colab":
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully")

            #Ensure the dataset diurectory exists
            os.makedirs(config.data_dir, exist_ok=True)
            print(f"Data directory setup {config.data_dir}")

        except ImportError:
            print("Not running on Google Colab")
            return False        
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")
            return False
    return True

