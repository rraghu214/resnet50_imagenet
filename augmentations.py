"""
Data augmentation classes using Albumentations
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from PIL import Image
import random

class ImageNetAugmentations:
    """ImageNet augmentation strategies"""
    
    def __init__(self, config):
        self.config = config
        self.img_size = config.img_size
        
        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def get_train_transforms(self, current_size=None):
        """Get training augmentations"""
        size = current_size or self.img_size
        
        transforms = [
            A.RandomResizedCrop(size=(size, size), scale=self.config.crop_scale, ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
        ]
        
        # Color augmentations
        if self.config.color_jitter > 0:
            transforms.append(
                A.ColorJitter(
                    brightness=self.config.color_jitter,
                    contrast=self.config.color_jitter,
                    saturation=self.config.color_jitter,
                    hue=self.config.color_jitter * 0.25,
                    p=0.8
                )
            )
        
        # Additional augmentations  
        transforms.extend([
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(size//8, size//4),
                hole_width_range=(size//8, size//4),
                fill=tuple([int(x * 255) for x in self.mean]),
                fill_mask=None,
                p=0.5
            ),
            A.Affine(
                translate_percent=0.1, 
                scale=(0.9, 1.1), 
                rotate=(-15, 15), 
                p=0.5
            ),
        ])
        
        # AutoAugment (if enabled)
        if self.config.auto_augment:
            auto_aug = AutoAugment(self.config)
            transforms.extend(auto_aug.get_policy().transforms)
        
        # Final normalization and tensor conversion
        transforms.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def get_val_transforms(self):
        """Get validation augmentations"""
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(self.img_size, self.img_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def get_test_transforms(self):
        """Get test augmentations (same as validation)"""
        return self.get_val_transforms()

class MixupCutmix:
    """Mixup and CutMix implementation"""
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        
    def __call__(self, batch, target):
        if random.random() > self.prob:
            return batch, target
            
        if random.random() > 0.5:
            return self.mixup(batch, target)
        else:
            return self.cutmix(batch, target)
    
    def mixup(self, x, y):
        """Apply mixup augmentation"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, (y_a, y_b, lam)
    
    def cutmix(self, x, y):
        """Apply cutmix augmentation"""
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        
        return x, (y_a, y_b, lam)
    
    def rand_bbox(self, size, lam):
        """Generate random bounding box for cutmix"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class AutoAugment:
    """Simple AutoAugment implementation"""
    
    def __init__(self, config):
        self.config = config
        
    def get_policy(self):
        """Get AutoAugment policy for ImageNet"""
        return A.Compose([
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.5),
            
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.ChannelShuffle(),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            ], p=0.5),
            
            A.OneOf([
                A.Blur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ], p=0.3),
        ])
