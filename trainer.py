"""
Training class with comprehensive training loop
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast,GradScaler
import time
import os
from collections import defaultdict
import numpy as np

class Trainer:
    """Main training class"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, 
                 config, logger, mixup_cutmix=None, model_ema=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.mixup_cutmix = mixup_cutmix
        self.model_ema = model_ema
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.start_epoch = 0
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if config.mixed_precision else None
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Channels last for better performance
        if config.mixed_precision:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # Progressive resizing
        self.progressive_sizes = config.progressive_epochs if config.progressive_resize else None
        self.current_size_idx = 0
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Progressive resizing
        current_size = self._get_current_image_size()
        
        metrics = defaultdict(list)
        total_samples = 0
        correct_samples = 0
        
        # Progress bar setup
        progress_bar = self.logger.get_train_progress_bar(
            self.train_loader, self.current_epoch, self.config.epochs
        )
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Channels last
            if self.config.mixed_precision:
                data = data.to(memory_format=torch.channels_last)
            
            # Apply mixup/cutmix
            if self.mixup_cutmix is not None:
                data, target = self.mixup_cutmix(data, target)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with autocast('cuda'):
                    output = self.model(data)
                    loss = self._calculate_loss(output, target)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self._calculate_loss(output, target)
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                
                self.optimizer.step()
            
            # Update EMA
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            
            # Update scheduler (for OneCycleLR)
            if hasattr(self.scheduler, 'step') and self.config.scheduler == 'onecycle':
                self.scheduler.step()
            
            # Calculate accuracy
            if isinstance(target, tuple):  # Mixup/Cutmix case
                acc = self._calculate_mixup_accuracy(output, target)
            else:
                _, predicted = output.max(1)
                acc = predicted.eq(target).float().mean()
                correct_samples += predicted.eq(target).sum().item()
                total_samples += target.size(0)
            
            # Store metrics
            metrics['loss'].append(loss.item())
            metrics['acc'].append(acc.item() * 100)
            
            # Update progress bar with exact format from requirements
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix_str(f"Loss={loss.item():.4f}, Acc={acc.item() * 100:.2f}%")
            
            # Log batch metrics
            if batch_idx % self.config.log_interval == 0:
                self.logger.log_batch_metrics(
                    self.current_epoch, batch_idx, len(self.train_loader),
                    loss.item(), acc.item() * 100, current_lr
                )
        
        # Calculate epoch metrics
        epoch_loss = np.mean(metrics['loss'])
        epoch_acc = correct_samples / total_samples * 100 if total_samples > 0 else np.mean(metrics['acc'])
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        
        # Use EMA model if available
        if self.model_ema is not None:
            self.model_ema.apply_shadow(self.model)
        
        metrics = defaultdict(list)
        total_samples = 0
        correct_samples = 0
        top5_correct = 0
        
        progress_bar = self.logger.get_val_progress_bar(self.val_loader)
        
        with torch.no_grad():
            for data, target in progress_bar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Channels last
                if self.config.mixed_precision:
                    data = data.to(memory_format=torch.channels_last)
                
                if self.config.mixed_precision:
                    with autocast('cuda'):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Calculate accuracy
                _, predicted = output.max(1)
                correct_samples += predicted.eq(target).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = output.topk(5, 1, True, True)
                top5_correct += top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).sum().item()
                
                total_samples += target.size(0)
                metrics['loss'].append(loss.item())
        
        # Restore original model if EMA was used
        if self.model_ema is not None:
            self.model_ema.restore(self.model)
        
        epoch_loss = np.mean(metrics['loss'])
        epoch_acc = correct_samples / total_samples * 100
        top5_acc = top5_correct / total_samples * 100
        
        return epoch_loss, epoch_acc, top5_acc
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        
        for epoch in range(self.start_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Log epoch start
            self.logger.log_epoch_start(epoch, self.config.epochs)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_top5 = self.validate_epoch()
            
            # Update scheduler (for non-OneCycleLR schedulers)
            if hasattr(self.scheduler, 'step') and self.config.scheduler != 'onecycle':
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Log epoch results
            self.logger.log_epoch_end(
                epoch, self.config.epochs, train_loss, train_acc,
                val_loss, val_acc, val_top5, current_lr, 
                self.best_val_acc, self.best_epoch, is_best
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot progress
            if (epoch + 1) % self.config.plot_interval == 0:
                self.logger.plot_training_curves()
            
            # Progressive resizing
            self._update_progressive_size(epoch)
        
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        
        # Final plot
        self.logger.plot_training_curves()
        
        return self.best_val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        if self.model_ema is not None:
            checkpoint['ema_state_dict'] = self.model_ema.shadow
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        
        if self.model_ema is not None and 'ema_state_dict' in checkpoint:
            self.model_ema.shadow = checkpoint['ema_state_dict']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation accuracy so far: {self.best_val_acc:.2f}%")
    
    def _calculate_loss(self, output, target):
        """Calculate loss (handles mixup/cutmix)"""
        if isinstance(target, tuple):
            # Mixup/Cutmix case
            y_a, y_b, lam = target
            return lam * self.criterion(output, y_a) + (1 - lam) * self.criterion(output, y_b)
        else:
            return self.criterion(output, target)
    
    def _calculate_mixup_accuracy(self, output, target):
        """Calculate accuracy for mixup/cutmix"""
        y_a, y_b, lam = target
        _, predicted = output.max(1)
        acc_a = predicted.eq(y_a).float().mean()
        acc_b = predicted.eq(y_b).float().mean()
        return lam * acc_a + (1 - lam) * acc_b
    
    def _get_current_image_size(self):
        """Get current image size for progressive resizing"""
        if not self.config.progressive_resize:
            return self.config.img_size
        
        # Determine which size to use based on epoch
        epoch_per_size = self.config.epochs // len(self.progressive_sizes)
        size_idx = min(self.current_epoch // epoch_per_size, len(self.progressive_sizes) - 1)
        
        return self.progressive_sizes[size_idx]
    
    def _update_progressive_size(self, epoch):
        """Update image size for progressive training"""
        if not self.config.progressive_resize:
            return
        
        new_size = self._get_current_image_size()
        
        # Update transforms if size changed
        if hasattr(self, '_last_size') and new_size != self._last_size:
            print(f"Updating image size to {new_size}x{new_size}")
            # Here you would update the dataset transforms
            # This is a simplified version - in practice, you'd need to update the dataloaders
        
        self._last_size = new_size
