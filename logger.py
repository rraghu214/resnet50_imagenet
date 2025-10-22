"""
Logging and visualization utilities
"""
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import torch

class Logger:
    """Training logger with progress bars and visualization"""
    
    def __init__(self, config):
        self.config = config
        self.log_dir = config.log_dir
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Log file
        self.log_file = os.path.join(self.log_dir, 'training_log.txt')
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Configuration:\n{self._format_config()}\n")
            f.write("=" * 80 + "\n\n")
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _format_config(self):
        """Format configuration for logging"""
        config_dict = vars(self.config)
        formatted = []
        for key, value in config_dict.items():
            formatted.append(f"  {key}: {value}")
        return "\n".join(formatted)
    
    def log_epoch_start(self, epoch, total_epochs):
        """Log epoch start"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[START {timestamp}] Epoch {epoch + 1}/{total_epochs}"
        
        print(message)
        self._write_to_log(message)
    
    def log_epoch_end(self, epoch, total_epochs, train_loss, train_acc, 
                     val_loss, val_acc, val_top5, lr, best_val_acc, best_epoch, is_best):
        """Log epoch end with comprehensive metrics"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Store metrics
        self.metrics['epochs'].append(epoch + 1)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['val_top5'].append(val_top5)
        self.metrics['learning_rates'].append(lr)
        
        # Format message exactly as requested
        message = f"""Epoch {epoch + 1}/{total_epochs}:
  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%
  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%
  Val Top-5 Acc: {val_top5:.2f}%
  Learning Rate: {lr:.6f}
  Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch + 1})"""
        
        if is_best:
            message += f"\nNew best model saved with validation accuracy: {val_acc:.2f}%"
        
        end_message = f"[END   {timestamp}] Epoch {epoch + 1}/{total_epochs}"
        separator = "-" * 50
        
        print(message)
        if is_best:
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        print(end_message)
        print(separator)
        
        # Write to log file
        full_message = f"{message}\n{end_message}\n{separator}\n"
        self._write_to_log(full_message)
        
        # Save metrics to JSON
        self._save_metrics()
    
    def log_batch_metrics(self, epoch, batch_idx, total_batches, loss, acc, lr):
        """Log batch-level metrics"""
        if batch_idx % self.config.log_interval == 0:
            message = f"Epoch {epoch + 1}, Batch {batch_idx}/{total_batches}: Loss={loss:.4f}, Acc={acc:.2f}%, LR={lr:.6f}"
            self._write_to_log(message)
    
    def get_train_progress_bar(self, dataloader, epoch, total_epochs):
        """Get training progress bar with real-time Loss/Acc display"""
        return tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            unit="it/s",
            ncols=120,
            leave=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]'
        )
    
    def get_val_progress_bar(self, dataloader):
        """Get validation progress bar"""
        return tqdm(
            dataloader,
            desc="Validating",
            unit="it/s", 
            ncols=120,
            leave=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def plot_training_curves(self):
        """Plot training curves"""
        if len(self.metrics['epochs']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        epochs = self.metrics['epochs']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.metrics['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.metrics['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics['val_top5'], label='Val Top-5', linewidth=2)
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.metrics['learning_rates'], linewidth=2, color='red')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training summary
        axes[1, 1].axis('off')
        summary_text = f"""Training Summary:
        
Current Epoch: {epochs[-1]}
Best Val Acc: {max(self.metrics['val_acc']):.2f}%
Best Val Top-5: {max(self.metrics['val_top5']):.2f}%
Current LR: {self.metrics['learning_rates'][-1]:.2e}

Latest Metrics:
Train Acc: {self.metrics['train_acc'][-1]:.2f}%
Val Acc: {self.metrics['val_acc'][-1]:.2f}%
Train Loss: {self.metrics['train_loss'][-1]:.4f}
Val Loss: {self.metrics['val_loss'][-1]:.4f}"""
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, f'training_curves_epoch_{epochs[-1]:03d}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Save latest plot
        latest_plot_path = os.path.join(self.log_dir, 'training_curves_latest.png')
        plt.savefig(latest_plot_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
    
    def plot_lr_finder_results(self, lrs, losses):
        """Plot learning rate finder results"""
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, linewidth=2)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        plt.grid(True, alpha=0.3)
        
        # Find optimal LR (minimum loss with smooth region)
        min_idx = losses.index(min(losses))
        optimal_lr = lrs[min_idx] / 10  # Typically use 1/10th of min loss LR
        
        plt.axvline(x=optimal_lr, color='red', linestyle='--', 
                   label=f'Suggested LR: {optimal_lr:.2e}')
        plt.legend()
        
        # Save plot
        lr_plot_path = os.path.join(self.log_dir, 'lr_finder_results.png')
        plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"LR finder results saved to {lr_plot_path}")
        print(f"Suggested learning rate: {optimal_lr:.2e}")
        
        return optimal_lr
    
    def _write_to_log(self, message):
        """Write message to log file"""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_metrics(self):
        """Load metrics from JSON file"""
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
    
    def get_summary_stats(self):
        """Get training summary statistics"""
        if not self.metrics['epochs']:
            return "No training data available"
        
        stats = {
            'total_epochs': len(self.metrics['epochs']),
            'best_val_acc': max(self.metrics['val_acc']),
            'best_val_top5': max(self.metrics['val_top5']),
            'final_train_acc': self.metrics['train_acc'][-1],
            'final_val_acc': self.metrics['val_acc'][-1],
            'final_train_loss': self.metrics['train_loss'][-1],
            'final_val_loss': self.metrics['val_loss'][-1]
        }
        
        return stats

class ProgressBarManager:
    """Enhanced progress bar management"""
    
    def __init__(self, config):
        self.config = config
        
    def create_epoch_bar(self, total_epochs):
        """Create epoch-level progress bar"""
        return tqdm(
            total=total_epochs,
            desc="Training Progress",
            unit="epoch",
            ncols=100,
            position=0
        )
    
    def create_batch_bar(self, dataloader, epoch, total_epochs):
        """Create batch-level progress bar with detailed metrics"""
        return tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            unit="batch",
            ncols=120,
            position=1,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

# Utility functions for logging
def setup_logging_directory(config):
    """Setup logging directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    config.log_dir = log_dir
    return config

def log_system_info(logger):
    """Log system information"""
    info = f"""System Information:
CUDA Available: {torch.cuda.is_available()}
CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
PyTorch Version: {torch.__version__}
GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"""
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            info += f"\nGPU {i}: {gpu_name} ({gpu_memory:.1f} GB)"
    
    logger._write_to_log(info)
    print(info)
