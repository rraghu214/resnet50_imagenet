import torch
import torch.nn as nn
import os
import argparse
from datetime import datetime

# Import local modules
from config import Config, DATASET_INFO
from augmentations import ImageNetAugmentations, MixupCutmix
from models import create_model
from datasets import create_dataloaders, setup_colab_dataset
from optimizers import OptimizerFactory, SchedulerFactory, ModelEMA, LRFinder
from trainer import Trainer
from logger import Logger, setup_logging_directory, log_system_info

class ResNetTrainingOrchestrator:
    """Main orchestration class for ResNet training"""
    
    def __init__(self, config):
        self.config = config
        
        # Setup logging
        self.config = setup_logging_directory(config)
        self.logger = Logger(self.config)
        
        # Log system info
        log_system_info(self.logger)
        
        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.trainer = None
        
        print(f"Initialized ResNet Training Orchestrator")
        print(f"Dataset: {config.dataset}")
        print(f"Model: {config.model_name}")
        print(f"Platform: {config.platform}")
        print(f"Log directory: {config.log_dir}")
    
    def setup_dataset(self):
        """Setup dataset and data loaders"""
        print("Setting up dataset...")
        
        # Platform specific setup
        if self.config.platform == "colab":
            setup_colab_dataset(self.config)
        elif self.config.platform == "ec2":
            # Optimize EC2 data loading with EBS volumes
            from datasets import optimize_ec2_data_loading
            optimize_ec2_data_loading(self.config)
        
        # Update config based on dataset
        dataset_info = DATASET_INFO[self.config.dataset]
        self.config.num_classes = dataset_info["num_classes"]
        
        # Create augmentations
        augmentations = ImageNetAugmentations(self.config)
        
        # Create data loaders
        self.train_loader, self.val_loader = create_dataloaders(self.config, augmentations)
        
        print(f"Dataset setup complete:")
        print(f"  Training samples: {len(self.train_loader.dataset)}")
        print(f"  Validation samples: {len(self.val_loader.dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Number of classes: {self.config.num_classes}")
    
    def setup_model(self):
        """Setup model"""
        print("Setting up model...")
        
        # Create model
        self.model = create_model(self.config)
        
        # Compile model for PyTorch 2.0+ (if available)
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Could not compile model: {e}")
        
        # Move to device
        device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model setup complete:")
        print(f"  Architecture: {self.config.model_name}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {device}")
    
    def setup_optimizer_scheduler(self):
        """Setup optimizer and scheduler"""
        print("Setting up optimizer and scheduler...")
        
        # Create optimizer
        optimizer_factory = OptimizerFactory(self.config)
        self.optimizer = optimizer_factory.create_optimizer(self.model)
        
        # Create scheduler
        scheduler_factory = SchedulerFactory(self.config)
        steps_per_epoch = len(self.train_loader)
        self.scheduler = scheduler_factory.create_scheduler(self.optimizer, steps_per_epoch)
        
        print(f"Optimizer and scheduler setup complete:")
        print(f"  Optimizer: {self.config.optimizer}")
        print(f"  Scheduler: {self.config.scheduler}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Steps per epoch: {steps_per_epoch}")
    
    def setup_training_components(self):
        """Setup training-specific components"""
        print("Setting up training components...")
        
        # Mixup/CutMix
        mixup_cutmix = None
        if self.config.mixup_alpha > 0 or self.config.cutmix_alpha > 0:
            mixup_cutmix = MixupCutmix(
                mixup_alpha=self.config.mixup_alpha,
                cutmix_alpha=self.config.cutmix_alpha,
                prob=0.5
            )
            print(f"  Mixup/CutMix enabled (mixup_alpha={self.config.mixup_alpha}, cutmix_alpha={self.config.cutmix_alpha})")
        
        # Model EMA
        model_ema = None
        if self.config.model_ema:
            model_ema = ModelEMA(self.model, decay=self.config.ema_decay)
            print(f"  Model EMA enabled (decay={self.config.ema_decay})")
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            logger=self.logger,
            mixup_cutmix=mixup_cutmix,
            model_ema=model_ema
        )
        
        print("Training components setup complete")
    
    def run_lr_finder(self):
        """Run learning rate range test"""
        print("Running learning rate finder...")
        
        # Create LR finder
        criterion = nn.CrossEntropyLoss()
        lr_finder = LRFinder(self.model, self.optimizer, criterion, 
                           torch.device(self.config.device))
        
        # Run range test
        lrs, losses = lr_finder.range_test(self.train_loader, num_iter=100)
        
        # Plot results
        optimal_lr = self.logger.plot_lr_finder_results(lrs, losses)
        
        return optimal_lr
    
    def train(self, resume_from_checkpoint=None):
        """Main training function"""
        print("Starting training...")
        
        # Load checkpoint if specified
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self.trainer.load_checkpoint(resume_from_checkpoint)
        
        # Start training
        start_time = datetime.now()
        best_acc = self.trainer.train()
        end_time = datetime.now()
        
        # Training summary
        training_time = end_time - start_time
        print(f"\nTraining Summary:")
        print(f"  Duration: {training_time}")
        print(f"  Best validation accuracy: {best_acc:.2f}%")
        print(f"  Target accuracy (78%): {'✓ ACHIEVED' if best_acc >= 78.0 else '✗ NOT ACHIEVED'}")
        
        # Save final summary
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': str(training_time),
            'best_accuracy': best_acc,
            'target_achieved': best_acc >= 78.0,
            'config': vars(self.config)
        }
        
        import json
        summary_path = os.path.join(self.config.log_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return best_acc
    
    def evaluate(self, checkpoint_path=None):
        """Evaluate model"""
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
        
        print("Evaluating model...")
        val_loss, val_acc, val_top5 = self.trainer.validate_epoch()
        
        print(f"Evaluation Results:")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.2f}%")
        print(f"  Top-5 Accuracy: {val_top5:.2f}%")
        
        return val_acc, val_top5
    
    def run_full_pipeline(self, run_lr_finder=False, resume_checkpoint=None):
        """Run the complete training pipeline"""
        try:
            # Setup all components
            self.setup_dataset()
            self.setup_model()
            self.setup_optimizer_scheduler()
            self.setup_training_components()
            
            # Optional LR finder
            if run_lr_finder:
                optimal_lr = self.run_lr_finder()
                print(f"Suggested learning rate from LR finder: {optimal_lr:.2e}")
                
                # Ask user if they want to use the suggested LR
                response = input(f"Use suggested LR {optimal_lr:.2e}? (y/n): ")
                if response.lower() == 'y':
                    self.config.learning_rate = optimal_lr
                    self.config.max_lr = optimal_lr
                    # Recreate optimizer and scheduler with new LR
                    self.setup_optimizer_scheduler()
                    self.setup_training_components()
            
            # Train
            best_acc = self.train(resume_checkpoint)
            
            # Final evaluation
            print("\nFinal evaluation with best model...")
            best_checkpoint = os.path.join(self.config.save_dir, 'best_checkpoint.pth')
            if os.path.exists(best_checkpoint):
                final_acc, final_top5 = self.evaluate(best_checkpoint)
            
            return best_acc
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            return None
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            return None

def create_config_from_args(args):
    """Create config from command line arguments"""
    config = Config()
    
    # Update config with command line arguments
    if args.dataset:
        config.dataset = args.dataset
    if args.model:
        config.model_name = args.model
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
        config.max_lr = args.lr
    if args.platform:
        config.platform = args.platform
    if args.data_dir:
        config.data_dir = args.data_dir
    
    # Platform-specific adjustments
    if config.platform == "colab":
        config.data_dir = "/content/drive/MyDrive/datasets"
        config.epochs = min(config.epochs, 50)  # Limit for Colab
        config.batch_size = min(config.batch_size, 128)  # Memory constraints
    
    return config

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ResNet50 ImageNet Training')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='imagenet100',
                       choices=['imagenet1k', 'imagenet100k', 'tiny_imagenet'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet50d'],
                       help='Model architecture')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # Platform arguments
    parser.add_argument('--platform', type=str, default='local',
                       choices=['local', 'colab', 'ec2'],
                       help='Platform to run on')
    
    # Utility arguments
    parser.add_argument('--lr-finder', action='store_true',
                       help='Run learning rate finder')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--eval-only', type=str, default=None,
                       help='Only evaluate with given checkpoint')
    
    args = parser.parse_args()
    
    # Create config
    config = create_config_from_args(args)
    
    # Create orchestrator
    orchestrator = ResNetTrainingOrchestrator(config)
    
    # Run evaluation only
    if args.eval_only:
        orchestrator.setup_dataset()
        orchestrator.setup_model()
        orchestrator.setup_optimizer_scheduler()
        orchestrator.setup_training_components()
        orchestrator.evaluate(args.eval_only)
        return
    
    # Run full pipeline
    best_acc = orchestrator.run_full_pipeline(
        run_lr_finder=args.lr_finder,
        resume_checkpoint=args.resume
    )
    
    if best_acc is not None:
        print(f"\nTraining completed successfully!")
        print(f"Best accuracy: {best_acc:.2f}%")
        if best_acc >= 78.0:
            print("🎉 TARGET ACHIEVED! Accuracy >= 78%")
        else:
            print(f"❌ Target not achieved. Need {78.0 - best_acc:.2f}% more.")

if __name__ == "__main__":
    main()
