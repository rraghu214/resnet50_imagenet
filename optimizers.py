"""
Optimizer and Scheduler classes
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
import math

class OptimizerFactory:
    """Factory class for creating optimizers"""
    
    def __init__(self, config):
        self.config = config
        
    def create_optimizer(self, model):
        """Create optimizer based on config"""
        if self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        elif self.config.optimizer.lower() == "adam":
            return optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

class SchedulerFactory:
    """Factory class for creating learning rate schedulers"""
    
    def __init__(self, config):
        self.config = config
        
    def create_scheduler(self, optimizer, steps_per_epoch):
        """Create scheduler based on config"""
        total_steps = self.config.epochs * steps_per_epoch
        
        if self.config.scheduler.lower() == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                total_steps=total_steps,
                pct_start=self.config.pct_start,
                div_factor=self.config.div_factor,
                final_div_factor=self.config.final_div_factor,
                anneal_strategy='cos'
            )
        elif self.config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate / 100
            )
        elif self.config.scheduler.lower() == "step":
            return StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "warmup_cosine":
            return WarmupCosineScheduler(
                optimizer,
                warmup_epochs=5,
                total_epochs=self.config.epochs,
                base_lr=self.config.learning_rate,
                max_lr=self.config.max_lr
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

class WarmupCosineScheduler:
    """Custom warmup + cosine annealing scheduler"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_epoch = 0
        
    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr + (self.max_lr - self.base_lr) * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            cosine_epochs = self.total_epochs - self.warmup_epochs
            cosine_epoch = self.current_epoch - self.warmup_epochs
            lr = self.base_lr + (self.max_lr - self.base_lr) * 0.5 * (
                1 + math.cos(math.pi * cosine_epoch / cosine_epochs)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1
        
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

class ModelEMA:
    """Exponential Moving Average of model parameters"""
    
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self, model):
        """Update EMA parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
                
    def apply_shadow(self, model):
        """Apply EMA parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self, model):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class LRFinder:
    """Learning Rate Range Test"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def range_test(self, dataloader, start_lr=1e-7, end_lr=10, num_iter=100):
        """Perform learning rate range test"""
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        lr = start_lr
        losses = []
        lrs = []
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.model.train()
        
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_iter:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            lrs.append(lr)
            
            # Update learning rate
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
        return lrs, losses
