"""
Model definitions for ResNet50 and ResNet50-D
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import math

class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_path=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = self.drop_path(out)
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    """Bottleneck ResNet block"""
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_path=0.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = self.drop_path(out)
        out += identity
        out = self.relu(out)
        
        return out

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) implementation"""
    
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ResNet(nn.Module):
    """ResNet implementation with optional ResNet-D modifications"""
    
    def __init__(self, block, layers, num_classes=1000, use_resnet_d=False, 
                 drop_path_rate=0.0, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.use_resnet_d = use_resnet_d
        
        # ResNet-D stem
        if use_resnet_d:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Calculate drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        
        self.layer1 = self._make_layer(block, 64, layers[0], drop_path_rates=dpr[0:layers[0]])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       drop_path_rates=dpr[layers[0]:layers[0]+layers[1]])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       drop_path_rates=dpr[layers[0]+layers[1]:layers[0]+layers[1]+layers[2]])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       drop_path_rates=dpr[layers[0]+layers[1]+layers[2]:])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights(zero_init_residual)
        
    def _make_layer(self, block, planes, blocks, stride=1, drop_path_rates=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        drop_path_rate = drop_path_rates[0] if drop_path_rates else 0.0
        layers.append(block(self.inplanes, planes, stride, downsample, drop_path_rate))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            drop_path_rate = drop_path_rates[i] if drop_path_rates else 0.0
            layers.append(block(self.inplanes, planes, drop_path=drop_path_rate))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_model(config):
    """Create model based on configuration"""
    if config.model_name == "resnet50":
        model = ResNet(
            Bottleneck, 
            [3, 4, 6, 3], 
            num_classes=config.num_classes,
            use_resnet_d=False,
            drop_path_rate=0.0
        )
    elif config.model_name == "resnet50d":
        model = ResNet(
            Bottleneck, 
            [3, 4, 6, 3], 
            num_classes=config.num_classes,
            use_resnet_d=True,
            drop_path_rate=0.1,
            zero_init_residual=True
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model

def load_pretrained_backbone(model, config):
    """Load pretrained backbone (if needed for transfer learning)"""
    if config.model_name == "resnet50":
        pretrained = resnet50(pretrained=True)
        model_dict = model.state_dict()
        pretrained_dict = pretrained.state_dict()
        
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    return model
