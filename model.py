"""
Model architectures for Offroad Semantic Segmentation
Implements SegFormer and DeepLabV3+ with pretrained backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from config import Config

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not available. Install with: pip install segmentation-models-pytorch")


def get_model(num_classes):
    """
    Factory function to create segmentation model
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        Model instance
    """
    if Config.MODEL_NAME == 'segformer':
        if not SMP_AVAILABLE:
            print("SegFormer requires segmentation_models_pytorch. Falling back to DeepLabV3+")
            return get_deeplabv3plus(num_classes)
        return get_segformer(num_classes)
    elif Config.MODEL_NAME == 'deeplabv3plus':
        return get_deeplabv3plus(num_classes)
    else:
        raise ValueError(f"Unknown model name: {Config.MODEL_NAME}")


def get_segformer(num_classes):
    """
    Create SegFormer model with pretrained encoder
    
    SegFormer is chosen for:
    - Superior global context modeling via efficient self-attention
    - Better generalization to domain shifts
    - Hierarchical multi-scale feature extraction
    - State-of-the-art performance on segmentation benchmarks
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        SegFormer model
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is required for SegFormer")
    
    print(f"Creating SegFormer model with {Config.SEGFORMER_ENCODER} encoder")
    
    model = smp.SegFormer(
        encoder_name=Config.SEGFORMER_ENCODER,
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes
    )
    
    return model


def get_deeplabv3plus(num_classes):
    """
    Create DeepLabV3+ model with pretrained ResNet backbone
    
    DeepLabV3+ advantages:
    - Atrous spatial pyramid pooling for multi-scale context
    - Strong pretrained ResNet backbones
    - Proven performance on segmentation tasks
    - Encoder-decoder structure with skip connections
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        DeepLabV3+ model
    """
    if SMP_AVAILABLE:
        print(f"Creating DeepLabV3+ model with {Config.DEEPLABV3_BACKBONE} backbone")
        
        model = smp.DeepLabV3Plus(
            encoder_name=Config.DEEPLABV3_BACKBONE,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    else:
        # Fallback to torchvision implementation
        print(f"Using torchvision DeepLabV3 with {Config.DEEPLABV3_BACKBONE} backbone")
        
        if Config.DEEPLABV3_BACKBONE == 'resnet50':
            model = deeplabv3_resnet50(pretrained=True)
        elif Config.DEEPLABV3_BACKBONE == 'resnet101':
            model = deeplabv3_resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {Config.DEEPLABV3_BACKBONE}")
        
        # Modify classifier for correct number of classes
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    return model


class DeepLabV3PlusWrapper(nn.Module):
    """
    Wrapper for torchvision DeepLabV3 to match expected interface
    """
    
    def __init__(self, base_model):
        super(DeepLabV3PlusWrapper, self).__init__()
        self.base_model = base_model
    
    def forward(self, x):
        output = self.base_model(x)
        return output['out']


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """Print model information"""
    total_params = count_parameters(model)
    print(f"\nModel Information:")
    print(f"  Total trainable parameters: {total_params:,}")
    print(f"  Model size (approx): {total_params * 4 / 1024 / 1024:.2f} MB")


# Custom model implementations (if needed)

class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module
    Multi-scale feature extraction with dilated convolutions
    """
    
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.atrous_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        total_out = out_channels * (len(dilation_rates) + 2)
        self.project = nn.Sequential(
            nn.Conv2d(total_out, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        
        # 1x1 conv
        feat1 = self.conv1x1(x)
        
        # Atrous convolutions
        atrous_feats = [conv(x) for conv in self.atrous_convs]
        
        # Global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        features = torch.cat([feat1] + atrous_feats + [global_feat], dim=1)
        
        # Project to output channels
        out = self.project(features)
        
        return out


class SegmentationHead(nn.Module):
    """
    Simple segmentation head for output
    """
    
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super(SegmentationHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, num_classes, 1)
        )
    
    def forward(self, x):
        return self.head(x)
