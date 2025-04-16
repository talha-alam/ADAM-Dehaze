import torch
import torch.nn as nn
import torchvision.models as models
import timm

class FogIntensityClassifier(nn.Module):
    """Classifier for fog intensity (low, medium, high)"""
    
    def __init__(self, model_name='resnet18', num_classes=3, pretrained=True):
        """
        Args:
            model_name (str): Name of the backbone model
            num_classes (int): Number of fog intensity classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(FogIntensityClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Initialize backbone based on model name
        if model_name.startswith('resnet'):
            if model_name == 'resnet18':
                self.backbone = models.resnet18(pretrained=pretrained)
                self.feature_dim = 512
            elif model_name == 'resnet34':
                self.backbone = models.resnet34(pretrained=pretrained)
                self.feature_dim = 512
            elif model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                self.feature_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet variant: {model_name}")
                
            # Replace the final fully connected layer
            self.backbone.fc = nn.Identity()  # Remove original FC layer
        
        elif model_name.startswith('efficientnet'):
            # Use timm for EfficientNet
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            
            # Get the feature dimension
            if hasattr(self.backbone, 'classifier'):
                self.feature_dim = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            elif hasattr(self.backbone, 'fc'):
                self.feature_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            else:
                raise ValueError(f"Cannot determine feature dimension for {model_name}")
        
        elif model_name.startswith('mobilenet'):
            if model_name == 'mobilenet_v2':
                self.backbone = models.mobilenet_v2(pretrained=pretrained)
                self.feature_dim = 1280
            elif model_name == 'mobilenet_v3_small':
                self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
                self.feature_dim = 576
            elif model_name == 'mobilenet_v3_large':
                self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
                self.feature_dim = 960
            else:
                raise ValueError(f"Unsupported MobileNet variant: {model_name}")
            
            # Replace classifier
            self.backbone.classifier = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, H, W]
            
        Returns:
            logits (torch.Tensor): Class logits of shape [batch_size, num_classes]
            features (torch.Tensor): Backbone features of shape [batch_size, feature_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get classification logits
        logits = self.classifier(features)
        
        return logits, features
    
    def extract_features(self, x):
        """Extract features without classification"""
        with torch.no_grad():
            features = self.backbone(x)
        return features

class DenseFeatureExtractor(nn.Module):
    """Extract dense feature maps from the backbone"""
    
    def __init__(self, model_name='resnet18', pretrained=True):
        super(DenseFeatureExtractor, self).__init__()
        
        # Initialize the backbone
        if model_name == 'resnet18':
            # Remove the average pooling and FC layers
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        elif model_name == 'mobilenet_v2':
            # For MobileNet, extract the feature extractor (features)
            self.backbone = models.mobilenet_v2(pretrained=pretrained).features
            
        elif model_name.startswith('efficientnet'):
            # EfficientNet requires more careful handling
            model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            # Use only the convolutional parts (exclude head)
            self.backbone = model
        
        else:
            raise ValueError(f"Unsupported model for feature extraction: {model_name}")
    
    def forward(self, x):
        """Extract dense feature maps"""
        return self.backbone(x)


def create_classifier(config):
    """Create fog intensity classifier based on config"""
    return FogIntensityClassifier(
        model_name=config['classifier']['model'],
        num_classes=config['classifier']['num_classes'],
        pretrained=config['classifier']['pretrained']
    )