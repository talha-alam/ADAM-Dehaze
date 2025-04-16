import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lpips

class ContentLoss(nn.Module):
    """Content loss based on pretrained VGG features"""
    
    def __init__(self, pretrained_model='vgg16', content_layers=None):
        super(ContentLoss, self).__init__()
        
        if content_layers is None:
            content_layers = ['relu2_2', 'relu3_3', 'relu4_3']
        
        self.content_layers = content_layers
        
        # Initialize the VGG16 model
        if pretrained_model == 'vgg16':
            self.model = models.vgg16(pretrained=True).features.eval()
        elif pretrained_model == 'vgg19':
            self.model = models.vgg19(pretrained=True).features.eval()
        else:
            raise ValueError(f"Unsupported model: {pretrained_model}")
        
        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Layer mapping for VGG16
        self.layer_mapping = {
            'relu1_1': '2',
            'relu1_2': '4',
            'relu2_1': '7',
            'relu2_2': '9',
            'relu3_1': '12',
            'relu3_2': '14',
            'relu3_3': '16',
            'relu4_1': '19',
            'relu4_2': '21',
            'relu4_3': '23',
            'relu5_1': '26',
            'relu5_2': '28',
            'relu5_3': '30'
        }
    
    def forward(self, x, target):
        """
        Compute the content loss
        
        Args:
            x (torch.Tensor): Predicted image batch
            target (torch.Tensor): Target image batch
            
        Returns:
            torch.Tensor: Content loss value
        """
        # Ensure inputs are in the right range for VGG
        x = x.clone()
        target = target.clone()
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x = (x - mean) / std
        target = (target - mean) / std
        
        # Compute loss at specified layers
        loss = 0.0
        
        for name in self.content_layers:
            idx = int(self.layer_mapping[name])
            model = nn.Sequential(*list(self.model.children())[:idx+1])
            
            # Get features
            x_features = model(x)
            target_features = model(target)
            
            # Compute MSE loss between features
            layer_loss = F.mse_loss(x_features, target_features)
            loss += layer_loss
        
        return loss / len(self.content_layers)

class PerceptualLoss(nn.Module):
    """Perceptual loss based on LPIPS"""
    
    def __init__(self, net='alex'):
        super(PerceptualLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net=net)
        
    def forward(self, x, target):
        """
        Compute the perceptual loss
        
        Args:
            x (torch.Tensor): Predicted image batch [0, 1]
            target (torch.Tensor): Target image batch [0, 1]
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        # LPIPS expects input in [-1, 1] range
        x = 2 * x - 1
        target = 2 * target - 1
        
        return self.loss_fn(x, target)

class DehazingLoss(nn.Module):
    """Combined loss for image dehazing"""
    
    def __init__(self, lambda_l1=1.0, lambda_content=0.1, lambda_perceptual=0.1):
        super(DehazingLoss, self).__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_content = lambda_content
        self.lambda_perceptual = lambda_perceptual
        
        # Initialize individual losses
        self.l1_loss = nn.L1Loss()
        self.content_loss = ContentLoss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target):
        """
        Compute the combined dehazing loss
        
        Args:
            pred (torch.Tensor): Predicted dehazed image batch
            target (torch.Tensor): Target clean image batch
            
        Returns:
            torch.Tensor: Total loss value
            dict: Dictionary of individual loss components
        """
        # L1 reconstruction loss
        l1 = self.l1_loss(pred, target)
        
        # Content loss
        content = self.content_loss(pred, target)
        
        # Perceptual loss
        perceptual = self.perceptual_loss(pred, target)
        # Reduce perceptual loss to a scalar if needed
        if perceptual.dim() > 0:
            perceptual = perceptual.mean()
        
        # Combine losses
        total_loss = (
            self.lambda_l1 * l1 +
            self.lambda_content * content +
            self.lambda_perceptual * perceptual
        )
        
        # Return both total loss and individual components
        return total_loss, {
            'l1': l1,
            'content': content,
            'perceptual': perceptual,
            'total': total_loss
        }

class JointLoss(nn.Module):
    """Combined loss for joint training of classification and dehazing"""
    
    def __init__(self, lambda_dehazing=1.0, lambda_classification=0.2, 
                 lambda_detection=0.5, config=None):
        super(JointLoss, self).__init__()
        
        self.lambda_dehazing = lambda_dehazing
        self.lambda_classification = lambda_classification
        self.lambda_detection = lambda_detection
        
        # Initialize individual losses
        self.dehazing_loss = DehazingLoss()
        self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, target_clear, pred_intensity=None, target_intensity=None, 
                detection_loss=None):
        """
        Compute the joint loss
        
        Args:
            pred (torch.Tensor): Predicted dehazed image batch
            target_clear (torch.Tensor): Target clean image batch
            pred_intensity (torch.Tensor, optional): Predicted fog intensity logits
            target_intensity (torch.Tensor, optional): Target fog intensity labels
            detection_loss (torch.Tensor, optional): Loss from detection model
            
        Returns:
            torch.Tensor: Total loss value
            dict: Dictionary of individual loss components
        """
        # Dehazing loss
        dehazing_loss, dehazing_components = self.dehazing_loss(pred, target_clear)
        
        # Classification loss (if provided)
        if pred_intensity is not None and target_intensity is not None:
            classification_loss = self.classification_loss(pred_intensity, target_intensity)
        else:
            classification_loss = torch.tensor(0.0, device=pred.device)
        
        # Detection loss (if provided)
        if detection_loss is not None:
            detection_component = detection_loss
        else:
            detection_component = torch.tensor(0.0, device=pred.device)
        
        # Combine losses
        total_loss = (
            self.lambda_dehazing * dehazing_loss +
            self.lambda_classification * classification_loss +
            self.lambda_detection * detection_component
        )
        
        # Return both total loss and individual components
        return total_loss, {
            'dehazing': dehazing_loss,
            'classification': classification_loss,
            'detection': detection_component,
            'total': total_loss,
            'dehazing_components': dehazing_components
        }

def get_dehazing_loss(config):
    """Create dehazing loss based on config"""
    return DehazingLoss(
        lambda_l1=1.0,
        lambda_content=0.1,
        lambda_perceptual=0.1
    )

def get_joint_loss(config):
    """Create joint training loss based on config"""
    return JointLoss(
        lambda_dehazing=config['joint_training']['lambda_dehazing'],
        lambda_classification=config['joint_training']['lambda_classification'],
        lambda_detection=config['joint_training']['lambda_detection'],
        config=config
    )