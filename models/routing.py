import torch
import torch.nn as nn
import torch.nn.functional as F

class HardRouter(nn.Module):
    """
    Hard routing mechanism that directs input to a single dehazing model
    based on classification results
    """
    
    def __init__(self, models, classifier=None, device='cuda'):
        """
        Args:
            models (dict): Dictionary of dehazing models with keys 'low', 'medium', 'high'
            classifier (nn.Module): Fog intensity classifier model
            device (str): Device to use ('cuda' or 'cpu')
        """
        super(HardRouter, self).__init__()
        self.models = nn.ModuleDict(models)
        self.classifier = classifier
        self.device = device
        
    def forward(self, x, intensity=None):
        """
        Forward pass through the router
        
        Args:
            x (torch.Tensor): Input hazy image batch
            intensity (torch.Tensor, optional): Precalculated intensity labels
                                               (None to use classifier)
        
        Returns:
            torch.Tensor: Dehazed images
            dict: Additional outputs including classification results
        """
        batch_size = x.shape[0]
        outputs = torch.zeros_like(x)
        
        # Classify fog intensity if not provided
        if intensity is None and self.classifier is not None:
            with torch.no_grad():
                logits, _ = self.classifier(x)
                intensity = torch.argmax(logits, dim=1)
        
        # Process each sample based on intensity
        intensity_masks = {
            'low': intensity == 0,
            'medium': intensity == 1,
            'high': intensity == 2
        }
        
        # Forward pass through each model based on intensity
        for intensity_name, model in self.models.items():
            mask = intensity_masks[intensity_name]
            if torch.any(mask):
                # Process only the relevant subset
                subset_inputs = x[mask]
                subset_outputs = model(subset_inputs)
                
                # Place results back in the original batch order
                outputs[mask] = subset_outputs
        
        return outputs, {
            'intensity': intensity,
            'low_mask': intensity_masks['low'],
            'medium_mask': intensity_masks['medium'],
            'high_mask': intensity_masks['high']
        }

class SoftRouter(nn.Module):
    """
    Soft routing mechanism that blends outputs from all dehazing models
    based on classification confidence
    """
    
    def __init__(self, models, classifier=None, temperature=1.0, device='cuda'):
        """
        Args:
            models (dict): Dictionary of dehazing models with keys 'low', 'medium', 'high'
            classifier (nn.Module): Fog intensity classifier model
            temperature (float): Temperature parameter for softmax (lower = sharper)
            device (str): Device to use ('cuda' or 'cpu')
        """
        super(SoftRouter, self).__init__()
        self.models = nn.ModuleDict(models)
        self.classifier = classifier
        self.temperature = temperature
        self.device = device
        
    def forward(self, x, classifier_logits=None):
        """
        Forward pass through the router
        
        Args:
            x (torch.Tensor): Input hazy image batch
            classifier_logits (torch.Tensor, optional): Precalculated classifier logits
        
        Returns:
            torch.Tensor: Dehazed images
            dict: Additional outputs including classification results
        """
        batch_size = x.shape[0]
        
        # Get classification weights
        if classifier_logits is None and self.classifier is not None:
            logits, _ = self.classifier(x)
        else:
            logits = classifier_logits
            
        # Apply temperature scaling and softmax to get blending weights
        weights = F.softmax(logits / self.temperature, dim=1)
        
        # Process through each model
        outputs = {}
        for i, intensity_name in enumerate(['low', 'medium', 'high']):
            if intensity_name in self.models:
                # Forward pass through the model
                outputs[intensity_name] = self.models[intensity_name](x)
        
        # Blend outputs based on weights
        blended_output = torch.zeros_like(x)
        
        for i, intensity_name in enumerate(['low', 'medium', 'high']):
            if intensity_name in outputs:
                # Reshape weights to match output dimensions for broadcasting
                model_weights = weights[:, i].view(batch_size, 1, 1, 1)
                blended_output += model_weights * outputs[intensity_name]
        
        return blended_output, {
            'weights': weights,
            'individual_outputs': outputs
        }

class GatedRouter(nn.Module):
    """
    Gated routing mechanism that uses learned gates to blend model outputs
    and also incorporates feature-level fusion
    """
    
    def __init__(self, models, classifier=None, feature_dim=512, device='cuda'):
        """
        Args:
            models (dict): Dictionary of dehazing models with keys 'low', 'medium', 'high'
            classifier (nn.Module): Fog intensity classifier model
            feature_dim (int): Dimension of feature vectors from classifier
            device (str): Device to use ('cuda' or 'cpu')
        """
        super(GatedRouter, self).__init__()
        self.models = nn.ModuleDict(models)
        self.classifier = classifier
        self.device = device
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(models)),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion module (optional)
        self.use_feature_fusion = False
        if self.use_feature_fusion:
            self.fusion_module = nn.Sequential(
                nn.Conv2d(3 * 3, 32, kernel_size=3, padding=1),  # 3 models x 3 channels
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        """
        Forward pass through the router
        
        Args:
            x (torch.Tensor): Input hazy image batch
        
        Returns:
            torch.Tensor: Dehazed images
            dict: Additional outputs including gate values
        """
        batch_size = x.shape[0]
        
        # Get features and classification from classifier
        if self.classifier is not None:
            logits, features = self.classifier(x)
            # Generate gating weights from features
            gate_weights = self.gate_network(features)
        else:
            # Default to equal weighting
            gate_weights = torch.ones(batch_size, len(self.models)).to(self.device)
            gate_weights = gate_weights / len(self.models)
        
        # Process through each model
        outputs = {}
        for i, intensity_name in enumerate(['low', 'medium', 'high']):
            if intensity_name in self.models:
                # Forward pass through the model
                outputs[intensity_name] = self.models[intensity_name](x)
        
        # Combine outputs based on gating weights
        if self.use_feature_fusion:
            # Concatenate all outputs along channel dimension
            all_outputs = []
            for intensity_name in ['low', 'medium', 'high']:
                if intensity_name in outputs:
                    all_outputs.append(outputs[intensity_name])
            
            concat_outputs = torch.cat(all_outputs, dim=1)
            fused_output = self.fusion_module(concat_outputs)
            final_output = fused_output
        else:
            # Weighted sum of outputs
            final_output = torch.zeros_like(x)
            for i, intensity_name in enumerate(['low', 'medium', 'high']):
                if intensity_name in outputs:
                    # Reshape weights to match output dimensions for broadcasting
                    model_weights = gate_weights[:, i].view(batch_size, 1, 1, 1)
                    final_output += model_weights * outputs[intensity_name]
        
        return final_output, {
            'gate_weights': gate_weights,
            'individual_outputs': outputs
        }

def create_router(models, classifier, config):
    """Create a routing mechanism based on config"""
    routing_type = config['routing']['type']
    
    if routing_type == 'hard':
        return HardRouter(
            models=models,
            classifier=classifier,
            device=config['device']
        )
    elif routing_type == 'soft':
        return SoftRouter(
            models=models,
            classifier=classifier,
            temperature=config['routing']['temperature'],
            device=config['device']
        )
    elif routing_type == 'gated':
        return GatedRouter(
            models=models,
            classifier=classifier,
            device=config['device']
        )
    else:
        raise ValueError(f"Unsupported routing type: {routing_type}")