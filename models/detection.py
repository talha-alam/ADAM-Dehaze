import torch
import torch.nn as nn
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class DetectionModel(nn.Module):
    """Object detection model with support for multiple architectures"""
    
    def __init__(self, num_classes=91, model_name='faster_rcnn_resnet50_fpn', pretrained=True):
        """
        Args:
            num_classes (int): Number of classes for detection
            model_name (str): Name of the detection model to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(DetectionModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Initialize the model based on the architecture
        if model_name == 'faster_rcnn_resnet50_fpn':
            # Load a pre-trained Faster R-CNN model
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
            
            # Replace the classifier with a new one for our number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        elif model_name == 'faster_rcnn_mobilenet_v3_large_fpn':
            # Load a pre-trained Faster R-CNN model with MobileNet backbone
            self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
            
            # Replace the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        elif model_name == 'mask_rcnn_resnet50_fpn':
            # Load a pre-trained Mask R-CNN model
            self.model = detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
            
            # Replace the classifiers
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # And the mask predictor if needed
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes)
            
        else:
            raise ValueError(f"Unsupported detection model: {model_name}")
        
    def forward(self, images, targets=None):
        """
        Forward pass of the detection model
        
        Args:
            images (List[Tensor]): Images to be processed
            targets (List[Dict[str, Tensor]], optional): List of target dicts for training
                
        Returns:
            List[Dict[str, Tensor]]: During inference, returns pred boxes, scores, and labels
                                    During training, returns the loss dict
        """
        # If targets is provided, the model is in training mode
        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)

class IntegratedDetectionSystem(nn.Module):
    """
    Integrated system that combines dehazing with object detection
    """
    
    def __init__(self, dehazing_model, detection_model):
        """
        Args:
            dehazing_model (nn.Module): Dehazing model to preprocess images
            detection_model (nn.Module): Object detection model
        """
        super(IntegratedDetectionSystem, self).__init__()
        
        self.dehazing_model = dehazing_model
        self.detection_model = detection_model
        
        # Fix detection model parameters if needed
        for param in self.detection_model.parameters():
            param.requires_grad = False
        
    def forward(self, images, targets=None):
        """
        Forward pass through the integrated system
        
        Args:
            images (List[Tensor]): Input hazy images
            targets (List[Dict[str, Tensor]], optional): List of target dicts for training
                
        Returns:
            List[Dict[str, Tensor]]: Detection results
            torch.Tensor: Dehazed images
        """
        # Apply dehazing
        dehazed_images, dehazing_info = self.dehazing_model(images)
        
        # Normalize for detection
        normalized_images = []
        for img in dehazed_images:
            # Convert to expected format for detection
            img_normalized = img.clone()
            # Most detection models expect normalization with ImageNet statistics
            img_normalized = img_normalized.sub(
                torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            ).div(
                torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            )
            normalized_images.append(img_normalized)
        
        # Run detection
        detection_results = self.detection_model(normalized_images, targets)
        
        return detection_results, dehazed_images

def create_detection_model(config):
    """Create object detection model based on config"""
    return DetectionModel(
        num_classes=91,  # COCO dataset has 91 classes by default
        model_name=config['detection']['model'],
        pretrained=config['detection']['pretrained']
    )

def create_integrated_system(dehazing_router, detection_model):
    """Create the integrated dehazing and detection system"""
    return IntegratedDetectionSystem(
        dehazing_model=dehazing_router,
        detection_model=detection_model
    )