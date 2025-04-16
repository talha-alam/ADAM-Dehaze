import os
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from collections import defaultdict
import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def calculate_image_metrics(pred, target):
    """
    Calculate image quality metrics between predicted and target images
    
    Args:
        pred (numpy.ndarray): Predicted image (H, W, 3) in range [0, 1]
        target (numpy.ndarray): Target image (H, W, 3) in range [0, 1]
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # PSNR
    metrics['psnr'] = peak_signal_noise_ratio(target, pred, data_range=1.0)
    
    # SSIM (on grayscale)
    gray_pred = np.mean(pred, axis=2)
    gray_target = np.mean(target, axis=2)
    metrics['ssim'] = structural_similarity(gray_target, gray_pred, data_range=1.0)
    
    # Additional metrics could be added here
    
    return metrics

class ImageQualityMetrics:
    """Class to compute and store image quality metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.results = defaultdict(list)
        
    def add_sample(self, pred, target, category=None):
        """
        Add a sample for evaluation
        
        Args:
            pred (torch.Tensor): Predicted image tensor
            target (torch.Tensor): Target image tensor
            category (str, optional): Category for grouping results
        """
        # Make sure inputs are on CPU and in numpy format
        if isinstance(pred, torch.Tensor):
            pred_np = pred.permute(1, 2, 0).cpu().numpy()
            target_np = target.permute(1, 2, 0).cpu().numpy()
        else:
            pred_np = pred
            target_np = target
        
        # Calculate standard metrics
        metrics = calculate_image_metrics(pred_np, target_np)
        
        # Calculate LPIPS
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            # LPIPS expects input in [-1, 1]
            pred_lpips = 2 * pred.unsqueeze(0).to(self.device) - 1
            target_lpips = 2 * target.unsqueeze(0).to(self.device) - 1
            
            with torch.no_grad():
                lpips_value = self.lpips_fn(pred_lpips, target_lpips).item()
            
            metrics['lpips'] = lpips_value
        
        # Store metrics
        if category:
            self.results[category].append(metrics)
        else:
            self.results['all'].append(metrics)
    
    def compute_averages(self):
        """Compute average metrics for each category"""
        avg_results = {}
        
        for category, metrics_list in self.results.items():
            if not metrics_list:
                continue
                
            avg_results[category] = {}
            
            # Compute averages for each metric
            for metric_name in metrics_list[0].keys():
                avg_results[category][metric_name] = np.mean([m[metric_name] for m in metrics_list])
                
            # Add sample count
            avg_results[category]['samples'] = len(metrics_list)
        
        return avg_results
    
    def print_results(self):
        """Print the evaluation results"""
        avg_results = self.compute_averages()
        
        print("Image Quality Evaluation Results:")
        
        for category, metrics in sorted(avg_results.items()):
            print(f"\n{category.upper()} ({metrics['samples']} samples):")
            for metric_name, value in metrics.items():
                if metric_name != 'samples':  # Skip sample count
                    print(f"  {metric_name.upper()}: {value:.4f}")
        
        return avg_results
    
    def save_results(self, output_path):
        """Save the results to a JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.compute_averages(), f, indent=2)
        
        print(f"Results saved to {output_path}")

class DetectionMetrics:
    """Class to evaluate object detection metrics"""
    
    def __init__(self, annotation_file):
        """
        Initialize with COCO-format annotation file
        
        Args:
            annotation_file (str): Path to COCO format annotation file
        """
        self.coco_gt = COCO(annotation_file)
        self.results = []
        self.category_results = defaultdict(list)
        
    def add_detection_result(self, image_id, category_id, bbox, score, category=None):
        """
        Add a detection result
        
        Args:
            image_id (int): Image ID
            category_id (int): Category ID
            bbox (list): Bounding box in [x, y, width, height] format
            score (float): Confidence score
            category (str, optional): Category for grouping results
        """
        result = {
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'score': score
        }
        
        self.results.append(result)
        
        if category:
            self.category_results[category].append(result)
    
    def evaluate(self, iou_thresholds=None):
        """
        Evaluate detection results
        
        Args:
            iou_thresholds (list, optional): List of IoU thresholds
            
        Returns:
            dict: Dictionary of evaluation results
        """
        if not self.results:
            print("No detection results to evaluate")
            return {}
        
        # Create COCO result format
        coco_dt = self.coco_gt.loadRes(self.results)
        
        # Create COCO evaluator
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        
        # Set IoU thresholds if provided
        if iou_thresholds:
            coco_eval.params.iouThrs = np.array(iou_thresholds)
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract results
        results = {
            'mAP': coco_eval.stats[0],  # AP @ IoU=0.5:0.95
            'mAP_50': coco_eval.stats[1],  # AP @ IoU=0.5
            'mAP_75': coco_eval.stats[2],  # AP @ IoU=0.75
            'mAP_small': coco_eval.stats[3],  # AP for small objects
            'mAP_medium': coco_eval.stats[4],  # AP for medium objects
            'mAP_large': coco_eval.stats[5],  # AP for large objects
            'AR_1': coco_eval.stats[6],  # AR with max 1 det per image
            'AR_10': coco_eval.stats[7],  # AR with max 10 det per image
            'AR_100': coco_eval.stats[8],  # AR with max 100 det per image
            'AR_small': coco_eval.stats[9],  # AR for small objects
            'AR_medium': coco_eval.stats[10],  # AR for medium objects
            'AR_large': coco_eval.stats[11]  # AR for large objects
        }
        
        return results
    
    def evaluate_by_category(self, iou_thresholds=None):
        """
        Evaluate detection results grouped by category
        
        Args:
            iou_thresholds (list, optional): List of IoU thresholds
            
        Returns:
            dict: Dictionary of evaluation results by category
        """
        results_by_category = {}
        
        # Evaluate all results
        overall_results = self.evaluate(iou_thresholds)
        results_by_category['overall'] = overall_results
        
        # Evaluate each category
        for category, category_results in self.category_results.items():
            # Backup current results
            backup_results = self.results.copy()
            
            # Replace with category results
            self.results = category_results
            
            # Evaluate
            cat_results = self.evaluate(iou_thresholds)
            results_by_category[category] = cat_results
            
            # Restore results
            self.results = backup_results
        
        return results_by_category
    
    def print_results(self, results=None):
        """Print evaluation results"""
        if results is None or not results:
            print("No detection results to evaluate")
            # Return empty dictionary to avoid KeyError
            return {
                'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0,
                'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.0
            }
        
        print("Object Detection Evaluation Results:")
        print(f"  mAP (IoU=0.5:0.95): {results['mAP']:.4f}")
        print(f"  mAP (IoU=0.5): {results['mAP_50']:.4f}")
        print(f"  mAP (IoU=0.75): {results['mAP_75']:.4f}")
        print(f"  mAP (small objects): {results['mAP_small']:.4f}")
        print(f"  mAP (medium objects): {results['mAP_medium']:.4f}")
        print(f"  mAP (large objects): {results['mAP_large']:.4f}")
        
        return results
    
    def save_results(self, results, output_path):
        """Save results to a JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")

def calculate_perceptual_scores(model, dataset, config):
    """
    Calculate perceptual quality scores using a trained model
    
    Args:
        model (nn.Module): Trained dehazing model
        dataset (Dataset): Dataset to evaluate on
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary of perceptual scores
    """
    import torch.nn.functional as F
    from torchvision import models
    
    device = torch.device(config['device'])
    model = model.to(device)
    model.eval()
    
    # Use a pretrained VGG16 for feature extraction
    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    
    # Layer indices for feature extraction
    layers = {
        'relu1_2': 4,
        'relu2_2': 9,
        'relu3_3': 16,
        'relu4_3': 23
    }
    
    # Metrics
    metrics = {
        'naturalness': 0.0,
        'structure_similarity': 0.0,
        'samples': 0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Calculating perceptual scores"):
            hazy_imgs = batch['hazy'].to(device)
            clear_imgs = batch['clear'].to(device)
            
            # Forward pass
            dehazed_imgs = model(hazy_imgs)
            
            # Calculate metrics for each image
            for i in range(hazy_imgs.size(0)):
                dehazed = dehazed_imgs[i].unsqueeze(0)
                clear = clear_imgs[i].unsqueeze(0)
                
                # Extract features from both images
                dehazed_features = {}
                clear_features = {}
                
                x_dehazed = dehazed
                x_clear = clear
                
                for name, idx in layers.items():
                    for j in range(idx+1):
                        x_dehazed = vgg[j](x_dehazed)
                        x_clear = vgg[j](x_clear)
                    
                    dehazed_features[name] = x_dehazed
                    clear_features[name] = x_clear
                
                # Calculate naturalness score (based on high-level features)
                naturalness = F.mse_loss(dehazed_features['relu4_3'], clear_features['relu4_3']).item()
                metrics['naturalness'] += naturalness
                
                # Calculate structure similarity (based on low-level features)
                structure_sim = F.mse_loss(dehazed_features['relu2_2'], clear_features['relu2_2']).item()
                metrics['structure_similarity'] += structure_sim
                
                metrics['samples'] += 1
    
    # Compute averages
    for key in metrics:
        if key != 'samples':
            metrics[key] /= metrics['samples']
            
            # Convert to a score (lower is better for MSE, so invert)
            metrics[key] = 1.0 / (1.0 + metrics[key])
    
    return metrics