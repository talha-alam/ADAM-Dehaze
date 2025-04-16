import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
import json
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from data.dataset import get_dataloader, get_detection_dataloader
from models.classifier import create_classifier
from models.dehazing.low_intensity import create_low_intensity_model
from models.dehazing.medium_intensity import create_medium_intensity_model
from models.dehazing.high_intensity import create_high_intensity_model
from models.detection import create_detection_model, create_integrated_system
from models.routing import create_router
from evaluation.metrics import ImageQualityMetrics, DetectionMetrics
from utils.visualize import visualize_results, create_comparison_grid

def load_model_from_checkpoint(model, checkpoint_path, key='model_state_dict'):
    """Load model weights from checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint[key])
        print(f"Loaded model from {checkpoint_path}")
        return True
    else:
        print(f"Checkpoint {checkpoint_path} not found.")
        return False

def evaluate_baseline_models(config):
    """Evaluate individual dehazing models for each fog intensity"""
    # Set device
    device = torch.device(config['device'])
    
    # Create models
    low_model = create_low_intensity_model(config).to(device)
    medium_model = create_medium_intensity_model(config).to(device)
    high_model = create_high_intensity_model(config).to(device)
    
    # Load weights
    low_checkpoint = os.path.join(config['dehazing']['checkpoint_dir'], 'low', 'best_model.pth')
    med_checkpoint = os.path.join(config['dehazing']['checkpoint_dir'], 'medium', 'best_model.pth')
    high_checkpoint = os.path.join(config['dehazing']['checkpoint_dir'], 'high', 'best_model.pth')
    
    load_model_from_checkpoint(low_model, low_checkpoint)
    load_model_from_checkpoint(medium_model, med_checkpoint)
    load_model_from_checkpoint(high_model, high_checkpoint)
    
    # Set models to evaluation mode
    low_model.eval()
    medium_model.eval()
    high_model.eval()
    
    # Initialize metrics
    metrics = ImageQualityMetrics(device=device)
    
    # Get test dataloader
    test_loader = get_dataloader(config, split='test')
    
    # Evaluate models on their respective fog intensities
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating baseline models"):
            hazy_imgs = batch['hazy'].to(device)
            clear_imgs = batch['clear'].to(device)
            intensity_labels = batch['intensity']
            
            # Process each image based on its intensity
            for i in range(hazy_imgs.size(0)):
                # Select the model based on intensity
                if intensity_labels[i] == 0:  # Low intensity
                    model = low_model
                    category = 'low_intensity'
                elif intensity_labels[i] == 1:  # Medium intensity
                    model = medium_model
                    category = 'medium_intensity'
                else:  # High intensity
                    model = high_model
                    category = 'high_intensity'
                
                # Forward pass
                dehazed_img = model(hazy_imgs[i:i+1])
                
                # Add to metrics
                metrics.add_sample(dehazed_img[0], clear_imgs[i], category)
    
    # Print and save results
    results = metrics.print_results()
    metrics.save_results(os.path.join(config['evaluation']['results_dir'], 'baseline_results.json'))
    
    return results

def evaluate_joint_model(config):
    """Evaluate the joint adaptive fog intensity dehazing framework"""
    # Set device
    device = torch.device(config['device'])
    
    # Create models
    classifier = create_classifier(config).to(device)
    low_model = create_low_intensity_model(config).to(device)
    medium_model = create_medium_intensity_model(config).to(device)
    high_model = create_high_intensity_model(config).to(device)
    
    # Create router
    dehazing_models = {
        'low': low_model,
        'medium': medium_model,
        'high': high_model
    }
    router = create_router(dehazing_models, classifier, config).to(device)
    
    # Load weights from joint training
    joint_checkpoint = os.path.join(config['joint_training']['checkpoint_dir'], 'best_model.pth')
    
    if os.path.exists(joint_checkpoint):
        checkpoint = torch.load(joint_checkpoint, map_location='cpu')
        router.load_state_dict(checkpoint['router_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        low_model.load_state_dict(checkpoint['low_model_state_dict'])
        medium_model.load_state_dict(checkpoint['medium_model_state_dict'])
        high_model.load_state_dict(checkpoint['high_model_state_dict'])
        print(f"Loaded joint model from {joint_checkpoint}")
    else:
        print(f"Joint checkpoint {joint_checkpoint} not found. Loading individual models...")
        # Load individual models
        load_model_from_checkpoint(classifier, os.path.join(config['classifier']['checkpoint_dir'], 'best_model.pth'))
        load_model_from_checkpoint(low_model, os.path.join(config['dehazing']['checkpoint_dir'], 'low', 'best_model.pth'))
        load_model_from_checkpoint(medium_model, os.path.join(config['dehazing']['checkpoint_dir'], 'medium', 'best_model.pth'))
        load_model_from_checkpoint(high_model, os.path.join(config['dehazing']['checkpoint_dir'], 'high', 'best_model.pth'))
    
    # Set models to evaluation mode
    classifier.eval()
    for model in dehazing_models.values():
        model.eval()
    router.eval()
    
    # Initialize metrics
    metrics = ImageQualityMetrics(device=device)
    
    # Get test dataloader
    test_loader = get_dataloader(config, split='test')
    
    # Evaluate joint model
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating joint model"):
            hazy_imgs = batch['hazy'].to(device)
            clear_imgs = batch['clear'].to(device)
            intensity_labels = batch['intensity']
            
            # Forward pass through classifier
            logits, _ = classifier(hazy_imgs)
            
            # Forward pass through router
            dehazed_imgs, routing_info = router(hazy_imgs, logits)
            
            # Add to metrics
            for i in range(hazy_imgs.size(0)):
                # Determine intensity category
                if intensity_labels[i] == 0:  # Low intensity
                    category = 'low_intensity'
                elif intensity_labels[i] == 1:  # Medium intensity
                    category = 'medium_intensity'
                else:  # High intensity
                    category = 'high_intensity'
                
                # Add to metrics
                metrics.add_sample(dehazed_imgs[i], clear_imgs[i], category)
    
    # Print and save results
    results = metrics.print_results()
    metrics.save_results(os.path.join(config['evaluation']['results_dir'], 'joint_model_results.json'))
    
    # Generate visualizations for a few samples
    visualize_joint_model(router, classifier, config)
    
    return results

def evaluate_object_detection(config):
    """Evaluate the impact of dehazing on object detection performance"""
    # Set device
    device = torch.device(config['device'])
    
    # Create models
    classifier = create_classifier(config).to(device)
    low_model = create_low_intensity_model(config).to(device)
    medium_model = create_medium_intensity_model(config).to(device)
    high_model = create_high_intensity_model(config).to(device)
    detection_model = create_detection_model(config).to(device)
    
    # Create router
    dehazing_models = {
        'low': low_model,
        'medium': medium_model,
        'high': high_model
    }
    router = create_router(dehazing_models, classifier, config).to(device)
    
    # Create integrated system
    integrated_system = create_integrated_system(router, detection_model).to(device)
    
    # Load weights
    joint_checkpoint = os.path.join(config['joint_training']['checkpoint_dir'], 'best_model.pth')
    detection_checkpoint = os.path.join(config['detection']['checkpoint_dir'], 'best_model.pth')
    
    # Load joint model
    if os.path.exists(joint_checkpoint):
        checkpoint = torch.load(joint_checkpoint, map_location='cpu')
        router.load_state_dict(checkpoint['router_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        low_model.load_state_dict(checkpoint['low_model_state_dict'])
        medium_model.load_state_dict(checkpoint['medium_model_state_dict'])
        high_model.load_state_dict(checkpoint['high_model_state_dict'])
        print(f"Loaded joint model from {joint_checkpoint}")
    else:
        print(f"Joint checkpoint {joint_checkpoint} not found. Loading individual models...")
        # Load individual models
        load_model_from_checkpoint(classifier, os.path.join(config['classifier']['checkpoint_dir'], 'best_model.pth'))
        load_model_from_checkpoint(low_model, os.path.join(config['dehazing']['checkpoint_dir'], 'low', 'best_model.pth'))
        load_model_from_checkpoint(medium_model, os.path.join(config['dehazing']['checkpoint_dir'], 'medium', 'best_model.pth'))
        load_model_from_checkpoint(high_model, os.path.join(config['dehazing']['checkpoint_dir'], 'high', 'best_model.pth'))
    
    # Load detection model if available
    if os.path.exists(detection_checkpoint):
        detection_model.load_state_dict(torch.load(detection_checkpoint, map_location='cpu')['model_state_dict'])
        print(f"Loaded detection model from {detection_checkpoint}")
    
    # Set models to evaluation mode
    classifier.eval()
    for model in dehazing_models.values():
        model.eval()
    router.eval()
    detection_model.eval()
    integrated_system.eval()
    
    # Get detection dataloader
    detection_loader = get_detection_dataloader(config)
    
    # Initialize detection metrics
    # Check if annotation file exists
    annotation_file = os.path.join(config['dataset']['test_path'], 'annotations', 'instances.json')
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        print("Generating dummy annotations for testing...")
        
        # Generate empty annotation file with valid structure
        os.makedirs(os.path.dirname(annotation_file), exist_ok=True)
        dummy_annotations = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "bicycle", "supercategory": "vehicle"},
                {"id": 2, "name": "bus", "supercategory": "vehicle"},
                {"id": 3, "name": "car", "supercategory": "vehicle"},
                {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
                {"id": 5, "name": "person", "supercategory": "person"}
            ]
        }
        with open(annotation_file, 'w') as f:
            json.dump(dummy_annotations, f)
    # annotation_file = os.path.join(config['dataset']['test_path'], 'annotations', 'instances.json')
    
    hazy_detection_metrics = DetectionMetrics(annotation_file)
    dehazed_detection_metrics = DetectionMetrics(annotation_file)

    # Create a wrapper function to handle list inputs
    def process_batch(model, images):
        # If input is a list, process each image individually
        if isinstance(images, list):
            dehazed_imgs = []
            for img in images:
                # Add batch dimension
                img_batch = img.unsqueeze(0)
                # Process
                with torch.no_grad():
                    logits, _ = classifier(img_batch)
                    dehazed_img, _ = router(img_batch, logits)
                # Add to results
                dehazed_imgs.append(dehazed_img.squeeze(0))
            return dehazed_imgs, {}
        else:
            # Process as normal batch
            with torch.no_grad():
                logits, _ = classifier(images)
                return router(images, logits)
    
    # Evaluate detection performance
    with torch.no_grad():
        for images, targets, filenames, intensities in tqdm(detection_loader, desc="Evaluating object detection"):

            images = [img.to(device) for img in images]
            
            # Forward pass through detection model directly on hazy images
            hazy_results = detection_model(images)
            
            # Forward pass through integrated system (dehazing + detection)
            dehazed_results, dehazed_imgs = process_batch(router, images)
            
            # Process results
            for i, (hazy_pred, dehazed_pred, target) in enumerate(zip(hazy_results, dehazed_results, targets)):

                print(f"Debug - hazy_results type: {type(hazy_pred)}")
                if isinstance(hazy_pred, dict):
                    print(f"Keys: {hazy_pred.keys()}")
                elif isinstance(hazy_pred, torch.Tensor):
                    print(f"Shape: {hazy_pred.shape}")
                
                # Only process if format is correct
                if not isinstance(hazy_pred, dict) or 'boxes' not in hazy_pred:
                    print("Skipping invalid detection format")
                    continue
            
                image_id = target['image_id'].item()
                intensity = intensities[i]
                
                # Process hazy detections
                for box, label, score in zip(hazy_pred['boxes'], hazy_pred['labels'], hazy_pred['scores']):
                    if score > 0.5:  # Threshold
                        # Convert box to COCO format [x, y, width, height]
                        x1, y1, x2, y2 = box.cpu().numpy()
                        coco_box = [x1, y1, x2-x1, y2-y1]
                        
                        hazy_detection_metrics.add_detection_result(
                            image_id, 
                            label.item(), 
                            coco_box, 
                            score.item(),
                            intensity
                        )
                
                # Process dehazed detections
                for box, label, score in zip(dehazed_pred['boxes'], dehazed_pred['labels'], dehazed_pred['scores']):
                    if score > 0.5:  # Threshold
                        # Convert box to COCO format [x, y, width, height]
                        x1, y1, x2, y2 = box.cpu().numpy()
                        coco_box = [x1, y1, x2-x1, y2-y1]
                        
                        dehazed_detection_metrics.add_detection_result(
                            image_id, 
                            label.item(), 
                            coco_box, 
                            score.item(),
                            intensity
                        )
    
    # Evaluate results
    print("\nObject Detection on Hazy Images:")
    hazy_results = hazy_detection_metrics.evaluate_by_category()
    hazy_detection_metrics.print_results(hazy_results['overall'])
    
    print("\nObject Detection on Dehazed Images:")
    dehazed_results = dehazed_detection_metrics.evaluate_by_category()
    dehazed_detection_metrics.print_results(dehazed_results['overall'])
    
    # Save results
    hazy_detection_metrics.save_results(
        hazy_results,
        os.path.join(config['evaluation']['results_dir'], 'hazy_detection_results.json')
    )
    
    dehazed_detection_metrics.save_results(
        dehazed_results,
        os.path.join(config['evaluation']['results_dir'], 'dehazed_detection_results.json')
    )
    
    # Compare results by intensity
    print("\nComparison by Fog Intensity:")
    for intensity in ['low', 'medium', 'high']:
        if intensity in hazy_results and intensity in dehazed_results:
            hazy_map = hazy_results[intensity]['mAP']
            dehazed_map = dehazed_results[intensity]['mAP']
            improvement = (dehazed_map - hazy_map) / hazy_map * 100
            
            print(f"\n{intensity.capitalize()} Intensity:")
            print(f"  Hazy mAP: {hazy_map:.4f}")
            print(f"  Dehazed mAP: {dehazed_map:.4f}")
            print(f"  Improvement: {improvement:.2f}%")
    
    # Return results
    return {
        'hazy': hazy_results,
        'dehazed': dehazed_results
    }

def visualize_joint_model(router, classifier, config, num_samples=5):
    """Generate visualizations of joint model results"""
    # Set device
    device = torch.device(config['device'])
    
    # Get test dataloader
    test_loader = get_dataloader(config, split='test')
    
    # Create visualization directory
    vis_dir = config['evaluation']['visualization_dir']
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set models to evaluation mode
    classifier.eval()
    router.eval()
    
    # Get samples
    samples = []
    with torch.no_grad():
        for batch in test_loader:
            for i in range(min(batch['hazy'].size(0), num_samples)):
                # Add sample to list
                samples.append({
                    'hazy': batch['hazy'][i].to(device),
                    'clear': batch['clear'][i].to(device),
                    'intensity': batch['intensity'][i].item(),
                    'name': batch['name'][i]
                })
                
                if len(samples) >= num_samples * 3:  # Get samples from different batches
                    break
            
            if len(samples) >= num_samples * 3:
                break
    
    # Select samples from each intensity level
    low_samples = [s for s in samples if s['intensity'] == 0][:num_samples]
    med_samples = [s for s in samples if s['intensity'] == 1][:num_samples]
    high_samples = [s for s in samples if s['intensity'] == 2][:num_samples]
    
    # Process and visualize samples
    intensity_names = ['Low', 'Medium', 'High']
    
    for i, sample_group in enumerate([low_samples, med_samples, high_samples]):
        intensity_name = intensity_names[i]
        
        for j, sample in enumerate(sample_group):
            # Forward pass through classifier
            logits, _ = classifier(sample['hazy'].unsqueeze(0))
            
            # Forward pass through router
            dehazed_img, routing_info = router(sample['hazy'].unsqueeze(0), logits)
            dehazed_img = dehazed_img[0]
            
            # Get routing weights if using soft router
            if 'weights' in routing_info:
                weights = routing_info['weights'][0].detach().cpu().numpy()
                weight_str = f"Weights: Low={weights[0]:.2f}, Med={weights[1]:.2f}, High={weights[2]:.2f}"
            else:
                weight_str = "Hard routing"
            
            # Convert tensors to numpy arrays for visualization
            hazy_np = sample['hazy'].permute(1, 2, 0).cpu().numpy()
            dehazed_np = dehazed_img.permute(1, 2, 0).detach().cpu().numpy()
            clear_np = sample['clear'].permute(1, 2, 0).cpu().numpy()
            
            # Create visualization grid
            grid = create_comparison_grid(
                hazy_np, 
                dehazed_np, 
                clear_np,
                titles=['Hazy', 'Dehazed', 'Ground Truth'],
                suptitle=f"{intensity_name} Fog Intensity: {sample['name']} - {weight_str}"
            )
            
            # Save visualization
            plt.savefig(os.path.join(vis_dir, f"{intensity_name.lower()}_sample_{j+1}.png"))
            plt.close()

def run_comprehensive_evaluation(config):
    """Run all evaluation methods"""
    # Create results directory
    os.makedirs(config['evaluation']['results_dir'], exist_ok=True)
    os.makedirs(config['evaluation']['visualization_dir'], exist_ok=True)
    
    print("=" * 50)
    print("ADAPTIVE FOG INTENSITY DEHAZING FRAMEWORK EVALUATION")
    print("=" * 50)
    
    # 1. Evaluate baseline models
    print("\n1. Evaluating Individual Dehazing Models:")
    print("-" * 50)
    baseline_results = evaluate_baseline_models(config)
    
    # 2. Evaluate joint model
    print("\n2. Evaluating Adaptive Framework:")
    print("-" * 50)
    joint_results = evaluate_joint_model(config)
    
    # 3. Evaluate object detection impact
    print("\n3. Evaluating Impact on Object Detection:")
    print("-" * 50)
    detection_results = evaluate_object_detection(config)
    
    # Generate comparison report
    print("\n4. Comparison Summary:")
    print("-" * 50)
    
    # Compare overall image quality
    baseline_avg_psnr = np.mean([
        baseline_results['low_intensity']['psnr'],
        baseline_results['medium_intensity']['psnr'],
        baseline_results['high_intensity']['psnr']
    ])
    
    joint_avg_psnr = np.mean([
        joint_results['low_intensity']['psnr'],
        joint_results['medium_intensity']['psnr'],
        joint_results['high_intensity']['psnr']
    ])
    
    # Compare detection performance
    hazy_map = detection_results['hazy']['overall']['mAP']
    dehazed_map = detection_results['dehazed']['overall']['mAP']
    detection_improvement = (dehazed_map - hazy_map) / hazy_map * 100
    
    print(f"Image Quality Comparison:")
    print(f"  Baseline Models Avg PSNR: {baseline_avg_psnr:.2f} dB")
    print(f"  Adaptive Framework Avg PSNR: {joint_avg_psnr:.2f} dB")
    print(f"  Improvement: {(joint_avg_psnr - baseline_avg_psnr):.2f} dB")
    
    print(f"\nObject Detection Comparison:")
    print(f"  Detection on Hazy Images mAP: {hazy_map:.4f}")
    print(f"  Detection on Dehazed Images mAP: {dehazed_map:.4f}")
    print(f"  Improvement: {detection_improvement:.2f}%")
    
    # Save comprehensive results
    comprehensive_results = {
        'baseline': baseline_results,
        'joint': joint_results,
        'detection': {
            'hazy': detection_results['hazy']['overall'],
            'dehazed': detection_results['dehazed']['overall'],
            'improvement_percent': detection_improvement
        },
        'comparison': {
            'baseline_avg_psnr': baseline_avg_psnr,
            'joint_avg_psnr': joint_avg_psnr,
            'psnr_improvement': joint_avg_psnr - baseline_avg_psnr
        }
    }
    
    with open(os.path.join(config['evaluation']['results_dir'], 'comprehensive_results.json'), 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\nComprehensive evaluation results saved to {config['evaluation']['results_dir']}/comprehensive_results.json")

if __name__ == "__main__":
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Run comprehensive evaluation
    run_comprehensive_evaluation(config)