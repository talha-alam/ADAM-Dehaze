import os
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from utils.helpers import seed_everything, create_experiment_dir
from data.preprocessing import preprocess_dataset, split_dataset
from training.train_classifier import train_classifier, evaluate_classifier
from training.train_dehazing import train_all_dehazing_models, evaluate_dehazing_model
from training.train_joint import train_joint_model, evaluate_joint_model
from evaluation.evaluate import run_comprehensive_evaluation
from models.classifier import create_classifier
from models.dehazing.low_intensity import create_low_intensity_model
from models.dehazing.medium_intensity import create_medium_intensity_model
from models.dehazing.high_intensity import create_high_intensity_model
from models.routing import create_router
from models.detection import create_detection_model

def update_checkpoint_paths(config, experiment_dir):
    """Update checkpoint paths to point to correct experiment directory"""
    config['classifier']['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints/classifier')
    config['dehazing']['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints/dehazing')
    config['routing']['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints/routing')
    config['joint_training']['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints/joint')
    return config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Adaptive Fog Intensity Dehazing Framework')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    
    parser.add_argument('--mode', type=str, default='train_all',
                        choices=['preprocess', 'train_classifier', 'train_dehazing', 
                                'train_joint', 'train_all', 'evaluate', 'demo'],
                        help='Operation mode')
    
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset directory (overrides config)')
    
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu, overrides config)')
    
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['dataset']['train_path'] = os.path.join(args.data_dir, 'train')
        config['dataset']['val_path'] = os.path.join(args.data_dir, 'val')
        config['dataset']['test_path'] = os.path.join(args.data_dir, 'test')
    
    if args.device:
        config['device'] = args.device
    
    if args.seed:
        config['seed'] = args.seed
    
    # Create experiment directory and update config paths
    exp_dir, config = create_experiment_dir(config, args.exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Set random seed
    seed_everything(config['seed'])
    print(f"Random seed set to {config['seed']}")
    
    # Execute based on mode
    if args.mode == 'preprocess':
        # Preprocess dataset
        print("Preprocessing dataset...")
        data_dir = Path(config['dataset']['train_path']).parent.parent
        raw_dir = os.path.join(data_dir, 'raw')
        processed_dir = os.path.join(data_dir, 'processed')
        
        preprocess_dataset(raw_dir, processed_dir, size=config['dataset']['img_size'])
        split_dataset(processed_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=config['seed'])
        
        print("Dataset preprocessing completed!")
    
    elif args.mode == 'train_classifier':
        # Train fog intensity classifier
        print("Training fog intensity classifier...")
        classifier = train_classifier(config)
        evaluate_classifier(classifier, config)
    
    elif args.mode == 'train_dehazing':
        # Train individual dehazing models
        print("Training dehazing models...")
        dehazing_models = train_all_dehazing_models(config)
        
        # Evaluate each model
        for level, model in dehazing_models.items():
            print(f"Evaluating {level} intensity model...")
            evaluate_dehazing_model(model, level, config)
    
    elif args.mode == 'train_joint':
        # Train joint model
        print("Training joint model...")
        router, dehazing_models, classifier = train_joint_model(config)
        evaluate_joint_model(router, classifier, config)
    
    elif args.mode == 'train_all':
        # Execute the complete training pipeline
        
        # 1. Train fog intensity classifier
        print("\n===== Step 1: Training Fog Intensity Classifier =====")
        classifier = train_classifier(config)
        evaluate_classifier(classifier, config)
        
        # 2. Train individual dehazing models
        print("\n===== Step 2: Training Dehazing Models =====")
        dehazing_models = train_all_dehazing_models(config)
        
        # 3. Train joint model
        print("\n===== Step 3: Training Joint Model =====")
        router, dehazing_models, classifier = train_joint_model(config)
        
        # 4. Evaluate the complete system
        print("\n===== Step 4: Comprehensive Evaluation =====")
        run_comprehensive_evaluation(config)
    
    elif args.mode == 'evaluate':
        # If you have a specific experiment to evaluate
        experiment_dir = "/home/coder/mohammed.alam/_dehazing/adaptive_dehazing/experiments/experiment_20250319_152941"  # Change this to your experiment folder
        config = update_checkpoint_paths(config, experiment_dir)
        print("Running comprehensive evaluation...")
        run_comprehensive_evaluation(config)
        # # Run comprehensive evaluation
        # print("Running comprehensive evaluation...")
        # run_comprehensive_evaluation(config)
    
    elif args.mode == 'demo':
        # Demo mode - load models and run inference on sample images
        print("Running demo...")
        
        # Create demo directory
        demo_dir = os.path.join(exp_dir, 'demo')
        os.makedirs(demo_dir, exist_ok=True)
        
        # Load models
        device = torch.device(config['device'])
        
        # Load classifier
        classifier = create_classifier(config).to(device)
        classifier_checkpoint = os.path.join(config['classifier']['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(classifier_checkpoint):
            checkpoint = torch.load(classifier_checkpoint, map_location=device)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded classifier from {classifier_checkpoint}")
        else:
            print(f"Warning: Classifier checkpoint not found at {classifier_checkpoint}")
        
        # Load dehazing models
        low_model = create_low_intensity_model(config).to(device)
        medium_model = create_medium_intensity_model(config).to(device)
        high_model = create_high_intensity_model(config).to(device)
        
        dehazing_models = {
            'low': low_model,
            'medium': medium_model,
            'high': high_model
        }
        
        # Load dehazing model checkpoints
        for level, model in dehazing_models.items():
            checkpoint_path = os.path.join(config['dehazing']['checkpoint_dir'], level, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded {level} intensity model from {checkpoint_path}")
            else:
                print(f"Warning: {level.capitalize()} intensity model checkpoint not found")
        
        # Create router
        router = create_router(dehazing_models, classifier, config).to(device)
        
        # Load router if available
        router_checkpoint = os.path.join(config['joint_training']['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(router_checkpoint):
            checkpoint = torch.load(router_checkpoint, map_location=device)
            router.load_state_dict(checkpoint['router_state_dict'])
            print(f"Loaded router from {router_checkpoint}")
        
        # Set models to evaluation mode
        classifier.eval()
        for model in dehazing_models.values():
            model.eval()
        router.eval()
        
        # TODO: Add demo code to process sample images
        # The demo functionality would involve:
        # 1. Loading sample hazy images
        # 2. Running them through the classifier
        # 3. Dehazing using the router or individual models
        # 4. Visualizing the results
        
        print("Demo completed. Results saved to:", demo_dir)
    
    print(f"All tasks completed successfully! Results are available in: {exp_dir}")

if __name__ == "__main__":
    main()