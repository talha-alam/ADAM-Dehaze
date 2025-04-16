import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from data.dataset import get_dataloader
from models.classifier import create_classifier
from models.dehazing.low_intensity import create_low_intensity_model
from models.dehazing.medium_intensity import create_medium_intensity_model
from models.dehazing.high_intensity import create_high_intensity_model
from models.routing import create_router
from training.loss import get_joint_loss

def load_pretrained_model(model, checkpoint_path):
    """Load pretrained weights into a model"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {checkpoint_path}")
        return True
    else:
        print(f"Checkpoint {checkpoint_path} not found. Starting with random weights.")
        return False

def train_joint_model(config):
    """Train the joint adaptive fog intensity dehazing framework"""
    # Set device
    device = torch.device(config['device'])
    
    # Initialize models
    print("Creating classifier model...")
    classifier = create_classifier(config)
    
    print("Creating dehazing models...")
    low_model = create_low_intensity_model(config)
    medium_model = create_medium_intensity_model(config)
    high_model = create_high_intensity_model(config)
    
    # Load pretrained weights if available
    load_pretrained_model(
        classifier, 
        os.path.join(config['classifier']['checkpoint_dir'], 'best_model.pth')
    )
    
    load_pretrained_model(
        low_model,
        os.path.join(config['dehazing']['checkpoint_dir'], 'low', 'best_model.pth')
    )
    
    load_pretrained_model(
        medium_model,
        os.path.join(config['dehazing']['checkpoint_dir'], 'medium', 'best_model.pth')
    )
    
    load_pretrained_model(
        high_model,
        os.path.join(config['dehazing']['checkpoint_dir'], 'high', 'best_model.pth')
    )
    
    # Create the router
    dehazing_models = {
        'low': low_model,
        'medium': medium_model,
        'high': high_model
    }
    
    print("Creating routing mechanism...")
    router = create_router(dehazing_models, classifier, config)
    
    # Move all models to device
    classifier = classifier.to(device)
    for model in dehazing_models.values():
        model = model.to(device)
    router = router.to(device)
    
    # Initialize optimizer (only fine-tune models)
    params = list(router.parameters())
    for model in dehazing_models.values():
        params.extend(list(model.parameters()))
    
    optimizer = optim.Adam(
        params,
        lr=config['joint_training']['learning_rate'],
        weight_decay=0.0001
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Loss function
    criterion = get_joint_loss(config)
    criterion = criterion.to(device)
    
    # Dataloaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # Setup tensorboard
    log_dir = os.path.join('logs', 'joint')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Create checkpoint directory
    os.makedirs(config['joint_training']['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    best_val_psnr = 0.0
    epochs = config['joint_training']['epochs']
    
    for epoch in range(epochs):
        # Set all models to training mode
        classifier.train()
        for model in dehazing_models.values():
            model.train()
        router.train()
        
        train_loss = 0.0
        train_dehaze_loss = 0.0
        train_class_loss = 0.0
        
        # Train for one epoch
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as t:
            for batch_idx, batch in enumerate(t):
                # Get the inputs
                hazy_imgs = batch['hazy'].to(device)
                clear_imgs = batch['clear'].to(device)
                intensity_labels = batch['intensity'].to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass through classifier
                logits, _ = classifier(hazy_imgs)
                
                # Forward pass through router
                dehazed_imgs, routing_info = router(hazy_imgs, logits)
                
                # Calculate joint loss
                loss, loss_components = criterion(
                    dehazed_imgs, 
                    clear_imgs, 
                    logits, 
                    intensity_labels
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                train_dehaze_loss += loss_components['dehazing'].item()
                train_class_loss += loss_components['classification'].item()
                
                # Update progress bar
                t.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'dehaze': train_dehaze_loss / (batch_idx + 1),
                    'class': train_class_loss / (batch_idx + 1)
                })
        
        # Compute epoch-level training metrics
        train_loss /= len(train_loader)
        train_dehaze_loss /= len(train_loader)
        train_class_loss /= len(train_loader)
        
        # Validation
        # Set all models to evaluation mode
        classifier.eval()
        for model in dehazing_models.values():
            model.eval()
        router.eval()
        
        val_loss = 0.0
        val_dehaze_loss = 0.0
        val_class_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get the inputs
                hazy_imgs = batch['hazy'].to(device)
                clear_imgs = batch['clear'].to(device)
                intensity_labels = batch['intensity'].to(device)
                
                # Forward pass through classifier
                logits, _ = classifier(hazy_imgs)
                
                # Forward pass through router
                dehazed_imgs, routing_info = router(hazy_imgs, logits)
                
                # Calculate joint loss
                loss, loss_components = criterion(
                    dehazed_imgs, 
                    clear_imgs, 
                    logits, 
                    intensity_labels
                )
                
                # Update statistics
                val_loss += loss.item() * hazy_imgs.size(0)
                val_dehaze_loss += loss_components['dehazing'].item() * hazy_imgs.size(0)
                val_class_loss += loss_components['classification'].item() * hazy_imgs.size(0)
                
                # Calculate image quality metrics
                for i in range(hazy_imgs.size(0)):
                    # Convert to numpy for PSNR/SSIM calculation
                    pred_np = dehazed_imgs[i].permute(1, 2, 0).cpu().numpy()
                    target_np = clear_imgs[i].permute(1, 2, 0).cpu().numpy()
                    
                    # Calculate PSNR
                    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
                    val_psnr += psnr
                    
                    # Calculate SSIM (convert to grayscale first)
                    gray_pred = np.mean(pred_np, axis=2)
                    gray_target = np.mean(target_np, axis=2)
                    ssim = structural_similarity(gray_target, gray_pred, data_range=1.0)
                    val_ssim += ssim
                
                val_samples += hazy_imgs.size(0)
        
        # Compute validation metrics
        val_loss /= val_samples
        val_dehaze_loss /= val_samples
        val_class_loss /= val_samples
        val_psnr /= val_samples
        val_ssim /= val_samples
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train_dehaze', train_dehaze_loss, epoch)
        writer.add_scalar('Loss/val_dehaze', val_dehaze_loss, epoch)
        writer.add_scalar('Loss/train_class', train_class_loss, epoch)
        writer.add_scalar('Loss/val_class', val_class_loss, epoch)
        writer.add_scalar('PSNR/val', val_psnr, epoch)
        writer.add_scalar('SSIM/val', val_ssim, epoch)
        
        # Log example images
        if epoch % 5 == 0:
            # Add some example images to tensorboard
            n_samples = min(4, hazy_imgs.size(0))
            for i in range(n_samples):
                writer.add_image(f'hazy_{i}', hazy_imgs[i], epoch)
                writer.add_image(f'dehazed_{i}', dehazed_imgs[i], epoch)
                writer.add_image(f'clear_{i}', clear_imgs[i], epoch)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} (Dehaze: {train_dehaze_loss:.4f}, Class: {train_class_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Dehaze: {val_dehaze_loss:.4f}, Class: {val_class_loss:.4f})")
        print(f"  Val PSNR: {val_psnr:.2f} dB, Val SSIM: {val_ssim:.4f}")
        
        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            checkpoint_path = os.path.join(
                config['joint_training']['checkpoint_dir'], 'best_model.pth'
            )
            torch.save({
                'epoch': epoch,
                'router_state_dict': router.state_dict(),
                'low_model_state_dict': dehazing_models['low'].state_dict(),
                'medium_model_state_dict': dehazing_models['medium'].state_dict(),
                'high_model_state_dict': dehazing_models['high'].state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model with validation PSNR: {val_psnr:.2f} dB")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                config['joint_training']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'router_state_dict': router.state_dict(),
                'low_model_state_dict': dehazing_models['low'].state_dict(),
                'medium_model_state_dict': dehazing_models['medium'].state_dict(),
                'high_model_state_dict': dehazing_models['high'].state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Close tensorboard writer
    writer.close()
    
    # Load best model for evaluation
    best_checkpoint = torch.load(os.path.join(
        config['joint_training']['checkpoint_dir'], 'best_model.pth'
    ))
    router.load_state_dict(best_checkpoint['router_state_dict'])
    dehazing_models['low'].load_state_dict(best_checkpoint['low_model_state_dict'])
    dehazing_models['medium'].load_state_dict(best_checkpoint['medium_model_state_dict'])
    dehazing_models['high'].load_state_dict(best_checkpoint['high_model_state_dict'])
    classifier.load_state_dict(best_checkpoint['classifier_state_dict'])
    
    # Return the trained models
    return router, dehazing_models, classifier

def evaluate_joint_model(router, classifier, config):
    """Evaluate the joint model on the test set"""
    # Set device
    device = torch.device(config['device'])
    router = router.to(device)
    classifier = classifier.to(device)
    
    # Set models to evaluation mode
    router.eval()
    classifier.eval()
    
    # Dataloader
    test_loader = get_dataloader(config, split='test')
    
    # Metrics
    test_psnr = 0.0
    test_ssim = 0.0
    
    # Metrics by intensity level
    intensity_metrics = {
        'low': {'psnr': 0.0, 'ssim': 0.0, 'samples': 0},
        'medium': {'psnr': 0.0, 'ssim': 0.0, 'samples': 0},
        'high': {'psnr': 0.0, 'ssim': 0.0, 'samples': 0},
    }
    
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating joint model"):
            # Get the inputs
            hazy_imgs = batch['hazy'].to(device)
            clear_imgs = batch['clear'].to(device)
            intensity_labels = batch['intensity'].to(device)
            
            # Forward pass through classifier
            logits, _ = classifier(hazy_imgs)
            
            # Forward pass through router
            dehazed_imgs, routing_info = router(hazy_imgs, logits)
            
            # Calculate image quality metrics
            for i in range(hazy_imgs.size(0)):
                # Convert to numpy for PSNR/SSIM calculation
                pred_np = dehazed_imgs[i].permute(1, 2, 0).cpu().numpy()
                target_np = clear_imgs[i].permute(1, 2, 0).cpu().numpy()
                
                # Get intensity level
                intensity_idx = intensity_labels[i].item()
                intensity_name = ['low', 'medium', 'high'][intensity_idx]
                
                # Calculate PSNR
                psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
                test_psnr += psnr
                intensity_metrics[intensity_name]['psnr'] += psnr
                
                # Calculate SSIM (convert to grayscale first)
                gray_pred = np.mean(pred_np, axis=2)
                gray_target = np.mean(target_np, axis=2)
                ssim = structural_similarity(gray_target, gray_pred, data_range=1.0)
                test_ssim += ssim
                intensity_metrics[intensity_name]['ssim'] += ssim
                
                # Increment sample counters
                intensity_metrics[intensity_name]['samples'] += 1
                total_samples += 1
    
    # Compute overall metrics
    test_psnr /= total_samples
    test_ssim /= total_samples
    
    # Compute metrics by intensity level
    for level in intensity_metrics:
        if intensity_metrics[level]['samples'] > 0:
            intensity_metrics[level]['psnr'] /= intensity_metrics[level]['samples']
            intensity_metrics[level]['ssim'] /= intensity_metrics[level]['samples']
    
    # Print results
    print("Joint Model Evaluation Results:")
    print(f"  Overall PSNR: {test_psnr:.2f} dB")
    print(f"  Overall SSIM: {test_ssim:.4f}")
    print("\nResults by Intensity Level:")
    for level, metrics in intensity_metrics.items():
        if metrics['samples'] > 0:
            print(f"  {level.capitalize()} Intensity ({metrics['samples']} samples):")
            print(f"    PSNR: {metrics['psnr']:.2f} dB")
            print(f"    SSIM: {metrics['ssim']:.4f}")
    
    # Return metrics
    return {
        'overall': {
            'psnr': test_psnr,
            'ssim': test_ssim,
            'samples': total_samples
        },
        'by_intensity': intensity_metrics
    }

if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Train joint model
    router, dehazing_models, classifier = train_joint_model(config)
    
    # Evaluate joint model
    evaluate_joint_model(router, classifier, config)