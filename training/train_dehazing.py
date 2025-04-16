import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from data.dataset import get_dataloader
from models.dehazing.low_intensity import create_low_intensity_model
from models.dehazing.medium_intensity import create_medium_intensity_model
from models.dehazing.high_intensity import create_high_intensity_model
from training.loss import get_dehazing_loss

def train_dehazing_model(model, intensity_level, config):
    """
    Train a dehazing model for a specific intensity level
    
    Args:
        model (nn.Module): The dehazing model to train
        intensity_level (str): 'low', 'medium', or 'high'
        config (dict): Configuration dictionary
        
    Returns:
        nn.Module: Trained dehazing model
    """
    # Set device
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['dehazing'][intensity_level]['learning_rate'],
        weight_decay=0.0001
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    criterion = get_dehazing_loss(config)
    criterion = criterion.to(device)
    
    # Dataloaders - filter by intensity level
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # Setup tensorboard
    log_dir = os.path.join('logs', 'dehazing', intensity_level)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(config['dehazing']['checkpoint_dir'], intensity_level)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_psnr = 0.0
    epochs = 30  # Specific epochs for dehazing models
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Train for one epoch
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as t:
            for batch_idx, batch in enumerate(t):
                # Filter based on intensity level
                intensity_indices = (batch['intensity'] == int({'low': 0, 'medium': 1, 'high': 2}[intensity_level]))
                if not torch.any(intensity_indices):
                    continue
                
                # Get the inputs for the current intensity level
                hazy_imgs = batch['hazy'][intensity_indices].to(device)
                clear_imgs = batch['clear'][intensity_indices].to(device)
                
                # Skip if batch is empty after filtering
                if hazy_imgs.size(0) == 0:
                    continue
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                dehazed_imgs = model(hazy_imgs)
                
                # Calculate loss
                loss, loss_components = criterion(dehazed_imgs, clear_imgs)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                
                # Update progress bar
                t.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'l1': loss_components['l1'].item(),
                    'perceptual': loss_components['perceptual'].item()
                })
        
        # Compute epoch-level training metrics
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_perceptual = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Filter based on intensity level
                intensity_indices = (batch['intensity'] == int({'low': 0, 'medium': 1, 'high': 2}[intensity_level]))
                if not torch.any(intensity_indices):
                    continue
                
                # Get the inputs for the current intensity level
                hazy_imgs = batch['hazy'][intensity_indices].to(device)
                clear_imgs = batch['clear'][intensity_indices].to(device)
                
                # Skip if batch is empty after filtering
                if hazy_imgs.size(0) == 0:
                    continue
                
                # Forward pass
                dehazed_imgs = model(hazy_imgs)
                
                # Calculate loss
                loss, loss_components = criterion(dehazed_imgs, clear_imgs)
                
                # Update statistics
                val_loss += loss.item() * hazy_imgs.size(0)
                val_perceptual += loss_components['perceptual'].item() * hazy_imgs.size(0)
                val_samples += hazy_imgs.size(0)
                
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
        
        # Compute validation metrics
        if val_samples > 0:
            val_loss /= val_samples
            val_psnr /= val_samples
            val_ssim /= val_samples
            val_perceptual /= val_samples
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('PSNR/val', val_psnr, epoch)
        writer.add_scalar('SSIM/val', val_ssim, epoch)
        writer.add_scalar('Perceptual/val', val_perceptual, epoch)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}")
        
        # Log example images
        if epoch % 5 == 0 and val_samples > 0:
            # Add some example images to tensorboard
            n_samples = min(4, hazy_imgs.size(0))
            for i in range(n_samples):
                writer.add_image(f'hazy_{i}', hazy_imgs[i], epoch)
                writer.add_image(f'dehazed_{i}', dehazed_imgs[i], epoch)
                writer.add_image(f'clear_{i}', clear_imgs[i], epoch)
        
        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model with validation PSNR: {val_psnr:.2f} dB")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Close tensorboard writer
    writer.close()
    
    # Load best model for evaluation
    best_checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Return the trained model
    return model

def train_all_dehazing_models(config):
    """Train all three dehazing models for different intensity levels"""
    print("Training Low Intensity Dehazing Model...")
    low_model = create_low_intensity_model(config)
    trained_low_model = train_dehazing_model(low_model, 'low', config)
    
    print("Training Medium Intensity Dehazing Model...")
    medium_model = create_medium_intensity_model(config)
    trained_medium_model = train_dehazing_model(medium_model, 'medium', config)
    
    print("Training High Intensity Dehazing Model...")
    high_model = create_high_intensity_model(config)
    trained_high_model = train_dehazing_model(high_model, 'high', config)
    
    return {
        'low': trained_low_model,
        'medium': trained_medium_model,
        'high': trained_high_model
    }

def evaluate_dehazing_model(model, intensity_level, config):
    """Evaluate a dehazing model on the test set"""
    # Set device
    device = torch.device(config['device'])
    model = model.to(device)
    model.eval()
    
    # Dataloader
    test_loader = get_dataloader(config, split='test')
    
    # Metrics
    test_psnr = 0.0
    test_ssim = 0.0
    test_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {intensity_level} model"):
            # Filter based on intensity level
            intensity_indices = (batch['intensity'] == int({'low': 0, 'medium': 1, 'high': 2}[intensity_level]))
            if not torch.any(intensity_indices):
                continue
            
            # Get the inputs for the current intensity level
            hazy_imgs = batch['hazy'][intensity_indices].to(device)
            clear_imgs = batch['clear'][intensity_indices].to(device)
            
            # Skip if batch is empty after filtering
            if hazy_imgs.size(0) == 0:
                continue
            
            # Forward pass
            dehazed_imgs = model(hazy_imgs)
            
            # Calculate image quality metrics
            for i in range(hazy_imgs.size(0)):
                # Convert to numpy for PSNR/SSIM calculation
                pred_np = dehazed_imgs[i].permute(1, 2, 0).cpu().numpy()
                target_np = clear_imgs[i].permute(1, 2, 0).cpu().numpy()
                
                # Calculate PSNR
                psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
                test_psnr += psnr
                
                # Calculate SSIM (convert to grayscale first)
                gray_pred = np.mean(pred_np, axis=2)
                gray_target = np.mean(target_np, axis=2)
                ssim = structural_similarity(gray_target, gray_pred, data_range=1.0)
                test_ssim += ssim
                
                test_samples += 1
    
    # Compute metrics
    if test_samples > 0:
        test_psnr /= test_samples
        test_ssim /= test_samples
    
    # Print results
    print(f"{intensity_level.capitalize()} Intensity Dehazing Results:")
    print(f"  Test PSNR: {test_psnr:.2f} dB")
    print(f"  Test SSIM: {test_ssim:.4f}")
    
    # Return metrics
    return {
        'psnr': test_psnr,
        'ssim': test_ssim,
        'samples': test_samples
    }

if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Train all dehazing models
    models = train_all_dehazing_models(config)
    
    # Evaluate all models
    results = {}
    for level, model in models.items():
        results[level] = evaluate_dehazing_model(model, level, config)
    
    # Print combined results
    print("\nSummary of Results:")
    for level, metrics in results.items():
        print(f"{level.capitalize()} Intensity: PSNR = {metrics['psnr']:.2f} dB, SSIM = {metrics['ssim']:.4f}")