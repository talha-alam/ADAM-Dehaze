import os
import torch
import numpy as np
import yaml
import random
import time
import cv2
from pathlib import Path

def seed_everything(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_experiment_dir(config, exp_name=None):
    """Create directory for experiment results"""
    if exp_name is None:
        exp_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Create main experiment directory
    exp_dir = os.path.join('experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    logs_dir = os.path.join(exp_dir, 'logs')
    results_dir = os.path.join(exp_dir, 'results')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Update config with new paths
    config['classifier']['checkpoint_dir'] = os.path.join(checkpoints_dir, 'classifier')
    config['dehazing']['checkpoint_dir'] = os.path.join(checkpoints_dir, 'dehazing')
    config['routing']['checkpoint_dir'] = os.path.join(checkpoints_dir, 'routing')
    config['joint_training']['checkpoint_dir'] = os.path.join(checkpoints_dir, 'joint')
    config['detection']['checkpoint_dir'] = os.path.join(checkpoints_dir, 'detection')
    
    config['evaluation']['results_dir'] = os.path.join(results_dir, 'metrics')
    config['evaluation']['visualization_dir'] = os.path.join(results_dir, 'visualizations')
    
    # Save the updated config
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    return exp_dir, config

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer from checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint on CPU to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Load optimizer if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Return additional info if available
        additional_info = {}
        for key in ['epoch', 'val_loss', 'val_psnr', 'val_ssim', 'val_acc']:
            if key in checkpoint:
                additional_info[key] = checkpoint[key]
        
        print(f"Checkpoint loaded successfully. Additional info: {additional_info}")
        return True, additional_info
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return False, {}

def save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path):
    """Save model and optimizer to checkpoint"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Add validation metrics if provided
    if val_metrics:
        checkpoint.update(val_metrics)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def get_learning_rate(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def calculate_inference_time(model, input_shape=(1, 3, 256, 256), device='cuda', n_samples=100):
    """Calculate average inference time"""
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_samples):
            _ = model(dummy_input)
    end_time = time.time()
    
    # Calculate average time
    avg_time = (end_time - start_time) / n_samples
    return avg_time

def get_gpu_memory_usage():
    """Get GPU memory usage in MB"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        return {
            'allocated': memory_allocated,
            'reserved': memory_reserved
        }
    else:
        return None

def create_mask_from_transmission(hazy_img, beta=1.0):
    """Create synthetic fog density mask from transmission estimation"""
    # Convert to grayscale and normalize
    if len(hazy_img.shape) == 3 and hazy_img.shape[2] == 3:
        gray = cv2.cvtColor(hazy_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = hazy_img.copy()
    
    # Normalize
    if gray.max() > 1.0:
        gray = gray / 255.0
    
    # Apply DCP (Dark Channel Prior) to estimate transmission
    patch_size = 15
    dark_channel = np.min(gray, axis=2) if len(gray.shape) == 3 else gray
    
    # Apply minimum filter
    dark_channel = cv2.erode(dark_channel, np.ones((patch_size, patch_size)))
    
    # Estimate transmission
    A = np.max(dark_channel)  # Atmospheric light
    omega = 0.95
    transmission = 1 - omega * dark_channel / max(A, 0.1)
    
    # Apply guided filter for refinement
    if len(hazy_img.shape) == 3:
        refined_transmission = cv2.ximgproc.guidedFilter(
            guide=cv2.cvtColor((hazy_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
            src=(transmission * 255).astype(np.uint8),
            radius=40,
            eps=1e-3
        ) / 255.0
    else:
        refined_transmission = cv2.ximgproc.guidedFilter(
            guide=(hazy_img * 255).astype(np.uint8),
            src=(transmission * 255).astype(np.uint8),
            radius=40,
            eps=1e-3
        ) / 255.0
    
    # Apply beta to control fog density
    fog_mask = np.exp(-beta * refined_transmission)
    
    return fog_mask

def apply_random_fog(clear_img, intensity='random'):
    """Apply synthetic fog of specified intensity to a clear image"""
    # Convert to numpy if tensor
    if isinstance(clear_img, torch.Tensor):
        is_tensor = True
        if clear_img.dim() == 4:  # Batch of images
            batch_size = clear_img.size(0)
            result = []
            for i in range(batch_size):
                img_np = clear_img[i].permute(1, 2, 0).cpu().numpy()
                result.append(apply_random_fog(img_np, intensity))
            return torch.stack(result)
        else:  # Single image
            img_np = clear_img.permute(1, 2, 0).cpu().numpy()
    else:
        is_tensor = False
        img_np = clear_img.copy()
    
    # Normalize image to [0, 1]
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    # Set fog parameters based on intensity
    if intensity == 'low':
        beta_range = (0.1, 0.4)
        A_range = (0.5, 0.7)
    elif intensity == 'medium':
        beta_range = (0.4, 0.7)
        A_range = (0.7, 0.9)
    elif intensity == 'high':
        beta_range = (0.7, 1.0)
        A_range = (0.8, 1.0)
    else:  # random
        beta_range = (0.1, 1.0)
        A_range = (0.5, 1.0)
    
    # Sample random parameters
    beta = np.random.uniform(*beta_range)
    A = np.random.uniform(*A_range)
    
    # Create depth map (simple approximation - farther objects have lower values)
    h, w = img_np.shape[:2]
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    depth_map = 0.3 + 0.7 * np.sqrt((xx - 0.5)**2 + (yy - 0.2)**2)
    
    # Calculate transmission based on depth
    transmission = np.exp(-beta * depth_map)
    
    # Apply atmospheric scattering model: I = J * t + A * (1 - t)
    # where I is hazy image, J is clear image, t is transmission, A is atmospheric light
    hazy_img = np.zeros_like(img_np)
    for c in range(3):  # Apply to each channel
        hazy_img[..., c] = img_np[..., c] * transmission + A * (1 - transmission)
    
    # Clip values to [0, 1]
    hazy_img = np.clip(hazy_img, 0, 1)
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        hazy_tensor = torch.from_numpy(hazy_img).permute(2, 0, 1)
        return hazy_tensor
    else:
        return hazy_img

def create_progressive_test_set(clear_imgs_dir, output_dir, fog_levels=5):
    """Create a test set with progressively increasing fog intensity"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all clear images
    clear_paths = list(Path(clear_imgs_dir).glob('*.jpg')) + list(Path(clear_imgs_dir).glob('*.png'))
    
    # Process each clear image
    for img_path in clear_paths:
        # Load clear image
        clear_img = cv2.imread(str(img_path))
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        
        # Create foggy versions with increasing intensity
        for i in range(fog_levels):
            # Calculate fog intensity
            fog_intensity = (i + 1) / fog_levels
            beta = 0.1 + 0.9 * fog_intensity
            A = 0.5 + 0.5 * fog_intensity
            
            # Apply fog
            hazy_img = apply_random_fog(clear_img, beta, A)
            
            # Save result
            output_name = f"{img_path.stem}_fog{i+1}.png"
            output_path = os.path.join(output_dir, output_name)
            
            # Convert to BGR for cv2 saving
            hazy_img_bgr = cv2.cvtColor((hazy_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, hazy_img_bgr)
    
    print(f"Created progressive test set with {fog_levels} fog levels for {len(clear_paths)} images")