import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.utils import make_grid
import matplotlib.patches as patches

def tensor_to_numpy(tensor):
    """Convert tensor to numpy image"""
    if isinstance(tensor, torch.Tensor):
        # Convert to numpy and move channel to last dimension
        img = tensor.detach().cpu().numpy()
        if img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        return img
    else:
        return tensor

def normalize_image(img):
    """Normalize image to [0, 1] range"""
    img = img.copy()
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0, 1)

def create_comparison_grid(img1, img2, img3, titles=None, suptitle=None):
    """Create a comparison grid of three images"""
    # Normalize images
    img1 = normalize_image(tensor_to_numpy(img1))
    img2 = normalize_image(tensor_to_numpy(img2))
    img3 = normalize_image(tensor_to_numpy(img3))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    axes[2].imshow(img3)
    
    # Set titles
    if titles:
        for ax, title in zip(axes, titles):
            ax.set_title(title)
    
    # Set suptitle
    if suptitle:
        fig.suptitle(suptitle)
    
    # Remove axis ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def visualize_results(hazy_img, dehazed_img, clear_img, output_path=None):
    """Visualize dehazing results"""
    # Convert to numpy
    hazy_img = tensor_to_numpy(hazy_img)
    dehazed_img = tensor_to_numpy(dehazed_img)
    clear_img = tensor_to_numpy(clear_img)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    axes[0].imshow(normalize_image(hazy_img))
    axes[0].set_title('Hazy Image')
    
    axes[1].imshow(normalize_image(dehazed_img))
    axes[1].set_title('Dehazed Image')
    
    axes[2].imshow(normalize_image(clear_img))
    axes[2].set_title('Ground Truth')
    
    # Remove axis ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def visualize_detection_results(image, detections, threshold=0.5, output_path=None, title=None):
    """Visualize object detection results"""
    # Convert to numpy
    img = tensor_to_numpy(image)
    img = normalize_image(img)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # Display image
    ax.imshow(img)
    
    # Define colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, 91))  # COCO has 91 classes
    
    # Add detections
    if 'boxes' in detections:
        for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
            if score > threshold:
                # Get coordinates
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                
                # Create rectangle
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1, 
                    linewidth=2, 
                    edgecolor=colors[label.item()], 
                    facecolor='none'
                )
                
                # Add rectangle to plot
                ax.add_patch(rect)
                
                # Add label and score
                ax.text(
                    x1, y1-5, 
                    f'Class {label.item()}: {score.item():.2f}', 
                    color='white', 
                    fontsize=8,
                    bbox=dict(facecolor=colors[label.item()], alpha=0.5)
                )
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def visualize_routing_weights(weights, intensity, output_path=None):
    """Visualize routing weights"""
    # Convert to numpy
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors for different intensity levels
    colors = ['#3498db', '#f39c12', '#e74c3c']  # Blue, Orange, Red
    
    # Create bar chart
    bars = ax.bar(
        ['Low Intensity', 'Medium Intensity', 'High Intensity'],
        weights,
        color=colors
    )
    
    # Highlight the actual intensity
    bars[intensity].set_edgecolor('black')
    bars[intensity].set_linewidth(2)
    
    # Add labels
    for i, v in enumerate(weights):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # Set title and labels
    ax.set_title('Routing Weights')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Weight')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def create_result_montage(images, titles, output_path=None, grid_size=None):
    """Create a montage of result images"""
    n_images = len(images)
    
    # Determine grid size
    if grid_size is None:
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
    else:
        rows, cols = grid_size
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Make sure axes is a 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Fill in the images
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n_images:
                # Display image
                axes[i, j].imshow(normalize_image(tensor_to_numpy(images[idx])))
                
                # Set title
                if titles and idx < len(titles):
                    axes[i, j].set_title(titles[idx])
            
            # Remove axis ticks
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def visualize_joint_training_progress(train_losses, val_losses, val_psnrs, output_path=None):
    """Visualize joint training progress"""
    epochs = list(range(1, len(train_losses) + 1))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot PSNR
    ax2.plot(epochs, val_psnrs, 'g-')
    ax2.set_title('Validation PSNR')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def create_result_summary_chart(baseline_results, joint_results, detection_results, output_path=None):
    """Create summary chart comparing all results"""
    # Extract data
    intensities = ['Low', 'Medium', 'High']
    
    # PSNR comparison
    baseline_psnrs = [
        baseline_results['low_intensity']['psnr'],
        baseline_results['medium_intensity']['psnr'],
        baseline_results['high_intensity']['psnr']
    ]
    
    joint_psnrs = [
        joint_results['low_intensity']['psnr'],
        joint_results['medium_intensity']['psnr'],
        joint_results['high_intensity']['psnr']
    ]
    
    # Detection mAP comparison
    hazy_maps = [
        detection_results['hazy']['low']['mAP'],
        detection_results['hazy']['medium']['mAP'],
        detection_results['hazy']['high']['mAP']
    ]
    
    dehazed_maps = [
        detection_results['dehazed']['low']['mAP'],
        detection_results['dehazed']['medium']['mAP'],
        detection_results['dehazed']['high']['mAP']
    ]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Width of bars
    width = 0.35
    
    # Set positions
    x = np.arange(len(intensities))
    
    # Plot PSNR comparison
    ax1.bar(x - width/2, baseline_psnrs, width, label='Separate Models')
    ax1.bar(x + width/2, joint_psnrs, width, label='Adaptive Framework')
    
    ax1.set_title('PSNR Comparison')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(intensities)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add PSNR values on bars
    for i, v in enumerate(baseline_psnrs):
        ax1.text(i - width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    for i, v in enumerate(joint_psnrs):
        ax1.text(i + width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    # Plot mAP comparison
    ax2.bar(x - width/2, hazy_maps, width, label='Hazy Images')
    ax2.bar(x + width/2, dehazed_maps, width, label='Dehazed Images')
    
    ax2.set_title('Object Detection mAP Comparison')
    ax2.set_ylabel('mAP')
    ax2.set_xticks(x)
    ax2.set_xticklabels(intensities)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add mAP values on bars
    for i, v in enumerate(hazy_maps):
        ax2.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    for i, v in enumerate(dehazed_maps):
        ax2.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()