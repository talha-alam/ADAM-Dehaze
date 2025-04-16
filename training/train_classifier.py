import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data.dataset import get_dataloader
from models.classifier import create_classifier

def train_classifier(config):
    """Train the fog intensity classifier"""
    
    # Set device
    device = torch.device(config['device'])
    
    # Initialize model
    model = create_classifier(config)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['classifier']['learning_rate'],
        weight_decay=config['classifier']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Dataloaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # Setup tensorboard
    log_dir = os.path.join('logs', 'classifier')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Create checkpoint directory
    os.makedirs(config['classifier']['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config['classifier']['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Train for one epoch
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['classifier']['epochs']}") as t:
            for batch_idx, batch in enumerate(t):
                # Get the inputs
                hazy_imgs = batch['hazy'].to(device)
                intensity_labels = batch['intensity'].to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                logits, _ = model(hazy_imgs)
                
                # Calculate loss
                loss = criterion(logits, intensity_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += intensity_labels.size(0)
                train_correct += (predicted == intensity_labels).sum().item()
                
                # Update progress bar
                t.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total
                })
        
        # Compute epoch-level training metrics
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get the inputs
                hazy_imgs = batch['hazy'].to(device)
                intensity_labels = batch['intensity'].to(device)
                
                # Forward pass
                logits, _ = model(hazy_imgs)
                
                # Calculate loss
                loss = criterion(logits, intensity_labels)
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                
                # Store predictions and targets for metrics
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(intensity_labels.cpu().numpy())
        
        # Compute validation metrics
        val_loss /= len(val_loader)
        val_acc = 100. * accuracy_score(val_targets, val_preds)
        val_cm = confusion_matrix(val_targets, val_preds)
        val_report = classification_report(val_targets, val_preds, target_names=['low', 'medium', 'high'])
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{config['classifier']['epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Confusion Matrix:\n{val_cm}")
        print(f"Classification Report:\n{val_report}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                config['classifier']['checkpoint_dir'], 'best_model.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                config['classifier']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Close tensorboard writer
    writer.close()
    
    # Load best model for evaluation
    best_checkpoint = torch.load(os.path.join(
        config['classifier']['checkpoint_dir'], 'best_model.pth'
    ))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Return the trained model
    return model

def evaluate_classifier(model, config):
    """Evaluate the classifier on the test set"""
    # Set device
    device = torch.device(config['device'])
    model = model.to(device)
    model.eval()
    
    # Dataloader
    test_loader = get_dataloader(config, split='test')
    
    # Evaluation
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Get the inputs
            hazy_imgs = batch['hazy'].to(device)
            intensity_labels = batch['intensity'].to(device)
            
            # Forward pass
            logits, _ = model(hazy_imgs)
            _, predicted = torch.max(logits.data, 1)
            
            # Store predictions and targets
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(intensity_labels.cpu().numpy())
    
    # Compute metrics
    test_acc = 100. * accuracy_score(test_targets, test_preds)
    test_cm = confusion_matrix(test_targets, test_preds)
    test_report = classification_report(test_targets, test_preds, target_names=['low', 'medium', 'high'])
    
    # Print results
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Confusion Matrix:\n{test_cm}")
    print(f"Classification Report:\n{test_report}")
    
    # Return metrics
    return {
        'accuracy': test_acc,
        'confusion_matrix': test_cm,
        'classification_report': test_report
    }

if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Train classifier
    model = train_classifier(config)
    
    # Evaluate on test set
    evaluate_classifier(model, config)