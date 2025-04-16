import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class HazyImageDataset(Dataset):
    """Dataset for dehazing with fog intensity classification"""
    
    def __init__(self, root_dir, split='train', transform=None, img_size=256):
        """
        Args:
            root_dir (str): Root directory of the dataset
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
            img_size (int): Size to resize images to
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        
        # Parse intensity labels
        self.intensity_map = {'low': 0, 'medium': 1, 'high': 2}
        
        # Build the dataset
        for intensity in ['low', 'medium', 'high']:
            hazy_dir = os.path.join(self.root_dir, intensity, 'hazy')
            clear_dir = os.path.join(self.root_dir, intensity, 'clear')
            dehazed_dir = os.path.join(self.root_dir, intensity, 'dehazed')
            
            intensity_label = self.intensity_map[intensity]
            
            for img_name in os.listdir(hazy_dir):
                if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
                    continue
                
                hazy_path = os.path.join(hazy_dir, img_name)
                clear_path = os.path.join(clear_dir, img_name)
                dehazed_path = os.path.join(dehazed_dir, img_name)
                
                # Only add if all three images exist
                if os.path.exists(hazy_path) and os.path.exists(clear_path) and os.path.exists(dehazed_path):
                    self.samples.append({
                        'hazy': hazy_path,
                        'clear': clear_path,
                        'dehazed': dehazed_path,
                        'intensity': intensity_label,
                        'name': img_name
                    })
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Default transformations if none provided
        if self.transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    # transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1)
                ])
            else:
                self.transform = transforms.Compose([
                    # transforms.ToTensor()
                ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        hazy_img = cv2.imread(sample['hazy'])
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        
        clear_img = cv2.imread(sample['clear'])
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        
        dehazed_img = cv2.imread(sample['dehazed'])
        dehazed_img = cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if hazy_img.shape[0] != self.img_size or hazy_img.shape[1] != self.img_size:
            hazy_img = cv2.resize(hazy_img, (self.img_size, self.img_size))
        
        if clear_img.shape[0] != self.img_size or clear_img.shape[1] != self.img_size:
            clear_img = cv2.resize(clear_img, (self.img_size, self.img_size))
            
        if dehazed_img.shape[0] != self.img_size or dehazed_img.shape[1] != self.img_size:
            dehazed_img = cv2.resize(dehazed_img, (self.img_size, self.img_size))
        
        # Convert to tensor and normalize
        hazy_tensor = transforms.ToTensor()(hazy_img)
        clear_tensor = transforms.ToTensor()(clear_img)
        dehazed_tensor = transforms.ToTensor()(dehazed_img)
        
        # Apply transformations if needed
        if self.transform:
            # Create a seed to ensure the same transformation is applied to all images
            seed = np.random.randint(2147483647)
            
            random.seed(seed)
            torch.manual_seed(seed)
            hazy_tensor = self.transform(hazy_tensor)
            
            random.seed(seed)
            torch.manual_seed(seed)
            clear_tensor = self.transform(clear_tensor)
            
            random.seed(seed)
            torch.manual_seed(seed)
            dehazed_tensor = self.transform(dehazed_tensor)
        
        return {
            'hazy': hazy_tensor,
            'clear': clear_tensor,
            'dehazed': dehazed_tensor,
            'intensity': torch.tensor(sample['intensity'], dtype=torch.long),
            'name': sample['name']
        }

class DetectionDataset(Dataset):
    """Dataset for object detection evaluation with hazy/dehazed images"""
    
    # Modify this in data/dataset.py - update the DetectionDataset constructor

    def __init__(self, root_dir, annotation_dir, split='test', img_size=512):
        """
        Args:
            root_dir (str): Root directory of the dataset
            annotation_dir (str): Directory with object detection annotations
            split (str): 'train', 'val', or 'test'
            img_size (int): Size to resize images to
        """
        self.root_dir = os.path.join(root_dir, split)
        self.annotation_dir = annotation_dir
        self.img_size = img_size
        self.samples = []
        
        # Build the dataset - modified to handle the actual directory structure
        print(f"Looking for hazy images in: {self.root_dir}")
        
        # Check all three intensity levels
        for intensity in ['low', 'medium', 'high']:
            hazy_dir = os.path.join(self.root_dir, intensity, 'hazy')
            
            if not os.path.exists(hazy_dir):
                print(f"Warning: Directory not found: {hazy_dir}")
                continue
                
            print(f"Processing intensity level: {intensity}, directory: {hazy_dir}")
            
            for img_name in os.listdir(hazy_dir):
                if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
                    continue
                
                base_name = os.path.splitext(img_name)[0]
                annotation_path = os.path.join(self.annotation_dir, f"{base_name}.json")
                
                # If annotation doesn't exist, try looking for it with just the base name
                if not os.path.exists(annotation_path):
                    annotation_path = os.path.join(self.annotation_dir, "instances.json")
                
                if os.path.exists(annotation_path):
                    self.samples.append({
                        'hazy': os.path.join(hazy_dir, img_name),
                        'annotation': annotation_path,
                        'name': img_name,
                        'intensity': intensity
                    })
                else:
                    print(f"Warning: No annotation found for {img_name}")
        
        print(f"Loaded {len(self.samples)} samples for detection evaluation")
        
        # Transform for detection
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load hazy image
        hazy_img = cv2.imread(sample['hazy'])
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        
        # Load annotations (assuming COCO format)
        import json
        with open(sample['annotation'], 'r') as f:
            annotation = json.load(f)
        
        # Create a target dictionary for object detection
        boxes = []
        labels = []
        
        for obj in annotation['annotations']:
            x, y, w, h = obj['bbox']
            # Convert to [x1, y1, x2, y2] format
            boxes.append([x, y, x+w, y+h])
            labels.append(obj['category_id'])
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        
        # Create the target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Transform the image
        image = self.transform(hazy_img)
        
        return image, target, sample['name'], sample['intensity']

def get_dataloader(config, split='train'):
    """Create a dataloader for the specified split"""
    dataset = HazyImageDataset(
        root_dir=config['dataset']['train_path'] if split == 'train' else 
                 config['dataset']['val_path'] if split == 'val' else
                 config['dataset']['test_path'],
        split=split,
        img_size=config['dataset']['img_size']
    )
    
    return DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )

def get_detection_dataloader(config, split='test'):
    """Create a dataloader for object detection evaluation"""
    dataset = DetectionDataset(
        root_dir=config['dataset']['test_path'],
        annotation_dir=os.path.join(config['dataset']['test_path'], 'annotations'),
        split=split,
        img_size=512  # Common size for object detection
    )
    
    return DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'] // 2,  # Smaller batch size for detection
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        collate_fn=detection_collate_fn  # Custom collate function for variable sized objects
    )

def detection_collate_fn(batch):
    """Custom collate function for detection dataset"""
    images = []
    targets = []
    names = []
    intensities = []
    
    for image, target, name, intensity in batch:
        images.append(image)
        targets.append(target)
        names.append(name)
        intensities.append(intensity)
    
    return torch.stack(images, 0), targets, names, intensities