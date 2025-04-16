import os
import cv2
import numpy as np
from tqdm import tqdm

def resize_and_normalize(image, size=256):
    """Resize image and normalize to [0, 1]"""
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size))
    return image.astype(np.float32) / 255.0

def preprocess_dataset(source_dir, dest_dir, size=256):
    """Preprocess all images in the dataset for consistency"""
    os.makedirs(dest_dir, exist_ok=True)
    
    # Process each fog intensity separately
    for intensity in ['low', 'medium', 'high']:
        print(f"Processing {intensity} intensity images...")
        
        # Create destination directories
        hazy_dest = os.path.join(dest_dir, intensity, 'hazy')
        clear_dest = os.path.join(dest_dir, intensity, 'clear')
        dehazed_dest = os.path.join(dest_dir, intensity, 'dehazed')
        
        os.makedirs(hazy_dest, exist_ok=True)
        os.makedirs(clear_dest, exist_ok=True)
        os.makedirs(dehazed_dest, exist_ok=True)
        
        # Source directories
        hazy_src = os.path.join(source_dir, intensity, 'hazy')
        clear_src = os.path.join(source_dir, intensity, 'clear')
        dehazed_src = os.path.join(source_dir, intensity, 'dehazed')
        
        # Process hazy images
        for img_name in tqdm(os.listdir(hazy_src)):
            if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
                continue
                
            # Load and process hazy image
            hazy_path = os.path.join(hazy_src, img_name)
            hazy_img = cv2.imread(hazy_path)
            if hazy_img is None:
                print(f"Warning: Could not read {hazy_path}")
                continue
            hazy_img = resize_and_normalize(hazy_img, size)
            
            # Load corresponding clear image
            clear_path = os.path.join(clear_src, img_name)
            if os.path.exists(clear_path):
                clear_img = cv2.imread(clear_path)
                clear_img = resize_and_normalize(clear_img, size)
                cv2.imwrite(os.path.join(clear_dest, img_name), (clear_img * 255).astype(np.uint8))
            
            # Load corresponding dehazed image
            dehazed_path = os.path.join(dehazed_src, img_name)
            if os.path.exists(dehazed_path):
                dehazed_img = cv2.imread(dehazed_path)
                dehazed_img = resize_and_normalize(dehazed_img, size)
                cv2.imwrite(os.path.join(dehazed_dest, img_name), (dehazed_img * 255).astype(np.uint8))
            
            # Save preprocessed hazy image
            cv2.imwrite(os.path.join(hazy_dest, img_name), (hazy_img * 255).astype(np.uint8))
            
    print("Preprocessing completed!")

def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    np.random.seed(seed)
    
    train_dir = os.path.join(source_dir, 'train')
    val_dir = os.path.join(source_dir, 'val')
    test_dir = os.path.join(source_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each fog intensity separately
    for intensity in ['low', 'medium', 'high']:
        print(f"Splitting {intensity} intensity data...")
        
        # Create intensity directories in each split
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, intensity, 'hazy'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, intensity, 'clear'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, intensity, 'dehazed'), exist_ok=True)
        
        # Get all image names
        hazy_dir = os.path.join(source_dir, intensity, 'hazy')
        img_names = [f for f in os.listdir(hazy_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Shuffle and split
        np.random.shuffle(img_names)
        n_train = int(len(img_names) * train_ratio)
        n_val = int(len(img_names) * val_ratio)
        
        train_names = img_names[:n_train]
        val_names = img_names[n_train:n_train+n_val]
        test_names = img_names[n_train+n_val:]
        
        # Function to copy files
        def copy_files(file_names, source_subdir, dest_subdir):
            for name in file_names:
                # Copy hazy image
                src = os.path.join(source_dir, intensity, 'hazy', name)
                dst = os.path.join(dest_subdir, intensity, 'hazy', name)
                if os.path.exists(src):
                    cv2.imwrite(dst, cv2.imread(src))
                
                # Copy clear image
                src = os.path.join(source_dir, intensity, 'clear', name)
                dst = os.path.join(dest_subdir, intensity, 'clear', name)
                if os.path.exists(src):
                    cv2.imwrite(dst, cv2.imread(src))
                
                # Copy dehazed image
                src = os.path.join(source_dir, intensity, 'dehazed', name)
                dst = os.path.join(dest_subdir, intensity, 'dehazed', name)
                if os.path.exists(src):
                    cv2.imwrite(dst, cv2.imread(src))
        
        # Copy files to respective splits
        copy_files(train_names, source_dir, train_dir)
        copy_files(val_names, source_dir, val_dir)
        copy_files(test_names, source_dir, test_dir)
        
    print("Dataset splitting completed!")

if __name__ == "__main__":
    # Example usage
    source_dir = "path/to/raw/dataset"
    processed_dir = "path/to/processed/dataset"
    
    # Step 1: Preprocess all images
    preprocess_dataset(source_dir, processed_dir)
    
    # Step 2: Split into train/val/test
    split_dataset(processed_dir)