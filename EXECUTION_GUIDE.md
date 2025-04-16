# Execution Guide

This guide provides detailed instructions for executing the Adaptive Fog Intensity Dehazing Framework. Follow these steps to set up your environment, prepare your data, train the models, and evaluate the results.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Configuration](#3-configuration)
4. [Training Pipeline](#4-training-pipeline)
5. [Evaluation](#5-evaluation)
6. [Troubleshooting](#6-troubleshooting)
7. [Performance Optimization](#7-performance-optimization)

## 1. Environment Setup

### System Requirements

- CUDA-capable GPU with at least 8GB memory (recommended)
- CUDA 11.3 or later
- Python 3.8 or later
- 16GB RAM (recommended)
- 100GB disk space for datasets and checkpoints

### Installation Steps

```bash
# Create a new conda environment
conda create -n dehazing python=3.8
conda activate dehazing

# Install PyTorch with CUDA support
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Clone the repository
git clone https://github.com/yourusername/adaptive-dehazing.git
cd adaptive-dehazing

# Install requirements
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

## 2. Dataset Preparation

### Required Data

The framework requires three types of images for each fog intensity level:
- **Hazy images**: Original images with fog
- **Clear images**: Ground truth fog-free images
- **Dehazed images**: Images processed by CORUN method

### Organizing Your Dataset

1. Create the following directory structure:

```
dataset/
├── raw/
│   ├── low/
│   │   ├── hazy/
│   │   ├── clear/
│   │   └── dehazed/
│   ├── medium/
│   │   ├── hazy/
│   │   ├── clear/
│   │   └── dehazed/
│   └── high/
│       ├── hazy/
│       ├── clear/
│       └── dehazed/
```

2. Place your images in the corresponding directories.
3. Ensure image filenames match across directories (e.g., `image_001.jpg` in hazy, clear, and dehazed folders).

### Preprocessing

Run the preprocessing script to prepare the dataset:

```bash
# Preprocess and split the dataset
python main.py --mode preprocess --data_dir path/to/dataset --config config/config.yaml
```

This will:
- Resize images to the dimensions specified in the config
- Normalize pixel values to [0, 1]
- Split the dataset into train/val/test sets
- Create the processed dataset structure

## 3. Configuration

The `config/config.yaml` file controls all aspects of the framework. Key sections include:

### Dataset Configuration

```yaml
dataset:
  train_path: 'path/to/train'
  val_path: 'path/to/val'
  test_path: 'path/to/test'
  img_size: 256
  batch_size: 16
  num_workers: 4
  augmentation: True
```

### Model Configurations

```yaml
# Fog Intensity Classifier
classifier:
  model: 'resnet18'
  pretrained: True
  num_classes: 3
  learning_rate: 0.0001
  epochs: 20

# Dehazing Models
dehazing:
  low:
    model_type: 'lightweight'
    channels: 32
    blocks: 3
  
  medium:
    model_type: 'standard'
    channels: 64
    blocks: 6
  
  high:
    model_type: 'complex'
    channels: 96
    blocks: 9
    attention: True

# Routing Mechanism
routing:
  type: 'soft'  # Options: 'hard', 'soft', 'gated'
  temperature: 0.5
```

### Training Parameters

```yaml
# Joint Training
joint_training:
  learning_rate: 0.00005
  epochs: 50
  lambda_dehazing: 1.0
  lambda_classification: 0.2
  lambda_perceptual: 0.1
  lambda_detection: 0.5
```

### Important Settings to Adjust

- **GPU device**: Set `device: 'cuda'` or `device: 'cpu'`
- **Batch size**: Adjust based on your GPU memory
- **Image size**: Higher values improve quality but increase memory usage
- **Learning rates**: Fine-tune based on training stability
- **Routing type**: Choose between hard routing (one model per image) or soft routing (weighted combination)

## 4. Training Pipeline

The training pipeline consists of three main stages:

### Stage 1: Train Fog Intensity Classifier

```bash
# Train the classifier
python main.py --mode train_classifier --config config/config.yaml
```

This will:
- Train a classifier to identify fog intensity levels
- Save checkpoints to `checkpoints/classifier/`
- Generate classification metrics and confusion matrices

**Expected duration**: 2-4 hours on a modern GPU.

### Stage 2: Train Specialized Dehazing Models

```bash
# Train all three dehazing models
python main.py --mode train_dehazing --config config/config.yaml
```

This will:
- Train separate models for low, medium, and high fog intensities
- Save checkpoints to `checkpoints/dehazing/{low,medium,high}/`
- Generate dehazing quality metrics for each model

**Expected duration**: 10-15 hours total on a modern GPU.

### Stage 3: Train Joint Model

```bash
# Train the joint model with routing
python main.py --mode train_joint --config config/config.yaml
```

This will:
- Initialize models with previously trained weights
- Train the routing mechanism
- Fine-tune all components jointly
- Save checkpoints to `checkpoints/joint/`

**Expected duration**: 5-8 hours on a modern GPU.

### Complete Pipeline

To run all stages sequentially:

```bash
# Run the complete training pipeline
python main.py --mode train_all --config config/config.yaml
```

**Expected duration**: 18-25 hours total on a modern GPU.

## 5. Evaluation

The evaluation process assesses the performance of the framework on image quality metrics and object detection tasks.

### Comprehensive Evaluation

```bash
# Run comprehensive evaluation
python main.py --mode evaluate --config config/config.yaml
```

This will:
1. Evaluate individual dehazing models on their respective fog intensities
2. Evaluate the joint adaptive framework on all fog intensities
3. Compare object detection performance on hazy vs. dehazed images
4. Generate visualizations in `results/visualizations/`
5. Save metrics to `results/metrics/`

### Visualization Examples

The evaluation generates visualizations including:
- Side-by-side comparisons of hazy, dehazed, and ground truth images
- Routing weight distributions
- Object detection results on hazy vs. dehazed images
- Performance comparison charts

### Custom Evaluation

For custom evaluation on specific images:

```bash
# Run a demo on custom images
python main.py --mode demo --config config/config.yaml
```

## 6. Troubleshooting

### Common Issues and Solutions

#### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size in config.yaml:
```yaml
dataset:
  batch_size: 8  # Try a smaller value
```

#### Slow Training

**Solutions**:
- Increase `num_workers` in config.yaml for faster data loading
- Reduce image size for faster training
- Use mixed precision training (add `use_amp: True` to config)

#### Poor Classification Accuracy

**Solutions**:
- Increase classifier training epochs
- Use a more powerful backbone (e.g., 'resnet34' instead of 'resnet18')
- Increase learning rate for classifier

#### Poor Dehazing Quality

**Solutions**:
- Increase model capacity (more channels and blocks)
- Use stronger augmentations during training
- Increase the weight of perceptual loss

## 7. Performance Optimization

### Memory Optimization

- Use mixed precision training to reduce memory usage
- Implement gradient checkpointing for larger models
- Use smaller batch sizes with gradient accumulation

### Speed Optimization

- Increase number of data loading workers
- Use a more efficient backbone for the classifier
- Implement TensorRT for inference optimization

### Quality Optimization

- Experiment with different loss functions (MS-SSIM, LPIPS)
- Use more sophisticated routing mechanisms
- Implement ensemble methods for better dehazing quality

---

## Research Extensions

Here are some suggested extensions to enhance the framework:

1. **Implement uncertainty-aware routing**: Modify the routing mechanism to account for uncertainty in fog intensity classification.

2. **Explore progressive dehazing**: Implement a cascaded approach where images are progressively dehazed through multiple stages.

3. **Integrate with different detection architectures**: Evaluate performance with YOLO, SSD, or other detection models.

4. **Add temporal consistency for videos**: Extend the framework to handle video dehazing with temporal consistency.

For support or questions, please open an issue on the GitHub repository.