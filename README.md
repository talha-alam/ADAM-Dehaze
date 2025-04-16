# Adaptive Fog Intensity Dehazing Framework

This repository contains the implementation of an adaptive fog intensity-aware dehazing framework. The system classifies input hazy images based on fog intensity and routes them to specialized dehazing networks, improving both visual quality and object detection performance.

## Key Features

- **Fog Intensity Classification**: Automatically categorizes hazy images into low, medium, or high fog intensity levels.
- **Specialized Dehazing Networks**: Dedicated dehazing architectures optimized for different fog intensities.
- **Adaptive Routing Mechanism**: Intelligently directs images to the appropriate dehazing model.
- **End-to-End Training**: Joint optimization of classification and dehazing objectives.
- **Object Detection Integration**: Evaluates performance improvements on downstream detection tasks.

## Architecture Overview

![Architecture Diagram](docs/architecture.png)

The framework consists of the following components:
1. **Fog Intensity Classifier**: A CNN-based classifier that categorizes input images.
2. **Specialized Dehazing Networks**:
   - Low Intensity: Lightweight model for preserving details in mildly hazy images.
   - Medium Intensity: Standard model with balanced complexity.
   - High Intensity: Complex model with attention mechanisms for dense fog.
3. **Routing Mechanism**: Directs inputs to appropriate dehazing networks based on classification results.
4. **Object Detection**: Evaluates the impact of dehazing on detection performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-dehazing.git
cd adaptive-dehazing

# Create and activate a conda environment
conda create -n dehazing python=3.8
conda activate dehazing

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Dataset Preparation

The framework is designed to work with a dataset containing hazy images, dehazed counterparts (from the CORUN method), and ground truth clear images.

### Expected Dataset Structure

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
└── processed/  # Will be created by preprocessing script
    ├── train/
    ├── val/
    └── test/
```

### Preprocessing

```bash
# Preprocess the dataset (resize, normalize, and split into train/val/test)
python main.py --mode preprocess --data_dir path/to/dataset
```

## Configuration

The `config/config.yaml` file contains all configuration parameters:

- Dataset paths and parameters
- Model architectures and hyperparameters
- Training settings
- Evaluation metrics

Modify this file to customize the framework to your needs.

## Training

### Step-by-Step Training

```bash
# 1. Train the fog intensity classifier
python main.py --mode train_classifier

# 2. Train individual dehazing models
python main.py --mode train_dehazing

# 3. Train the joint model
python main.py --mode train_joint
```

### Complete Training Pipeline

```bash
# Run the complete training pipeline
python main.py --mode train_all
```

## Evaluation

```bash
# Run comprehensive evaluation
python main.py --mode evaluate
```

This will:
1. Evaluate baseline models on their respective fog intensities
2. Evaluate the joint adaptive model
3. Assess object detection performance on hazy vs. dehazed images
4. Generate visualizations and comparison charts

## Demo

```bash
# Run a demo with pretrained models
python main.py --mode demo
```

## Results

### Image Quality Metrics

| Model | Low Intensity |  | Medium Intensity |  | High Intensity |  |
|-------|---------------|-----------------|-----------------|-----------------|----------------|-----------------|
|       | PSNR | SSIM | PSNR | SSIM | PSNR | SSIM |
| Individual Models | xx.xx | 0.xxx | xx.xx | 0.xxx | xx.xx | 0.xxx |
| Adaptive Framework | xx.xx | 0.xxx | xx.xx | 0.xxx | xx.xx | 0.xxx |

### Object Detection Performance (mAP)

| Input Images | Low Intensity | Medium Intensity | High Intensity | Average |
|--------------|---------------|------------------|----------------|---------|
| Hazy | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| Dehazed | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| Improvement | +xx.x% | +xx.x% | +xx.x% | +xx.x% |

## Code Structure

```
adaptive_dehazing/
├── config/              # Configuration files
├── data/                # Data loading and preprocessing
├── models/              # Neural network architectures
│   ├── classifier.py    # Fog intensity classifier
│   ├── dehazing/        # Dehazing models
│   ├── routing.py       # Routing mechanism
│   └── detection.py     # Object detection integration
├── training/            # Training procedures
├── evaluation/          # Evaluation metrics and procedures
├── utils/               # Utility functions
├── main.py              # Main script
└── requirements.txt     # Dependencies
```

## Citation

If you find this work useful for your research, please cite our paper:

```
@article{YourName2025,
  title={Adaptive Fog Intensity Dehazing Framework for Improved Object Detection},
  author={Your Name and Co-authors},
  journal={ArXiv},
  year={2025}
}
```

## Acknowledgements

This work builds upon the CORUN dehazing method and other prior works in image dehazing and object detection. We acknowledge the following papers:

- [CORUN: COntextualized RoUting Network for Image Dehazing](https://arxiv.org/pdf/2406.07966)
- [Unified Density-Aware Image Dehazing and Object Detection in Real-World Hazy Scenes](https://openaccess.thecvf.com/content/ACCV2020/papers/Zhang_Unified_Density-Aware_Image_Dehazing_and_Object_Detection_in_Real-World_Hazy_ACCV_2020_paper.pdf)

## License

This project is licensed under the MIT License - see the LICENSE file for details.