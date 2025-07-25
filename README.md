# ADAM-Dehaze: Adaptive Density-Aware Multi-Stage Dehazing for Improved Object Detection in Foggy Conditions

[![Paper](https://img.shields.io/badge/Paper-SMC%202025-red)](https://arxiv.org/abs/2506.15837)
[![Code](https://img.shields.io/badge/Code-Available-green)](https://github.com/your-repo/adam-dehaze)
[![Dataset](https://img.shields.io/badge/Dataset-FogIntensity--25K-blue)](https://github.com/your-repo/adam-dehaze)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Accepted at IEEE Systems, Man, and Cybernetics Conference (SMC) 2025**

## Authors

**Fatmah AlHindaassi**¹*, **Mohammed Talha Alam**¹*, **Fakhri Karray**¹

*Equal Contribution  
¹ Mohamed Bin Zayed University of Artificial Intelligence, UAE

## Abstract

Adverse weather conditions, particularly fog, pose a significant challenge to autonomous vehicles, surveillance systems, and other safety-critical applications by severely degrading visual information. We introduce **ADAM-Dehaze**, an adaptive, density-aware dehazing framework that jointly optimizes image restoration and object detection under varying fog intensities. First, a lightweight Haze Density Estimation Network (HDEN) classifies each input as light, medium, or heavy fog. Based on this score, the system dynamically routes the image through one of three CORUN branches—Light, Medium, or Complex—each tailored to its haze regime. A novel adaptive loss then balances physical-model coherence and perceptual fidelity, ensuring both accurate defogging and preservation of fine details. On Cityscapes and the real-world RTTS benchmark, ADAM-Dehaze boosts PSNR by up to 2.1 dB, reduces FADE by 30%, and improves object detection mAP by up to 13 points, all while cutting inference time by 20%.

## Key Features

- 🌫️ **Adaptive Fog Classification**: Lightweight HDEN network categorizes images into light/medium/heavy fog (99.80% accuracy)
- 🔄 **Multi-Branch Architecture**: Three specialized CORUN variants (2/4/6 stages) optimized for different fog intensities
- ⚡ **Dynamic Routing**: Intelligent branch selection reduces inference time by 20% while maintaining quality
- 🎯 **Joint Optimization**: Seamless integration with object detection for end-to-end performance
- 📊 **Novel Dataset**: FogIntensity-25K with 25,000 paired synthetic images across three fog levels
- 🔧 **Adaptive Loss Function**: Density-modulated loss balancing physical coherence and perceptual fidelity

## Architecture Overview

![Architecture Diagram](architecture.png)

The framework consists of:
1. **Haze Density Estimation Network (HDEN)**: DenseNet121-based classifier achieving 99.80% accuracy
2. **Specialized Dehazing Networks**:
   - **Light (β=0.03)**: 2-stage CORUN for efficient processing (18ms, 45G FLOPs)
   - **Medium (β=0.06)**: 4-stage CORUN balancing quality and speed (38ms, 80G FLOPs)  
   - **Complex (β=0.09)**: 6-stage CORUN with attention for dense fog (50ms, 150G FLOPs)
3. **Adaptive Loss**: Density-weighted combination of coherence, perceptual, and density terms
4. **Object Detection Integration**: YOLOv8n fine-tuned on dehazed outputs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/adam-dehaze.git
cd adam-dehaze

# Create and activate conda environment
conda create -n adam-dehaze python=3.8
conda activate adam-dehaze

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Dataset Preparation

#### FogIntensity-25K Dataset Structure
```
dataset/
├── raw/
│   ├── light/      # β=0.03, 8,333 images
│   ├── medium/     # β=0.06, 8,333 images  
│   └── heavy/      # β=0.09, 8,334 images
└── processed/
    ├── train/
    ├── val/
    └── test/
```

#### Preprocessing
```bash
# Preprocess the FogIntensity-25K dataset
python main.py --mode preprocess --data_dir path/to/dataset
```

### Training

#### Step-by-Step Training
```bash
# 1. Train the HDEN fog intensity classifier
python main.py --mode train_classifier

# 2. Train individual CORUN dehazing branches
python main.py --mode train_dehazing

# 3. Train the joint adaptive model
python main.py --mode train_joint
```

#### Complete Training Pipeline
```bash
# Run the complete training pipeline
python main.py --mode train_all
```

### Evaluation

```bash
# Run comprehensive evaluation on Cityscapes and RTTS
python main.py --mode evaluate
```

### Demo

```bash
# Run demo with pretrained models
python main.py --mode demo --input_path path/to/foggy/images
```

## 📊 Experimental Results

### Image Quality Metrics (Cityscapes)

| Fog Level | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|-----------|---------|---------|----------|
| Light (β=0.03) | **0.9188** | **23.95** | **0.0585** |
| Medium (β=0.06) | **0.8761** | **21.78** | **0.0929** |
| Complex (β=0.09) | **0.8060** | **19.39** | **0.1456** |

### Real-World Performance (RTTS Dataset)

| Method | FADE ↓ | BRISQUE ↓ | NIMA ↑ | mAP (%) |
|--------|---------|-----------|---------|---------|
| MBDN | 1.363 | 27.672 | 4.529 | 52.0 |
| RIDCP | 0.944 | 17.293 | 4.965 | 55.0 |
| PSD | 0.920 | 27.713 | 4.598 | 51.0 |
| **ADAM-Dehaze** | **0.828** | **11.961** | **5.346** | **56.0** |

### Object Detection Performance Improvement

| Object Class | mAP Improvement |
|--------------|-----------------|
| Bicycle | **59.0%** (+7%) |
| Car | **67.0%** (+4%) |
| Motor | **48.0%** (+6%) |
| Person | **77.0%** (+3%) |
| **Average** | **56.0%** (+5%) |

### Computational Efficiency

| Branch | Inference Time | FLOPs | Use Case |
|--------|----------------|-------|----------|
| CORUN-Light | 18ms | 45G | Mild fog (β≤0.03) |
| CORUN-Medium | 38ms | 80G | Moderate fog (0.03<β≤0.06) |
| CORUN-Complex | 50ms | 150G | Dense fog (β>0.06) |
| **Adaptive Average** | **30ms** | **92G** | **20% faster than fixed** |

## 🔧 Configuration

The framework is highly configurable through `config/config.yaml`:

```yaml
# Fog intensity thresholds
fog_thresholds:
  light_threshold: 0.03    # α parameter
  medium_threshold: 0.06   # β parameter

# Adaptive loss weights
loss_weights:
  light_coherence: 0.3     # γ(d) for light fog
  medium_coherence: 0.6    # γ(d) for medium fog  
  heavy_coherence: 0.9     # γ(d) for heavy fog

# Training parameters
training:
  batch_size: 16
  learning_rate: 2e-4
  epochs: 100
  weight_decay: 1e-4
```

## 📈 Ablation Studies

### Loss Function Components
| Loss Configuration | PSNR (dB) | SSIM | mAP (%) |
|-------------------|-----------|------|---------|
| **Full Model** | **23.95** | **0.9188** | **75.0** |
| w/o Perceptual Loss | 22.10 | 0.9025 | 72.5 |
| w/o Density Loss | 23.12 | 0.9113 | 73.8 |
| w/o Coherence Loss | 21.45 | 0.8762 | 68.2 |

### Cooperative Proximal Mapping Module (CPMM)
| Model Variant | PSNR (dB) | SSIM | mAP (%) |
|---------------|-----------|------|---------|
| **Full Model (w/ CPMM)** | **23.95** | **0.9188** | **75.0** |
| w/o CPMM | 22.50 | 0.8992 | 71.4 |

## 🏗️ Code Structure

```
adam_dehaze/
├── config/              # Configuration files
│   └── config.yaml      # Main configuration
├── data/                # Data loading and preprocessing
│   ├── dataset.py       # FogIntensity-25K dataset loader
│   └── transforms.py    # Data augmentation
├── models/              # Neural network architectures
│   ├── hden.py          # Haze Density Estimation Network
│   ├── corun.py         # CORUN dehazing variants
│   ├── routing.py       # Adaptive routing mechanism
│   └── detection.py     # YOLOv8 integration
├── training/            # Training procedures
│   ├── trainer.py       # Main training loop
│   └── losses.py        # Adaptive loss functions
├── evaluation/          # Evaluation metrics and procedures
│   ├── metrics.py       # PSNR, SSIM, FADE, etc.
│   └── detection_eval.py # Object detection evaluation
├── utils/               # Utility functions
│   ├── fog_synthesis.py # Atmospheric scattering model
│   └── visualization.py # Result visualization
├── main.py              # Main entry point
└── requirements.txt     # Dependencies
```

## 🔬 FogIntensity-25K Dataset

We introduce and will release **FogIntensity-25K**, a comprehensive synthetic dataset containing:

- **25,000 paired images** (hazy/clear) based on Cityscapes and Synscapes
- **Three fog intensities**: Light (β=0.03), Medium (β=0.06), Heavy (β=0.09)
- **Depth-aware fog synthesis** using atmospheric scattering model
- **Object detection annotations** for downstream task evaluation

### Dataset Composition
| Fog Level | β Value | Images | Description |
|-----------|---------|--------|-------------|
| Light | 0.03 | 8,333 | Mild degradation, visibility preserved |
| Medium | 0.06 | 8,333 | Moderate degradation |
| Heavy | 0.09 | 8,334 | Severe visibility loss |

## 📖 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{alhindaassi2025adam,
  title={ADAM-Dehaze: Adaptive Density-Aware Multi-Stage Dehazing for Improved Object Detection in Foggy Conditions},
  author={AlHindaassi, Fatmah and Alam, Mohammed Talha and Karray, Fakhri},
  journal={arXiv preprint arXiv:2506.15837},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

- Bug reports and feature requests via GitHub Issues
- Pull requests for improvements and extensions
- Dataset contributions and evaluation benchmarks

## 📧 Contact

For questions about this work, please contact:

- **Mohammed Talha Alam**: mohammed.alam@mbzuai.ac.ae

## 🙏 Acknowledgements

This work builds upon several important contributions:
- [CORUN: Contextualized Routing Network for Image Dehazing](https://arxiv.org/pdf/2406.07966)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [RTTS Real-world Task-driven Testing Set](https://github.com/RRTTS/RTTS)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: The FogIntensity-25K dataset and pretrained models will be made publicly available soon.
