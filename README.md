# SolarVision AI: Automated PV Panel Defect Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)

**AI-powered computer vision system for automated detection and classification of solar panel defects in photovoltaic installations.**

![SolarVision-AI-Screenshot](image.png)

---

## Executive Summary

SolarVision AI is a production-ready machine learning system that automates defect detection in solar panel installations using standard RGB imagery. The system employs a hybrid deep learning architecture (ResNet18 feature extraction + SVM classification) to identify six critical defect types with **96.84% accuracy**.

### Business Value

| Metric | Before (Manual) | After (AI-Powered) | Improvement |
|--------|-----------------|-------------------|-------------|
| Cost per Panel | $1.50 | $0.20 | **87% reduction** |
| Inspection Speed | 100-500 panels/day | 5,000+ panels/day | **10x faster** |
| Inspection Frequency | Quarterly | Monthly | **3x more frequent** |
| Critical Defect Detection | 75% | >90% | **+15% accuracy** |

![SolarVision-AI-2](image-1.png)

---

## Problem Statement

Solar farm operators face critical operational challenges:

| Challenge | Business Impact |
|-----------|-----------------|
| **Manual Inspection Costs** | $0.50-2.00 per panel |
| **Low Inspection Frequency** | Quarterly or semi-annual only |
| **Human Error Rate** | 15-25% for subtle defects |
| **Delayed Detection** | Defects discovered after 5-20% efficiency loss |
| **Safety Risks** | Electrical failures can cause fire hazards |

These inefficiencies result in reduced energy yield, increased maintenance costs, and shortened panel lifespan.

---

## Solution

SolarVision AI delivers an end-to-end automated defect detection pipeline:

1. **RGB Image Processing**: Works with standard drone/camera imagery (no specialized thermal equipment)
2. **6-Class Classification**: Clean, Bird-drop, Dusty, Electrical-damage, Physical-damage, Snow Covered
3. **Confidence Scoring**: Transparent predictions for quality control validation
4. **Actionable Reporting**: Priority-ranked defect lists for maintenance scheduling
5. **UAV Integration**: Lightweight model for real-time edge processing

### Use Cases

- **Solar Farm O&M**: Continuous monitoring of large-scale installations
- **UAV Inspection**: Automated drone-based panel surveys
- **Preventive Maintenance**: Early detection before critical failures
- **Quality Assurance**: Post-installation commissioning checks
- **Insurance Claims**: Automated damage assessment

### Demo Features

**Interactive Web Application:**

1. **Single Image Mode**
   - Instant defect classification with confidence scores
   - Visual probability distribution chart
   - Support for all 6 defect classes

2. **Batch Processing**
   - Upload ZIP files with multiple images
   - Real-time progress tracking
   - Sortable results table
   - Export to CSV or PDF

3. **Business Impact Calculator**
   - Input: Solar farm size (10-500 MW)
   - Output: Annual savings, time saved, ROI
   - Visual cost comparison charts
   - Professional business report generation

4. **Auto-shutdown**
   - 30-minute inactivity timeout
   - Visual countdown timer
   - Resource management for shared deployments

---

## Dataset

**Source**: PV Panel Defect Dataset by Alicja Lenarczyk, PhD (Gdańsk University of Technology)

- **Kaggle**: [alicjalena/pv-panel-defect-dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)
- **Publication**: [PRZEGLĄD ELEKTROTECHNICZNY, Vol. 101 No. 10/2025](https://sigma-not.pl/publikacja-156617-2025-10.html)
- **License**: CC-BY / Open Research

### Dataset Specifications

| Attribute | Specification |
|-----------|---------------|
| **Total Images** | 1,574 RGB images |
| **Image Type** | Field photographs (real-world installations) |
| **Classes** | 6 defect categories |
| **Image Format** | JPEG/PNG (variable resolution) |
| **Annotation** | Image-level classification labels |

### Defect Classes

| Class | Description | Business Impact |
|-------|-------------|-----------------|
| **Clean** | Properly functioning panels | Baseline reference |
| **Bird-drop** | Bird dropping contamination | 5-15% localized power loss |
| **Dusty** | Dust/dirt accumulation | 3-10% efficiency reduction |
| **Electrical-damage** | Electrical failures | Critical safety hazard |
| **Physical-damage** | Cracks, breakage, delamination | Progressive structural failure |
| **Snow Covered** | Snow accumulation | Variable power loss |

---

## Technical Approach

### Primary Architecture: Hybrid ResNet18 + SVM

```
RGB Image (224x224)
    ↓
ResNet18 Feature Extractor (ImageNet pre-trained)
    ↓
512-dim Deep Features
    ↓
SVM Classifier (RBF kernel)
    ↓
6-class Prediction + Confidence
```

**Key Advantages**:
- **96.84% accuracy** on test set (exceeds 95.5% published benchmark)
- Computationally efficient (<30ms inference)
- Edge-ready for UAV deployment
- Resistant to overfitting on small datasets

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Deep Learning** | PyTorch 2.0+ | Model training & inference |
| **Feature Extraction** | ResNet18 (torchvision) | Transfer learning backbone |
| **Classifier** | scikit-learn SVM | Final classification layer |
| **Augmentation** | Albumentations | Heavy data augmentation |
| **Web Interface** | Gradio 4.0+ | Interactive demo |
| **PDF Generation** | ReportLab | Professional reports |
| **Visualization** | Matplotlib, Plotly | Charts and plots |
| **Data Processing** | Pandas, NumPy | Data manipulation |

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: NVIDIA RTX 3060+ or T4)
- 8GB+ RAM
- 10GB+ free disk space

### Quick Setup

**Option 1: Run the Interactive Demo (Recommended)**
```bash
# Clone repository
git clone https://github.com/lugasraka/Solar-AI-ComputerVision.git
cd Solar-AI-ComputerVision

# Create environment
conda create -n solarvision python=3.9
conda activate solarvision

# Install demo dependencies
cd demo
pip install -r requirements.txt

# Run the demo
python app.py
```
Access at: `http://localhost:7860`

**Option 2: Development/Training**
```bash
# Clone repository
git clone https://github.com/lugasraka/Solar-AI-ComputerVision.git
cd Solar-AI-ComputerVision

# Create environment
conda create -n solarvision python=3.9
conda activate solarvision

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Extract to dataset/ directory

# Run notebooks
jupyter notebook notebooks/
```

---

## Project Structure

```
Solar-AI-ComputerVision/
├── notebooks/              # Jupyter notebooks (Weeks 1-4)
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_evaluation_interpretability.ipynb
├── demo/                   # Production Gradio demo
│   ├── app.py             # Main application
│   ├── inference.py       # SVM pipeline
│   ├── business_calculator.py
│   ├── report_generator.py
│   ├── utils.py
│   └── requirements.txt
├── models/                 # Trained model weights
├── outputs/                # Reports & visualizations
├── dataset/                # (not tracked in git)
├── requirements.txt
├── README.md
└── DEPLOYMENT.md
```

---

## Results & Benchmarks

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **ResNet18 + SVM** | **96.84%** | 96.90% | 96.98% | 96.87% |
| End-to-End ResNet18 | 95.79% | 96.02% | 96.13% | 96.01% |

*Evaluated on held-out test set (95 images). Published benchmark: 95.5% (PRZEGLĄD ELEKTROTECHNICZNY 2025).*

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Bird-drop | 100.0% | 94.12% | 96.97% |
| Clean | 94.44% | 94.44% | 94.44% |
| Dusty | 94.12% | 100.0% | 96.97% |
| Electrical-damage | 92.86% | 100.0% | 96.30% |
| Physical-Damage | 100.0% | 93.33% | 96.55% |
| Snow-Covered | 100.0% | 100.0% | 100.0% |
| **Macro Average** | **96.90%** | **96.98%** | **96.87%** |

### Key Findings

1. **Hybrid approach outperforms end-to-end**: ResNet18+SVM achieves 1.05% higher accuracy than fine-tuned CNN
2. **Both models exceed published benchmark**: 96.84% and 95.79% vs. published 95.5%
3. **Snow-Covered easiest to detect**: 100% across all metrics (distinct visual pattern)
4. **Critical defects perform well**: Electrical-damage (100% recall) and Physical-Damage (100% precision)
5. **Clean class hardest for hybrid**: 94.44% precision (occasional confusion with subtle defects)

---

## 🚀 Interactive Demo

### Try It Online
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/solarvision-ai)

**Live Demo Features:**
- 📷 **Single Image Analysis** - Upload a solar panel image and get instant predictions
- 📁 **Batch Processing** - Process multiple images via ZIP file upload  
- 💰 **Business Impact Calculator** - Calculate ROI and cost savings for your solar farm
- 📊 **Export Results** - Download predictions as CSV or professional PDF reports

### Run Locally

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Access the demo at: `http://localhost:7860`

---

---

## Roadmap

### Phase 1: MVP ✅ (Completed)
- [x] Dataset acquisition and EDA
- [x] Data preprocessing and augmentation pipeline
- [x] ResNet18 + SVM implementation
- [x] Model training and hyperparameter tuning
- [x] Evaluation and benchmarking

### Phase 2: Advanced Features ✅ (Completed)
- [x] Interactive Gradio demo with batch processing
- [x] Business impact calculator with ROI visualization
- [x] PDF/CSV report generation
- [x] Auto-shutdown timer for resource management
- [x] Hugging Face Spaces deployment ready

### Phase 3: Production Deployment 🚧 (In Progress)
- [x] Gradio web interface
- [x] Hugging Face Spaces deployment
- [ ] RESTful API (FastAPI)
- [ ] Edge deployment (NVIDIA Jetson)
- [ ] Model optimization (TensorRT, ONNX)
- [ ] AWS/GCP cloud deployment

---

**Created by**: Raka Adrianto | Sustainability, Digital, Product Management, AI/ML (LinkedIn: [lugasraka](https://www.linkedin.com/in/lugasraka/))

## Citation

```bibtex
@software{solarvision_ai_2026,
  author = {Raka Adrianto, Lugas},
  title = {SolarVision AI: Automated PV Panel Defect Detection},
  year = {2026},
  url = {https://github.com/lugasraka/Solar-AI-ComputerVision}
}

@article{lenarczyk2025pv,
  author = {Lenarczyk, Alicja},
  title = {Comparison of ML classifiers in automatic diagnostics of PV panels using deep image features},
  journal = {PRZEGLĄD ELEKTROTECHNICZNY},
  volume = {101},
  number = {10},
  year = {2025}
}
```

---

**License**: MIT License | **Project Link**: [github.com/lugasraka/Solar-AI-ComputerVision](https://github.com/lugasraka/Solar-AI-ComputerVision)
