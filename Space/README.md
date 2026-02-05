# SolarVision AI - PV Panel Defect Detection

[![Model Accuracy](https://img.shields.io/badge/Accuracy-96.8%25-green)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

**AI-powered defect detection for solar panel installations**

This Space demonstrates SolarVision AI, an automated system for detecting defects in photovoltaic panels using computer vision and machine learning.

## Features

☀️ **Single Image Analysis** - Upload a solar panel image and get instant defect detection  
📁 **Batch Processing** - Process multiple images via ZIP file upload  
💰 **Business Impact Calculator** - Calculate ROI and cost savings for solar farms  
📊 **Export Results** - Download predictions as CSV or professional PDF reports  

## Model Performance

- **Architecture**: ResNet18 + SVM (Hybrid approach)
- **Accuracy**: 96.8% on test set
- **Classes**: 6 defect types
  - Bird-drop
  - Clean (baseline)
  - Dusty
  - Electrical-damage
  - Physical-Damage
  - Snow-Covered

## How to Use

1. **Single Image Mode**: Upload a JPG/PNG image of a solar panel
2. **Batch Mode**: Upload a ZIP file containing multiple images
3. **Business Calculator**: Enter your solar farm size (MW) to calculate savings
4. **Export**: Download results as CSV or PDF reports

## Dataset

This model was trained on the [PV Panel Defect Dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset) by Alicja Lenarczyk, PhD from Gdańsk University of Technology.

## GitHub Repository

[github.com/lugasraka/Solar-AI-ComputerVision](https://github.com/lugasraka/Solar-AI-ComputerVision)

## Citation

```bibtex
@software{solarvision_ai_2026,
  author = {Lugas Raka},
  title = {SolarVision AI: Automated PV Panel Defect Detection},
  year = {2026},
  url = {https://github.com/lugasraka/Solar-AI-ComputerVision}
}
```

## License

MIT License
