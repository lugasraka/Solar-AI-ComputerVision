# SolarVision AI - Deployment Guide

## Week 5: Production Demo

This directory contains the production-ready Gradio demo for SolarVision AI.

## Quick Start

### 1. Local Development

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Access the demo at: `http://localhost:7860`

### 2. Hugging Face Spaces Deployment

#### Option A: Direct Upload

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Select "Gradio" as the SDK
4. Upload all files from the `Space/` directory:
   - app.py
   - inference.py
   - business_calculator.py
   - report_generator.py
   - utils.py
   - requirements.txt
   - README.md

#### Option B: Git Integration

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/solarvision-ai

# Copy files
cp -r Space/* solarvision-ai/

# Push to deploy
cd solarvision-ai
git add .
git commit -m "Initial deployment"
git push
```

### 3. Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY Space/requirements.txt .
RUN pip install -r requirements.txt

COPY Space/* .

EXPOSE 7860

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t solarvision-demo .
docker run -p 7860:7860 solarvision-demo
```

## Features

### Single Image Mode
- Upload JPG/PNG images
- Instant prediction with confidence scores
- Visual probability chart
- Support for all 6 defect classes

### Batch Processing
- Upload ZIP files with multiple images
- Real-time progress tracking
- Results table with sortable columns
- Export to CSV or PDF

### Business Impact Calculator
- Input: Solar farm size (MW)
- Output: Annual savings, time saved, ROI
- Visual cost comparison charts
- Professional business report generation

### Auto-shutdown
- 30-minute inactivity timeout
- Visual countdown timer
- Warning at 25 minutes
- Graceful resource cleanup

## File Structure

```
demo/
├── app.py                    # Main Gradio application
├── inference.py              # SVM prediction pipeline
├── business_calculator.py    # ROI calculator
├── report_generator.py       # PDF/CSV export
├── utils.py                  # Auto-shutdown & helpers
├── requirements.txt          # Dependencies
└── README.md                 # Documentation

Space/                        # Hugging Face Spaces files
├── app.py                    # Entry point
├── (copied demo files)
├── requirements.txt
└── README.md
```

## Model Requirements

The demo expects trained models in `../models/`:
- `svm_classifier.pkl` - Trained SVM model
- `resnet18_features.pkl` - Feature extractor (optional)

## Environment Variables

Optional configuration:
- `MODELS_DIR` - Path to models directory (default: ../models)
- `TIMEOUT_MINUTES` - Auto-shutdown timeout (default: 30)

## Troubleshooting

### Models not loading
Ensure the `models/` directory contains:
- `svm_classifier.pkl` (4.4 MB)
- Other model files from Week 3

### Memory issues
For large batch processing:
- Reduce batch size
- Process images sequentially
- Monitor system resources

### Port already in use
Change the port in `app.py`:
```python
app.launch(server_port=7861)  # Use different port
```

## Support

For issues or questions:
- GitHub: [github.com/lugasraka/Solar-AI-ComputerVision](https://github.com/lugasraka/Solar-AI-ComputerVision)
- Dataset: [Kaggle - PV Panel Defect Dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)

## License

MIT License - See LICENSE file
