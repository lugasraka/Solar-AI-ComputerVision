# SolarVision AI - Gradio Demo

Interactive web demo for automated PV panel defect detection using AI.

## Features

- **Single Image Analysis**: Upload an image and get instant predictions
- **Batch Processing**: Process multiple images via ZIP upload
- **Business Impact Calculator**: Calculate ROI for solar farms
- **Export Options**: Download results as CSV or PDF reports
- **Auto-shutdown**: Demo closes after 30 minutes of inactivity

## Model

- **Architecture**: ResNet18 + SVM (Hybrid approach)
- **Accuracy**: 96.8% on test set
- **Classes**: 6 defect types (Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered)

## Running Locally

```bash
# Navigate to demo directory
cd demo

# Install dependencies
pip install -r requirements.txt

# Run the demo
python app.py
```

The demo will be available at `http://localhost:7860`

## Project Structure

```
demo/
├── app.py              # Main Gradio application
├── inference.py        # Model loading and prediction
├── report_generator.py # PDF and CSV export
├── business_calculator.py # ROI calculations
├── utils.py            # Auto-shutdown timer
├── requirements.txt    # Dependencies
└── examples/           # Demo example images
```

## Deployment

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose "Gradio" as the SDK
3. Upload all files from this directory
4. The app will automatically deploy

### Docker (Optional)

```bash
docker build -t solarvision-demo .
docker run -p 7860:7860 solarvision-demo
```

## Dataset

This demo uses the PV Panel Defect Dataset by Alicja Lenarczyk, PhD (Gdańsk University of Technology).

## License

MIT License - See LICENSE file for details
