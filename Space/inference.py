"""
SolarVision AI - Inference Pipeline
Handles SVM model loading and predictions
"""

import os
import pickle
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class SolarVisionPredictor:
    """Unified inference pipeline for SolarVision AI"""
    
    def __init__(self, models_dir='../models'):
        """Initialize predictor with SVM model"""
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class mapping
        self.class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 
                           'Physical-Damage', 'Snow-Covered']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Image preprocessing
        self.img_size = 224
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
        
        # Lazy loading - models loaded on first prediction
        self.svm_model = None
        self.feature_extractor = None
        self._models_loaded = False
        
    def _load_models(self):
        """Lazy load SVM model and feature extractor"""
        if self._models_loaded:
            return
            
        print("[INFO] Loading SVM model...")
        
        # Load SVM classifier
        svm_path = self.models_dir / 'svm_classifier.pkl'
        with open(svm_path, 'rb') as f:
            self.svm_model = pickle.load(f)
        
        # Load ResNet18 feature extractor
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        self._models_loaded = True
        print("[OK] Models loaded successfully")
    
    def predict(self, image_path, return_features=False):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            return_features: If True, also return extracted features
            
        Returns:
            dict with prediction results
        """
        self._load_models()
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
            features = features.view(features.size(0), -1).cpu().numpy()
        
        # SVM prediction
        pred_label = self.svm_model.predict(features)[0]
        pred_proba = self.svm_model.predict_proba(features)[0]
        
        # Get top 3 predictions
        top3_idx = np.argsort(pred_proba)[-3:][::-1]
        top3 = [(self.class_names[i], float(pred_proba[i])) for i in top3_idx]
        
        result = {
            'filename': Path(image_path).name,
            'predicted_class': self.class_names[pred_label],
            'confidence': float(pred_proba[pred_label]),
            'all_probabilities': {cls: float(prob) for cls, prob in zip(self.class_names, pred_proba)},
            'top3': top3
        }
        
        if return_features:
            result['features'] = features[0]
            
        return result
    
    def predict_batch(self, image_paths, progress_callback=None):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            progress_callback: Optional callback function(progress_pct)
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(image_paths)
        
        for i, img_path in enumerate(image_paths):
            try:
                result = self.predict(img_path)
                results.append(result)
                
                if progress_callback:
                    progress = int((i + 1) / total * 100)
                    progress_callback(progress)
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
                results.append({
                    'filename': Path(img_path).name,
                    'error': str(e)
                })
        
        return results
    
    def get_supported_extensions(self):
        """Return list of supported image extensions"""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def is_valid_image(self, filepath):
        """Check if file is a valid image"""
        ext = Path(filepath).suffix.lower()
        return ext in self.get_supported_extensions()


# Singleton instance for reuse
_predictor = None

def get_predictor(models_dir='../models'):
    """Get or create predictor singleton"""
    global _predictor
    if _predictor is None:
        _predictor = SolarVisionPredictor(models_dir)
    return _predictor


if __name__ == '__main__':
    # Test the predictor
    predictor = SolarVisionPredictor()
    
    # Example prediction
    test_image = '../dataset/test/Clean/Clean (1).jpg'
    if os.path.exists(test_image):
        result = predictor.predict(test_image)
        print(f"\nPrediction Results:")
        print(f"  Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Top 3: {result['top3']}")
    else:
        print("Test image not found")
