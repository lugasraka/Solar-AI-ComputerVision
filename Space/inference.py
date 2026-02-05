"""
SolarVision AI - Inference Pipeline
Handles SVM and CNN models with Grad-CAM support
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

# Grad-CAM imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[WARNING] grad-cam not installed. Grad-CAM visualizations will be disabled.")


class ResNet18Classifier(nn.Module):
    """ResNet18 CNN model for end-to-end classification"""
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class SolarVisionPredictor:
    """Unified inference pipeline for SolarVision AI with dual model support"""
    
    def __init__(self, models_dir='../models'):
        """Initialize predictor"""
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
        
        # Models (lazy loading)
        self.svm_model = None
        self.feature_extractor = None
        self.cnn_model = None
        self.gradcam = None
        self._svm_loaded = False
        self._cnn_loaded = False
        
    def _load_svm_model(self):
        """Lazy load SVM model and feature extractor"""
        if self._svm_loaded:
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
        
        self._svm_loaded = True
        print("[OK] SVM model loaded successfully")
    
    def _load_cnn_model(self):
        """Lazy load CNN model for Grad-CAM support"""
        if self._cnn_loaded:
            return
            
        print("[INFO] Loading CNN model...")
        
        # Load CNN model
        self.cnn_model = ResNet18Classifier(num_classes=6)
        cnn_path = self.models_dir / 'resnet18_end2end_best.pth'
        
        if not cnn_path.exists():
            raise FileNotFoundError(f"CNN model not found at {cnn_path}")
        
        self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        # Initialize Grad-CAM
        if GRADCAM_AVAILABLE:
            target_layers = [self.cnn_model.resnet.layer4[-1]]
            self.gradcam = GradCAM(model=self.cnn_model, target_layers=target_layers)
        
        self._cnn_loaded = True
        print("[OK] CNN model loaded successfully")
    
    def predict(self, image_path, use_cnn=False, return_features=False):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            use_cnn: If True, use CNN model (supports Grad-CAM). If False, use SVM.
            return_features: If True, also return extracted features
            
        Returns:
            dict with prediction results
        """
        # Load appropriate model
        if use_cnn:
            self._load_cnn_model()
            return self._predict_cnn(image_path, return_features)
        else:
            self._load_svm_model()
            return self._predict_svm(image_path, return_features)
    
    def _predict_svm(self, image_path, return_features=False):
        """SVM prediction pipeline"""
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
            'top3': top3,
            'model_used': 'SVM (ResNet18 + SVM)',
            'model_accuracy': '96.84%'
        }
        
        if return_features:
            result['features'] = features[0]
            
        return result
    
    def _predict_cnn(self, image_path, return_features=False):
        """CNN prediction pipeline"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # CNN prediction
        with torch.no_grad():
            outputs = self.cnn_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_proba = probabilities[0].cpu().numpy()
        
        pred_label = np.argmax(pred_proba)
        
        # Get top 3 predictions
        top3_idx = np.argsort(pred_proba)[-3:][::-1]
        top3 = [(self.class_names[i], float(pred_proba[i])) for i in top3_idx]
        
        result = {
            'filename': Path(image_path).name,
            'predicted_class': self.class_names[pred_label],
            'confidence': float(pred_proba[pred_label]),
            'all_probabilities': {cls: float(prob) for cls, prob in zip(self.class_names, pred_proba)},
            'top3': top3,
            'model_used': 'CNN (End-to-End ResNet18)',
            'model_accuracy': '95.79%',
            'supports_gradcam': GRADCAM_AVAILABLE
        }
        
        return result
    
    def generate_gradcam(self, image_path, target_class=None):
        """
        Generate Grad-CAM visualization
        
        Args:
            image_path: Path to image file
            target_class: Class to visualize (None = use predicted class)
            
        Returns:
            dict with 'heatmap', 'overlay', and 'original' images (as numpy arrays)
        """
        if not GRADCAM_AVAILABLE:
            raise RuntimeError("Grad-CAM not available. Install with: pip install grad-cam")
        
        self._load_cnn_model()
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((self.img_size, self.img_size))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get prediction if target_class not specified
        if target_class is None:
            with torch.no_grad():
                outputs = self.cnn_model(img_tensor)
                target_class = outputs.argmax(dim=1).item()
        else:
            target_class = self.class_to_idx.get(target_class, target_class)
        
        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.gradcam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Prepare original image for overlay
        img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
        
        # Create visualization
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
        
        return {
            'original': np.array(img_resized),
            'heatmap': heatmap,
            'overlay': overlay,
            'target_class': self.class_names[target_class],
            'target_class_idx': target_class
        }
    
    def predict_with_gradcam(self, image_path):
        """
        Predict using CNN and generate Grad-CAM visualization
        
        Args:
            image_path: Path to image file
            
        Returns:
            tuple: (prediction_result, gradcam_images)
        """
        # Get prediction using CNN
        result = self.predict(image_path, use_cnn=True)
        
        # Generate Grad-CAM
        if GRADCAM_AVAILABLE:
            gradcam_images = self.generate_gradcam(image_path, target_class=None)
        else:
            gradcam_images = None
        
        return result, gradcam_images
    
    def predict_batch(self, image_paths, use_cnn=False, progress_callback=None):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            use_cnn: If True, use CNN model. If False, use SVM.
            progress_callback: Optional callback function(progress_pct)
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(image_paths)
        
        for i, img_path in enumerate(image_paths):
            try:
                result = self.predict(img_path, use_cnn=use_cnn)
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
    
    def get_model_info(self):
        """Get information about available models"""
        return {
            'svm': {
                'name': 'SVM (ResNet18 + SVM)',
                'accuracy': '96.84%',
                'supports_gradcam': False,
                'description': 'Higher accuracy, no explainability'
            },
            'cnn': {
                'name': 'CNN (End-to-End ResNet18)',
                'accuracy': '95.79%',
                'supports_gradcam': GRADCAM_AVAILABLE,
                'description': 'Slightly lower accuracy with Grad-CAM explainability'
            }
        }


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
    
    # Example prediction with SVM
    test_image = '../dataset/test/Clean/Clean (1).jpg'
    if os.path.exists(test_image):
        print("\n=== SVM Prediction ===")
        result = predictor.predict(test_image, use_cnn=False)
        print(f"Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Model: {result['model_used']}")
        
        print("\n=== CNN Prediction with Grad-CAM ===")
        result, gradcam = predictor.predict_with_gradcam(test_image)
        print(f"Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Model: {result['model_used']}")
        if gradcam:
            print(f"Grad-CAM generated for class: {gradcam['target_class']}")
    else:
        print("Test image not found")
