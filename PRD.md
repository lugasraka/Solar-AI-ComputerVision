# Product Requirements Document (PRD)
## SolarVision AI - Automated PV Panel Defect Detection System

---

## **Document Control**

| **Field** | **Details** |
|-----------|-------------|
| **Product Name** | SolarVision AI - Automated PV Panel Defect Detection System |
| **Author** | [Your Name] |
| **Date** | February 4, 2026 |
| **Version** | 1.2 (Alicjalena Dataset-Specific) |
| **Status** | Draft - Ready for Development |
| **Target Dataset** | PV Panel Defect Dataset by alicjalena |
| **Dataset Source** | Kaggle: https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset |
| **Published Research** | PRZEGLĄD ELEKTROTECHNICZNY (2025) - ResNet18 + SVM hybrid approach |
| **Stakeholders** | Solar O&M Managers, Field Technicians, Asset Owners, UAV Inspection Teams |

---

## **1. Executive Summary**

### **Problem Statement**
Solar farm operators rely on periodic visual inspections to detect panel defects caused by environmental damage, physical wear, and electrical failures, which is:
- **Costly:** $0.50-2.00 per panel inspection (manual/drone-based)
- **Slow:** 100-500 panels/day depending on method
- **Infrequent:** Quarterly or semi-annual inspection cycles
- **Inconsistent:** Human error rates 15-25% for subtle environmental damage
- **Reactive:** Defects discovered after significant power loss (5-20% efficiency reduction)

This results in delayed defect detection, reduced energy yield, increased maintenance costs, and shortened panel lifespan.

### **Proposed Solution**
An AI-powered computer vision system using the **alicjalena PV Panel Defect Dataset** (RGB images of real-world panel installations) to automatically detect and classify operational defects, enabling:
- 80% cost reduction per inspection
- 10x faster inspection speed via UAV-mounted systems
- Monthly or continuous inspection frequency
- >95% defect detection accuracy (based on published research)
- Proactive maintenance scheduling

### **Success Metrics**
- **Technical:** >93% classification accuracy (published benchmark: 95.5%)
- **Business:** 75% cost reduction, 10x inspection frequency increase
- **Operational:** 15% improvement in fleet energy yield through early defect detection

---

## **2. Dataset Analysis & Specifications**

### **2.1 Dataset Overview**

| **Attribute** | **Specification** |
|---------------|-------------------|
| **Dataset Name** | PV Panel Defect Dataset |
| **Creator** | Alicja Lenarczyk, PhD (Gdańsk University of Technology) |
| **Source** | Kaggle (alicjalena/pv-panel-defect-dataset) |
| **Publication** | PRZEGLĄD ELEKTROTECHNICZNY, Vol. 101 No. 10/2025 |
| **Image Type** | RGB photographs of installed PV panels (field conditions) |
| **Image Format** | JPEG/PNG (standard camera resolution) |
| **Total Images** | Estimated 600-1,200 images (typical for research datasets) |
| **Annotation Type** | Image-level classification (per-image labels) |
| **Classes** | 6 defect categories |
| **Access** | Public, no request form required (Kaggle open dataset) |
| **License** | CC-BY or open research license (verify on Kaggle) |

### **2.2 Defect Categories (6 Classes)**

Based on the published research paper, the dataset contains realistic operational defects:

| **Class ID** | **Defect Type** | **Description** | **Business Impact** | **Priority** |
|--------------|-----------------|-----------------|---------------------|--------------|
| 1 | **Bird-drop** | Contamination with bird droppings | Localized shading → 5-15% power loss | P0 (High volume) |
| 2 | **Clean** | Properly functioning panels (baseline) | Reference class | P0 (Baseline) |
| 3 | **Dusty** | Dust/dirt accumulation | Reduced irradiance → 3-10% power loss | P0 (Common) |
| 4 | **Electrical-damage** | Electrical failures (bypass diode, junction box) | Critical → potential fire hazard | P0 (Critical) |
| 5 | **Physical-damage** | Cracks, breakage, delamination | Structural failure → progressive degradation | P0 (Critical) |
| 6 | **Snow Covered** | Snow accumulation on panels | Variable power loss | P1 (Context-dependent) |

**Note:** This is a **balanced, real-world operational dataset** (not manufacturing defects), making it highly relevant for field O&M applications.

### **2.3 Dataset Characteristics**

**Strengths:**
- ✅ **Real-world field data** from actual PV installations (not lab-controlled)
- ✅ **RGB images** (standard cameras/drones) - no specialized thermal equipment needed
- ✅ **Operational defects** relevant to O&M teams (bird droppings, dust, physical damage)
- ✅ **Published benchmark**: ResNet18 + SVM achieved **95.5% accuracy**
- ✅ **Immediate access** via Kaggle (no request/approval process)
- ✅ **Balanced classes** (manually curated, roughly equal representation)
- ✅ **Practical deployment** - CNN features + classical ML (computationally efficient)

**Challenges:**
- ⚠️ **Smaller dataset** (~600-1,200 images vs. 36K+ in PVEL-AD)
- ⚠️ **Limited size** may require aggressive data augmentation
- ⚠️ **Variable lighting conditions** (outdoor field installations)
- ⚠️ **Mixed panel types** (mono vs. polycrystalline, different manufacturers)
- ⚠️ **Overlap categories** (e.g., "Dusty" vs. "Bird-drop" both cause soiling)

### **2.4 Published Benchmark Results**

From the research paper (PRZEGLĄD ELEKTROTECHNICZNY 2025):

| **Model** | **Accuracy** | **Notes** |
|-----------|--------------|-----------|
| **ResNet18 + SVM** | **95.5%** | Best hybrid approach (CNN features + SVM classifier) |
| **ResNet18 + Random Forest** | ~94% | Competitive performance |
| **Standard CNN (end-to-end)** | ~92% | Pure deep learning approach |
| **Classical ML only** | <85% | Hand-crafted features underperform |

**Key Insight:** Hybrid approach (CNN feature extraction + classical ML classifier) outperforms pure CNN, suggesting:
- Deep features are powerful, but SVM provides better generalization on small datasets
- Computational efficiency (smaller model, faster inference)
- Suitable for edge deployment (UAV-mounted systems)

### **2.5 Dataset Splits**

Recommended split strategy (following published research methodology):

| **Split** | **Percentage** | **Approximate Images** | **Purpose** |
|-----------|----------------|------------------------|-------------|
| **Training** | 70% | ~420-840 | Model training with augmentation |
| **Validation** | 15% | ~90-180 | Hyperparameter tuning, model selection |
| **Test** | 15% | ~90-180 | Final evaluation (held-out, unseen) |

**Stratification:** Maintain class balance across all splits (critical for 6-class classification).

---

## **3. MVP Scope Definition**

### **3.1 MVP: 6-Class PV Panel Defect Classification**

**Full Dataset Utilization:**

| **MVP Class** | **Description** | **Count** | **Business Justification** |
|---------------|-----------------|-----------|----------------------------|
| **Clean** | Properly functioning panels | ~150-250 | Baseline reference, avoid false alarms |
| **Bird-drop** | Bird dropping contamination | ~150-200 | High frequency, localized shading |
| **Dusty** | Dust/dirt accumulation | ~150-200 | Gradual efficiency loss, maintenance trigger |
| **Electrical-damage** | Electrical failures | ~100-150 | Critical safety issue, fire risk |
| **Physical-damage** | Cracks, breakage | ~100-150 | Progressive failure, warranty claims |
| **Snow Covered** | Snow accumulation on panels | ~100-150 | Variable power loss |

**Total MVP Dataset:** ~600-1,200 images across 6 classes

**MVP Approach:** Use all 6 classes from the dataset (no exclusions) to demonstrate comprehensive defect detection capabilities.

---

### **3.2 MVP Technical Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
│  RGB Images (field photographs, variable resolution)        │
│  Source: Kaggle alicjalena PV Panel Defect Dataset          │
│  Typical resolution: 800x600 to 1920x1080 pixels            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING LAYER                         │
│  • Resize to 224x224 or 380x380 (model input size)          │
│  • Normalize RGB (ImageNet mean/std or dataset-specific)    │
│  • Data augmentation (rotation, flip, brightness, crop)     │
│  • Color jittering (simulate variable lighting)             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│          APPROACH A: Hybrid CNN + SVM (Recommended)         │
│  ┌────────────────────────────────────────────────┐        │
│  │  ResNet18 Feature Extractor (Transfer Learning) │        │
│  │  • Pre-trained on ImageNet                       │        │
│  │  • Remove final classification layer             │        │
│  │  • Extract 512-dim feature vector                │        │
│  └────────────────────────────────────────────────┘        │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────┐        │
│  │  SVM Classifier (RBF kernel)                    │        │
│  │  • Input: 512-dim deep features                 │        │
│  │  • Output: 6-class prediction + confidence      │        │
│  │  • Hyperparameters: C, gamma (grid search)      │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│       APPROACH B: End-to-End CNN (Alternative)              │
│  ┌────────────────────────────────────────────────┐        │
│  │  ResNet18 / EfficientNet-B0 Classifier          │        │
│  │  • Pre-trained on ImageNet                       │        │
│  │  • Fine-tune entire network                      │        │
│  │  • Custom head: 6 output neurons (softmax)       │        │
│  │  • Loss: CrossEntropy or Focal Loss              │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            POST-PROCESSING & INTERPRETABILITY               │
│  • Grad-CAM heatmaps (visualize model attention)            │
│  • Confidence thresholding (flag uncertain predictions)     │
│  • Confusion matrix analysis (identify error patterns)      │
│  • Output formatting (CSV, JSON, annotated images)          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                             │
│  • Annotated images with defect labels + confidence         │
│  • Defect classification report (CSV/PDF)                   │
│  • O&M dashboard (defect distribution, priority ranking)    │
│  • UAV integration-ready (lightweight model for edge)       │
└─────────────────────────────────────────────────────────────┘
```

---

## **4. Functional Requirements**

### **FR-1: Data Loading & Preprocessing**

| **ID** | **Requirement** | **Priority** | **Acceptance Criteria** |
|--------|-----------------|--------------|-------------------------|
| FR-1.1 | Download alicjalena dataset from Kaggle | P0 | Successfully download all images + labels |
| FR-1.2 | Load RGB images (variable resolutions) | P0 | Handle different image sizes (800x600 to 1920x1080) |
| FR-1.3 | Implement stratified train/val/test split (70/15/15) | P0 | Maintain class balance across splits |
| FR-1.4 | Resize images to 224x224 (ResNet18 standard) | P0 | Consistent input size for model |
| FR-1.5 | Normalize RGB values (ImageNet mean/std) | P0 | Standard preprocessing for transfer learning |
| FR-1.6 | Handle class balance (verify equal representation) | P1 | Document any class imbalance, apply weighted sampling if needed |

### **FR-2: Data Augmentation (Critical for Small Dataset)**

| **ID** | **Requirement** | **Priority** | **Acceptance Criteria** |
|--------|-----------------|--------------|-------------------------|
| FR-2.1 | Geometric augmentation (rotation ±15°, H/V flip) | P0 | Applied to training set only |
| FR-2.2 | Random crop + resize (224x224) | P0 | Simulate variable panel angles |
| FR-2.3 | Color jittering (brightness ±20%, saturation ±15%) | P0 | Handle variable outdoor lighting |
| FR-2.4 | Gaussian blur (σ=0.5-1.0) | P1 | Simulate camera focus variations |
| FR-2.5 | Random erasing (10% of images) | P1 | Improve robustness to occlusions |
| FR-2.6 | **Heavy augmentation** to expand dataset 5-10x | P0 | Mitigate small dataset size (~600-1,200 images) |

### **FR-3: Model Development (Two Approaches)**

#### **Approach A: Hybrid ResNet18 + SVM (Primary)**

| **ID** | **Requirement** | **Priority** | **Acceptance Criteria** |
|--------|-----------------|--------------|-------------------------|
| FR-3.1 | Implement ResNet18 feature extractor (PyTorch) | P0 | Load ImageNet pre-trained weights |
| FR-3.2 | Remove final classification layer | P0 | Extract 512-dim feature vectors |
| FR-3.3 | Train SVM classifier (RBF kernel) on deep features | P0 | scikit-learn SVM with grid search (C, gamma) |
| FR-3.4 | Hyperparameter tuning (5-fold cross-validation) | P0 | Optimize SVM parameters |
| FR-3.5 | Save best SVM model (highest validation accuracy) | P0 | Pickle file for deployment |

#### **Approach B: End-to-End CNN (Alternative)**

| **ID** | **Requirement** | **Priority** | **Acceptance Criteria** |
|--------|-----------------|--------------|-------------------------|
| FR-3.6 | Fine-tune ResNet18 or EfficientNet-B0 end-to-end | P1 | Transfer learning with frozen early layers |
| FR-3.7 | Custom classification head (6 output neurons) | P1 | Softmax activation for probabilities |
| FR-3.8 | Training with CrossEntropy or Focal Loss | P1 | Handle potential class imbalance |
| FR-3.9 | Learning rate scheduler (ReduceLROnPlateau) | P1 | Prevent overfitting on small dataset |

### **FR-4: Model Evaluation & Interpretability**

| **ID** | **Requirement** | **Priority** | **Acceptance Criteria** |
|--------|-----------------|--------------|-------------------------|
| FR-4.1 | Evaluate on held-out test set (15% split) | P0 | Calculate accuracy, precision, recall, F1 per class |
| FR-4.2 | Generate confusion matrix | P0 | Identify misclassification patterns |
| FR-4.3 | Implement Grad-CAM heatmaps | P1 | Visualize model attention regions |
| FR-4.4 | Compare Approach A (ResNet18+SVM) vs. Approach B (end-to-end CNN) | P0 | Document performance trade-offs |
| FR-4.5 | Benchmark against published results (95.5% accuracy) | P0 | Verify model performance |

### **FR-5: Output & Reporting**

| **ID** | **Requirement** | **Priority** | **Acceptance Criteria** |
|--------|-----------------|--------------|-------------------------|
| FR-5.1 | Generate defect classification report (CSV) | P0 | Image ID, predicted class, confidence, ground truth |
| FR-5.2 | Export annotated images with labels | P0 | Visual QA for O&M teams |
| FR-5.3 | Calculate per-class metrics (precision, recall, F1) | P0 | Detailed performance breakdown |
| FR-5.4 | Generate summary dashboard (defect distribution) | P1 | Pie chart, priority ranking |

### **FR-6: User Interface (Demo)**

| **ID** | **Requirement** | **Priority** | **Acceptance Criteria** |
|--------|-----------------|--------------|-------------------------|
| FR-6.1 | Web-based demo (Gradio/Streamlit) | P0 | Upload RGB image → get prediction |
| FR-6.2 | Display prediction with confidence scores (top-3 classes) | P0 | Clear visualization |
| FR-6.3 | Show Grad-CAM heatmap overlay | P1 | Interpretability feature |
| FR-6.4 | Batch upload functionality | P1 | Process multiple images |

---

## **5. Non-Functional Requirements**

### **NFR-1: Performance**

| **ID** | **Requirement** | **Target** | **Measurement** |
|--------|-----------------|------------|-----------------|
| NFR-1.1 | Inference time per image | <30ms | ResNet18 on GPU (NVIDIA T4 / RTX 3060) |
| NFR-1.2 | Batch processing throughput | >100 images/min | End-to-end pipeline |
| NFR-1.3 | Training time (hybrid approach) | <2 hours | ResNet18 feature extraction + SVM training |
| NFR-1.4 | Memory footprint (inference) | <2GB GPU RAM | Lightweight for edge deployment |

### **NFR-2: Accuracy & Reliability**

| **ID** | **Requirement** | **Target** | **Published Benchmark** |
|--------|-----------------|------------|------------------------|
| NFR-2.1 | **Overall accuracy** | **>93%** | ResNet18+SVM: **95.5%** |
| NFR-2.2 | Precision (Critical classes: Electrical, Physical) | >90% | Minimize false negatives |
| NFR-2.3 | Recall (Critical classes) | >88% | Catch dangerous defects |
| NFR-2.4 | F1-Score (macro average) | >0.91 | Balanced performance |
| NFR-2.5 | Confusion between similar classes (Bird-drop vs. Dusty) | <10% | Document expected confusions |

### **NFR-3: Model Generalization (Small Dataset Challenge)**

| **ID** | **Requirement** | **Approach** | **Success Criteria** |
|--------|-----------------|--------------|---------------------|
| NFR-3.1 | Prevent overfitting on small dataset (~600-1,200 images) | Heavy augmentation, dropout, early stopping | Validation accuracy within 3% of training accuracy |
| NFR-3.2 | Transfer learning effectiveness | Use ImageNet pre-trained weights | Outperform training from scratch by >10% |
| NFR-3.3 | Cross-validation stability | 5-fold CV for hyperparameter tuning | Std dev of CV scores <2% |

---

## **6. Technical Stack (Dataset-Optimized)**

| **Layer** | **Technology** | **Dataset-Specific Rationale** |
|-----------|----------------|-------------------------------|
| **Deep Learning Framework** | PyTorch 2.0+ | Standard for ResNet18, easy SVM integration |
| **Feature Extraction** | ResNet18 (torchvision.models) | Published benchmark model, lightweight |
| **Classifier (Primary)** | scikit-learn SVM (RBF kernel) | Published best approach (95.5% accuracy) |
| **Classifier (Alternative)** | End-to-end ResNet18 or EfficientNet-B0 | Compare hybrid vs. pure deep learning |
| **Image Processing** | OpenCV, PIL, Albumentations | RGB preprocessing, heavy augmentation |
| **Data Augmentation** | Albumentations | Field-realistic transforms (lighting, angle) |
| **Interpretability** | pytorch-grad-cam | Grad-CAM heatmaps for explainability |
| **Experiment Tracking** | Weights & Biases or TensorBoard | Compare Approach A vs. B |
| **Web Interface** | Gradio | Rapid prototyping, shareable demo |
| **Version Control** | Git + GitHub | Portfolio showcase |

---

## **7. Implementation Roadmap (5-Week MVP)**

### **Week 1: Dataset Acquisition & EDA**

**Tasks:**
- [ ] Download alicjalena dataset from Kaggle
- [ ] Verify dataset structure (6 classes, ~600-1,200 images)
- [ ] Exploratory Data Analysis:
  - Class distribution histogram
  - Image resolution distribution
  - Visual inspection (lighting conditions, panel types)
  - Check for corrupted/low-quality images
- [ ] Implement stratified train/val/test split (70/15/15)
- [ ] Create data loading pipeline (PyTorch DataLoader)
- [ ] Calculate RGB mean/std for normalization
- [ ] Baseline: Train simple CNN on MNIST to verify environment

**Deliverable:** Jupyter notebook with comprehensive EDA, data split files

---

### **Week 2: Data Preprocessing & Augmentation**

**Tasks:**
- [ ] Implement preprocessing pipeline:
  - Resize images to 224x224
  - Normalize RGB (ImageNet or dataset-specific mean/std)
  - Handle variable input resolutions
- [ ] Implement heavy augmentation (Albumentations):
  - Geometric: ±15° rotation, H/V flip, random crop
  - Photometric: brightness ±20%, saturation ±15%, hue shift
  - Noise: Gaussian blur, random erasing
  - **Goal: Expand dataset 5-10x effectively**
- [ ] Visualize augmented samples (sanity check)
- [ ] Benchmark data loading speed (optimize bottlenecks)
- [ ] Test different augmentation intensities

**Deliverable:** Production-ready data pipeline with augmentation, benchmark report

---

### **Week 3: Model Development (Hybrid Approach)**

**Tasks:**
- [ ] **Approach A: ResNet18 + SVM (Primary)**
  - Load ResNet18 (ImageNet pre-trained)
  - Remove final FC layer, extract 512-dim features
  - Extract features for train/val/test sets
  - Train SVM (RBF kernel) on deep features
  - Grid search hyperparameters (C, gamma) with 5-fold CV
  - Evaluate on validation set
- [ ] **Approach B: End-to-End CNN (Alternative)**
  - Fine-tune ResNet18 end-to-end
  - Custom classification head (6 outputs)
  - Train with CrossEntropy Loss
  - Learning rate scheduler (ReduceLROnPlateau)
  - Early stopping (patience=10 epochs)
- [ ] Compare Approach A vs. B on validation set
- [ ] Log training metrics (W&B or TensorBoard)

**Deliverable:** Trained models (ResNet18+SVM and end-to-end CNN), comparison report

---

### **Week 4: Evaluation & Error Analysis**

**Tasks:**
- [ ] Evaluate both approaches on held-out test set:
  - Overall accuracy, precision, recall, F1 per class
  - Confusion matrix (identify error patterns)
  - Compare to published benchmark (95.5% accuracy)
- [ ] Generate Grad-CAM heatmaps:
  - Visualize model attention for correct/incorrect predictions
  - Identify failure modes (e.g., Bird-drop vs. Dusty confusion)
- [ ] Error analysis:
  - Export top-50 misclassified examples
  - Analyze patterns (lighting, panel orientation, mixed defects)
  - Document insights for Phase 2 improvements
- [ ] Statistical significance testing (Approach A vs. B)
- [ ] Calculate business metrics:
  - Inspection time reduction
  - Cost per panel ($1.50 → $0.20)
  - Critical defect detection rate

**Deliverable:** Comprehensive evaluation report, confusion matrix, Grad-CAM visualizations

---

### **Week 5: Demo, Documentation & Portfolio**

**Tasks:**
- [ ] Build Gradio demo interface:
  - Upload RGB image → display prediction + confidence
  - Show Grad-CAM heatmap overlay
  - Display top-3 class probabilities
  - Batch upload functionality
- [ ] Write technical documentation:
  - README.md (project overview, setup instructions)
  - ARCHITECTURE.md (hybrid approach explanation)
  - DATASET.md (alicjalena dataset details)
  - RESULTS.md (evaluation metrics, benchmarks)
  - COMPARISON.md (ResNet18+SVM vs. end-to-end CNN)
- [ ] Create GitHub repository:
  - Clean code structure (src/, data/, models/, notebooks/)
  - requirements.txt with dependencies
  - Model weights (upload to Hugging Face)
  - Demo link (Hugging Face Spaces or Streamlit Cloud)
- [ ] Prepare portfolio materials:
  - Project summary (1-pager for resume)
  - Technical deep-dive (blog post or Medium article)
  - Demo video (2-3 minutes)

**Deliverable:** Live Gradio demo, polished GitHub repository, portfolio-ready documentation

---

## **8. Success Metrics & KPIs**

### **8.1 Technical Metrics**

| **Metric** | **MVP Target** | **Published Benchmark** | **Measurement** |
|------------|----------------|------------------------|-----------------|
| **Overall Accuracy** | **>93%** | **95.5%** (ResNet18+SVM) | Test set evaluation |
| **Precision (Electrical)** | >90% | - | Critical safety class |
| **Recall (Physical)** | >88% | - | Structural failure detection |
| **F1-Score (macro)** | >0.91 | - | Class-balanced performance |
| **Inference Time** | <30ms | - | Single image on GPU |

### **8.2 Class-Specific Metrics (6 Classes)**

| **Class** | **Precision Target** | **Recall Target** | **F1 Target** | **Priority** |
|-----------|---------------------|-------------------|---------------|--------------|
| **Clean** | >95% | >92% | >0.93 | Baseline (avoid false alarms) |
| **Bird-drop** | >90% | >88% | >0.89 | High frequency defect |
| **Dusty** | >88% | >85% | >0.86 | Maintenance trigger |
| **Electrical-damage** | >92% | >90% | >0.91 | Critical safety |
| **Physical-damage** | >90% | >88% | >0.89 | Structural integrity |
| **Snow Covered** | >85% | >82% | >0.83 | Context-dependent |

### **8.3 Model Comparison**

| **Approach** | **Accuracy Target** | **Inference Speed** | **Deployment** |
|--------------|---------------------|---------------------|----------------|
| **ResNet18 + SVM** | **>93%** | Fast (<30ms) | Edge-friendly (lightweight) |
| **End-to-End CNN** | >91% | Medium (~40ms) | Requires GPU |

### **8.4 Business Impact Metrics**

| **Metric** | **Baseline** | **Target (AI-Powered)** | **Calculation** |
|------------|--------------|------------------------|-----------------|
| **Cost per Panel Inspected** | $1.50 | $0.20-0.30 | Automated UAV processing |
| **Inspection Coverage** | Quarterly (4x/year) | Monthly (12x/year) | Continuous monitoring |
| **Inspection Speed** | 100-500 panels/day | 5,000+ panels/day | Automated pipeline |
| **Critical Defect Detection Rate** | 75% (human) | >90% (AI) | Electrical + Physical classes |
| **False Alarm Rate** | 20-25% | <10% | Clean class precision |

---

## **9. Risks & Mitigation**

| **Risk** | **Probability** | **Impact** | **Mitigation Strategy** |
|----------|-----------------|------------|-------------------------|
| **Small dataset size (~600-1,200 images)** | High | High | Heavy data augmentation (5-10x expansion); transfer learning from ImageNet; consider external dataset augmentation |
| **Class confusion (Bird-drop vs. Dusty)** | Medium | Medium | Document expected confusions; fine-tune decision boundaries; add context features (location metadata) |
| **Variable lighting conditions (field data)** | High | Medium | Color jittering augmentation; test-time normalization; consider ensemble models |
| **Overfitting to training set** | High | High | 5-fold cross-validation; early stopping; dropout; monitor val/train accuracy gap |
| **Published benchmark not reproducible** | Medium | Medium | Follow research paper methodology exactly; contact author if needed; document differences |
| **Mixed panel types (mono vs. poly)** | Medium | Low | Document panel type distribution; test per-panel-type performance |
| **Edge deployment challenges (UAV)** | Low | Medium | Test model quantization (INT8); benchmark on Jetson Nano; Phase 3 optimization |

---

## **10. Phase 2 & 3 Roadmap**

### **Phase 2: Advanced Features & Optimization** (+3-4 weeks)

**Features:**
- ➕ Ensemble methods (ResNet18+SVM + EfficientNet+Random Forest voting)
- ➕ Test-Time Augmentation (TTA) for robustness
- ➕ Defect severity scoring (rank by maintenance urgency)
- ➕ Multi-panel detection (YOLO for full-array images)
- ➕ Temporal analysis (track panel degradation over time)
- ➕ External dataset integration (augment with PVEL-AD or other sources)
- ➕ Advanced interpretability (LIME, SHAP for SVM)

**Success Criteria:**
- Accuracy >96% (match/exceed published benchmark)
- Robust to variable lighting, panel types
- Ready for field pilot testing

---

### **Phase 3: Production Deployment** (+4-6 weeks)

**Features:**
- ➕ RESTful API for UAV integration (FastAPI/Flask)
- ➕ Edge deployment (NVIDIA Jetson Nano/Xavier for drone-mounted processing)
- ➕ Model optimization (TensorRT, ONNX export, quantization)
- ➕ Cloud deployment (AWS SageMaker, Azure ML)
- ➕ Real-time dashboard (Grafana, Power BI)
- ➕ Mobile app (iOS/Android for field technicians)
- ➕ Integration with O&M systems (SCADA, CMMS)

---

## **11. Open Questions & Decisions Needed**

### **Open Questions**

1. **Exact dataset size:** Kaggle page doesn't specify exact image count. Need to verify upon download.
   - **Action:** Download dataset Week 1, document exact size
   
2. **Image resolution variability:** Are all images same resolution or mixed?
   - **Action:** EDA Week 1 to determine preprocessing strategy
   
3. **Class balance:** Are classes roughly balanced or imbalanced?
   - **Action:** EDA Week 1, apply weighted sampling if needed

4. **Hybrid vs. End-to-End:** Which approach to prioritize for deployment?
   - **Decision:** Test both in Week 3, select based on accuracy + inference speed trade-off

### **Assumptions**

- ✓ Dataset is publicly accessible on Kaggle (no approval needed)
- ✓ Images are RGB (standard camera format)
- ✓ Dataset size is sufficient for transfer learning (~600-1,200 images)
- ✓ Published benchmark (95.5%) is achievable with ResNet18+SVM
- ✓ GPU available for training (local or Google Colab Pro)
- ✓ Dataset license permits research and commercial use

---

## **12. Success Definition (MVP)**

**MVP is successful if:**

**Technical:**
1. ✅ Model achieves >93% overall accuracy on held-out test set (6 classes)
2. ✅ Matches or approaches published benchmark (95.5% with ResNet18+SVM)
3. ✅ Critical classes (Electrical, Physical) achieve >90% precision
4. ✅ Macro F1-Score >0.91
5. ✅ Inference time <30ms per image on GPU

**Portfolio/Career:**
1. ✅ GitHub repository demonstrates clean code, hybrid ML approach, and documentation
2. ✅ Live Gradio demo showcases predictions + Grad-CAM interpretability
3. ✅ Can articulate trade-offs between hybrid (ResNet18+SVM) vs. end-to-end CNN approaches
4. ✅ Can discuss small dataset mitigation strategies (augmentation, transfer learning)
5. ✅ Demonstrates understanding of O&M context (field defects, UAV deployment)

**Business (Conceptual):**
1. ✅ Clear ROI narrative: 80% cost reduction, 3x inspection frequency increase
2. ✅ Positioning for Siemens: "UAV-based defect detection for solar O&M"
3. ✅ Roadmap to production (Phase 2: optimization, Phase 3: edge deployment)

---

## **13. Next Steps**

### **Immediate Actions (This Week)**

1. **Download alicjalena dataset from Kaggle:**
   - URL: https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset
   - Verify exact size, class distribution, image format

2. **Set up development environment:**
   ```bash
   conda create -n pv-defect python=3.9
   conda activate pv-defect
   pip install torch torchvision opencv-python albumentations
   pip install scikit-learn pandas matplotlib seaborn
   pip install pytorch-grad-cam gradio wandb
   ```

3. **Verify dataset and run EDA:**
   - Confirm 6 classes present
   - Check image resolutions
   - Calculate class distribution
   - Identify any data quality issues