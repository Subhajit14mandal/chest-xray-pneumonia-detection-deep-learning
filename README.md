# 🩺 Medical Image Classification for Pneumonia Detection
Deep learning–based medical image classification system for detecting Pneumonia from chest X-ray images using a transfer learning CNN architecture (Xception).
The model is trained with data augmentation and class-imbalance handling to improve generalization and reliability in medical diagnosis scenarios.

---
## 📌 Project Overview
Pneumonia is a serious lung infection that can be detected through chest X-ray imaging. However, manual diagnosis is time-consuming and requires expert radiologists.

This project builds an automated deep learning pipeline that classifies chest X-ray images into:

- Pneumonia
- Normal

The model leverages transfer learning with the Xception architecture to extract high-level features from medical images and achieve strong classification performance.

---
## 🧠 Model Architecture
The model uses transfer learning with a pre-trained convolutional neural network.

**Base Model**
- Xception (pre-trained on ImageNet)

**Custom Classification Head**

```text
GlobalAveragePooling2D
Dense (256) + ReLU
Dropout (0.3)
Dense (128) + ReLU
Dropout (0.3)
Dense (64) + ReLU
Dropout (0.2)
Dense (32) + ReLU
Dropout (0.2)
Dense (16) + ReLU
Dropout (0.1)
Dense (1) + Sigmoid
```
**Loss Function**
```text
Binary Crossentropy
```
**Evaluation Metrics**
- Accuracy
- AUC
- Recall
- Precision
---
## ⚙️ Key Features
1️. Data Augmentation
- To improve model generalization:
- Rotation
- Horizontal Flip
- Zoom
- Shear
- Width / Height Shift
Implemented using:
```text
ImageDataGenerator
```
2. Class Imbalance Handling
   
Medical datasets are often imbalanced.
This project uses class weighting to ensure the model learns minority classes effectively.
```text
sklearn.utils.class_weight
```
3. Transfer Learning
   
Instead of training from scratch, the model uses Xception pretrained weights to leverage learned visual features.

Benefits:
- Faster training
- Better accuracy
- Reduced data requirement
---
## 📂 Dataset
The project uses a Chest X-Ray Pneumonia dataset containing images divided into:
```text
train/
    NORMAL
    PNEUMONIA

validation/
    NORMAL
    PNEUMONIA

test/
    NORMAL
    PNEUMONIA
```
Each image is rescaled to improve training stability.

---
## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
---
## 📊 Training Pipeline
1. Load dataset using ImageDataGenerator
2. Apply data augmentation
3. Analyze class imbalance
4. Compute class weights
5. Load Xception pretrained model
6. Add custom classification layers
7. Compile with multiple evaluation metricsTrain the model on the training set
8. Evaluate on validation and test sets
---
## 📈 Evaluation Metrics
The model is evaluated using multiple metrics important for medical diagnosis systems:
| Metric    | Purpose                             |
| --------- | ----------------------------------- |
| Accuracy  | Overall classification performance  |
| AUC       | Model's ability to separate classes |
| Recall    | Detecting pneumonia cases correctly |
| Precision | Reducing false positives            |
---
## 🚀 How to Run the Project
1. Clone the repository
```Bash
git clone https://github.com/Subhajit14mandal/chest-xray-pneumonia-detection-deep-learning.git
cd chest-xray-pneumonia-detection-deep-learning
```
2. Install dependencies
```
pip install tensorflow scikit-learn matplotlib numpy
```
3. Run the notebook
Open the notebook in Google Colab or Jupyter Notebook:
```
Medical Image Classification for Disease Detection.ipynb
```
---
## 📷 Sample Workflow
```
Chest X-ray Image
        ↓
Preprocessing & Augmentation
        ↓
Xception Feature Extraction
        ↓
Fully Connected Layers
        ↓
Binary Classification
        ↓
Normal / Pneumonia
```
---
## 📌 Future Improvements
Implement Grad-CAM for model explainability
- Deploy model using Streamlit
- Convert model to TensorFlow Lite for mobile inference
- Train on larger medical datasets
---
## 👨‍💻 Author
Subhajit Mandal
Machine Learning Engineer (Aspiring)
- Focus: Deep Learning | Computer Vision | Medical AI
