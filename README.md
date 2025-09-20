# Bone Fracture Detection from X-Ray Images

A deep learning project for automated classification of bone fractures in X-ray images using Convolutional Neural Networks (CNNs).

## 📋 Project Overview

This project addresses the critical need for accurate and timely diagnosis of bone fractures in medical imaging. Traditional methods relying on expert radiologists can lead to diagnostic delays and potential human error. Our solution leverages deep learning to automatically classify X-ray images as either **'Fractured'** or **'Non-Fractured'**, providing a foundation for a powerful diagnostic aid tool.

**Dataset Source**: [Kaggle - X-ray Images of Fractured and Healthy Bones](https://www.kaggle.com/datasets/foyez767/x-ray-images-of-fractured-and-healthy-bones)

## 🎯 Key Features

- **Binary Classification**: Distinguishes between fractured and non-fractured bones
- **High Accuracy**: Achieves 98% test accuracy with optimized CNN architecture
- **Medical-Optimized Design**: Prioritizes fracture detection sensitivity while minimizing false alarms
- **Robust Preprocessing**: Includes data augmentation techniques (random flips, rotation, color jitter)
- **Clinical Relevance**: Designed specifically for medical diagnostic applications

## 📊 Dataset Characteristics

### Fractured Bone Observations:
- Discontinuity in bone structure
- Misalignment/displacement at fracture sites
- Sharp edges and irregular fracture lines
- Bone fragmentation in severe cases
- Variety of bone types (hands, wrists, arms, legs, ankles)

### Non-Fractured Bone Observations:
- Smooth and continuous bone structures
- Absence of irregular fracture lines
- Anatomical variety across different body parts
- Presence of annotations and markers (potential bias sources)

## 🏗️ Model Architecture

The project implements a specialized CNN with the following structure:

```
Input Layer (64x64 RGB images) → 
Conv2D (64 filters) + BatchNormalization + MaxPooling2D → 
Conv2D (128 filters) + BatchNormalization + MaxPooling2D → 
Conv2D (256 filters) + BatchNormalization + MaxPooling2D → 
GlobalAveragePooling2D → 
Dense (256 units) + BatchNormalization + Dropout → 
Output Layer (Softmax)
```

### Key Design Choices:
- Progressive filter increase (32→64→128→256) for feature extraction
- Strategic regularization with L2 regularization and dropout
- Recall-focused training with class weighting (fracture: 1.5)
- Medical-appropriate data augmentation

## 📈 Performance Results

### Training Metrics:
- **Training Accuracy**: 98.36%
- **Validation Accuracy**: 97.85%
- **Fractured Recall**: 98%
- **Non-Fractured Precision**: 97%

### Validation Metrics:
- **Fractured Precision**: 84%
- **Fractured Recall**: 97%
- **Non-Fractured Precision**: 97%
- **Non-Fractured Recall**: 82%

The model demonstrates excellent generalization with only 0.51% difference between training and validation accuracy.

## 💡 Key Insights

### Strengths:
- High test accuracy (98%) with balanced precision-recall
- Excellent fracture detection sensitivity (98% recall)
- Minimal false alarms for fractured cases (99% precision)
- Good generalization across datasets

### Limitations:
- Slightly lower recall for non-fractured cases (82%)
- Computationally intensive architecture
- Limited to binary classification (no fracture type detection)

## 🔮 Future Improvements

- Threshold tuning to further improve precision
- Advanced data augmentation and balancing techniques
- Explainability methods (Grad-CAM) for model interpretability
- Multi-class classification for different fracture types
- Model optimization for computational efficiency
- Integration with medical imaging systems


## 🙏 Acknowledgments

- Kaggle for providing the dataset
- TensorFlow and Keras communities for excellent documentation
- Medical imaging researchers whose work inspired this project

---

**Disclaimer**: This project is for educational and research purposes only. It should not be used as a sole diagnostic tool in clinical settings without proper validation and regulatory approval.
