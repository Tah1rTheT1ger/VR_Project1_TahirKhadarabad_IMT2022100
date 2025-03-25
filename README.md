# Face Mask Classification and Segmentation

## i. Introduction
This project aims to develop a computer vision solution to classify and segment face masks in images. The approach includes both traditional machine learning techniques using handcrafted features and deep learning-based methods like Convolutional Neural Networks (CNN) and U-Net.

## ii. Dataset
- **Face Mask Detection Dataset**: Contains images of people with and without face masks. Available at: [Face-Mask-Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
- **Masked Face Segmentation Dataset**: Includes ground truth face mask segmentations. Available at: [MFSD Dataset](https://github.com/sadjadrz/MFSD)

## iii. Methodology
### Binary Classification Using Handcrafted Features and ML Classifiers
1. **Feature Extraction**:
   - Extracted **Histogram of Oriented Gradients (HOG)** features for texture analysis.
   - Extracted **Scale-Invariant Feature Transform (SIFT)** features to capture key points and descriptors.
2. **Data Preprocessing**:
   - Converted images to grayscale and resized them to 128x128.
   - Removed duplicate images to ensure uniqueness.
3. **Training Machine Learning Models**:
   - Used three classifiers:
     - **Support Vector Machine (SVM)** with RBF kernel.
     - **Logistic Regression** with L1 regularization.
     - **Neural Network (MLPClassifier)** with three hidden layers (256, 128, 64).
4. **Evaluation**:
   - Compared models based on **Accuracy** and **F1-Score**.
   - Normalized feature data using **StandardScaler** before training models.

### Binary Classification Using CNN
1. **Dataset Preparation**:
   - Images were split into **80% training** and **20% testing** with no overlap.
   - Applied **data augmentation** (resizing, horizontal flip, rotation) to improve generalization.
2. **CNN Model Architecture**:
   - Three convolutional layers with increasing filters (32, 64, 128) and **ReLU activation**.
   - Max-pooling layers for downsampling.
   - Fully connected layers with final sigmoid activation for binary classification.
3. **Training Process**:
   - Used **Binary Cross-Entropy Loss (BCELoss)**.
   - Tested different optimizers (**Adam, SGD, RMSprop**) and learning rates.
   - Trained for 15 epochs with accuracy tracking.
4. **Evaluation**:
   - Tracked **training accuracy** over epochs.
   - Compared different hyperparameter combinations.

### Region Segmentation Using Traditional Techniques
1. **Image Preprocessing**:
   - Converted face images to grayscale.
   - Applied Gaussian Blur for noise reduction.
   - Used Otsu’s thresholding to create binary masks.
   - Applied Canny edge detection to refine segmentation.
2. **Morphological Processing**:
   - Used **morphological closing** to enhance mask contours.
3. **Dice Coefficient Evaluation**:
   - Compared predicted segmentation masks with ground truth masks.
   - Computed Dice Coefficient to measure segmentation accuracy.

### Mask Segmentation Using U-Net
1. **Dataset Preparation**:
   - Images and corresponding segmentation masks were resized to 128x128.
   - Normalized pixel values to [0, 1] range.
2. **U-Net Model Architecture**:
   - Encoder-decoder structure with skip connections.
   - Convolutional layers with batch normalization and ReLU activation.
   - Transposed convolution layers for upsampling.
3. **Training Process**:
   - Loss function: **Dice Loss**.
   - Optimizer: **Adam**.
   - Evaluation metric: **Dice Coefficient**.
   - Early stopping based on validation loss.
4. **Evaluation**:
   - Predicted segmentation masks were compared with ground truth.
   - Performance measured using the Dice Coefficient.

## iv. Hyperparameters and Experiments
### Machine Learning Classifiers
- **SVM**:
  - Kernel: RBF
  - C: 2
  - Gamma: scale
- **Logistic Regression**:
  - Solver: liblinear
  - Max Iterations: 500
  - Regularization Parameter (C): 0.5
- **Neural Network (MLPClassifier)**:
  - Hidden Layers: (256, 128, 64)
  - Activation Function: ReLU
  - Solver: Adam
  - Max Iterations: 500

### CNN Hyperparameter Variations
- **Learning Rates**: 0.01, 0.001
- **Batch Sizes**: 32, 64
- **Optimizers**: Adam, SGD, RMSprop
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Epochs**: 15

## v. Results
### Binary Classification Using Handcrafted Features and ML Classifiers
| Model                 | Accuracy | F1-Score |
|-----------------------|----------|----------|
| Support Vector Machine (SVM) | TBD      | TBD      |
| Logistic Regression  | TBD      | TBD      |
| Neural Network (MLP) | TBD      | TBD      |

### Binary Classification Using CNN
| Learning Rate | Batch Size | Optimizer | Activation | Accuracy |
|--------------|------------|-----------|------------|----------|
| 0.01        | 32         | Adam      | ReLU       | TBD      |
| 0.001       | 64         | SGD       | Sigmoid    | TBD      |
| 0.001       | 32         | RMSprop   | Tanh       | TBD      |

### Region Segmentation Results
| Metric | Value |
|--------|-------|
| Average Dice Coefficient | TBD |

### Mask Segmentation Using U-Net Results
| Metric | Value |
|--------|-------|
| Average Dice Coefficient | TBD |

## vi. Observations and Analysis
(Details will be added as the code is provided)

## vii. How to Run the Code
Run the Python notebooks, or view the results that are visible from our runs.

## Contributors
IMT2022100 - Tahir Khadarabad
IMT2022505 - Anirudh Pathaneni
IMT2022545 - Chaitanya Tadikonda

