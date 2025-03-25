# Face Mask Classification and Segmentation

## i. Introduction
This project aims to develop a computer vision solution to classify and segment face masks in images. The approach includes both traditional machine learning techniques using handcrafted features and deep learning-based methods like Convolutional Neural Networks (CNN) and U-Net.

## ii. Dataset
- **Face Mask Detection Dataset**: Contains images of people with and without face masks. Available at: [Face-Mask-Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
- **Masked Face Segmentation Dataset**: Includes ground truth face mask segmentations. Available at: [MFSD Dataset](https://github.com/sadjadrz/MFSD)

## iii. Methodology
### a) Binary Classification Using Handcrafted Features and ML Classifiers
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

### b) Binary Classification Using CNN
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

### c) Region Segmentation Using Traditional Techniques
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

### d) Mask Segmentation Using U-Net
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
### a) Binary Classification Using Handcrafted Features and ML Classifiers
| Model                 | Accuracy | F1-Score |
|-----------------------|----------|----------|
| Support Vector Machine (SVM) | 0.933      | 0.940      |
| Logistic Regression  | 0.892      | 0.902      |
| Neural Network (MLP) | 0.930      | 0.937      |

### b) Binary Classification Using CNN
| Learning Rate | Batch Size | Optimizer | Activation | Accuracy |
|--------------|------------|-----------|------------|----------|
| 0.01        | 32         | SGD      | ReLU       | 0.969      |
| 0.01       | 32          | SGD      | Tanh       | 0.973      |
| 0.01       | 64          | SGD      | ReLU       | 0.971      |
| 0.01       | 64          | SGD      | Tanh       | 0.971      |
| 0.001      | 32          | ADAM      | ReLU       | 0.974      |
| 0.001      | 64          | ADAM      | ReLU       | 0.981      |


### c) Region Segmentation Results
| Metric | Value |
|--------|-------|
| Average Dice Coefficient | 0.4608 |

### d) Mask Segmentation Using U-Net Results
| Metric | Value |
|--------|-------|
| Average Dice Coefficient | 0.9531 |

## vi. Observations and Analysis

### a) Binary Classification Using Handcrafted Features and ML Classifiers
Using **SIFT** for feature extraction resulted in **lower classification accuracy** than **HOG**. This is because SIFT generates **high-dimensional feature vectors** for each keypoint, which increases the complexity of the input data. This added complexity makes it harder for traditional classifiers to generalize effectively, often leading to **overfitting**.

Additionally, **SIFT focuses on local keypoints**, which may cause it to miss **important global patterns** that HOG captures. HOG extracts **gradient orientations over the entire image**, providing a structured representation of shapes and textures. This allows classifiers to **better distinguish between objects**, leading to improved classification performance.

### b) Binary Classification Using CNN

We trained **36 different CNN models** using all possible combinations of hyperparameters, including **learning rates (0.01, 0.001), batch sizes (32, 64), optimizers (Adam, SGD, RMSprop), and activation functions (ReLU, Sigmoid, Tanh)**.  

From these experiments, we observed significant variations in performance across different configurations. Some models struggled to converge, while others performed exceptionally well. The **top-performing models** are listed in the results table above.  

Key observations from the training process:  
- **ReLU** consistently outperformed **Sigmoid** and **Tanh**, likely due to its ability to mitigate vanishing gradient issues.  
- **Adam** showed the most stable convergence across different learning rates and batch sizes, while **SGD with momentum** performed well but was sensitive to the learning rate.  
- **Higher batch sizes (64)** led to smoother training curves but, in some cases, resulted in slightly lower final accuracy due to reduced updates per epoch.  
- **Models trained with a learning rate of 0.001** performed better than those with 0.01, as a higher learning rate often caused instability.  

These insights highlight the **importance of hyperparameter tuning** in CNN training, demonstrating that minor changes in configuration can significantly impact classification accuracy. The inconsistencies across the various combinations can be observed in the accompanying notebook.

### c) Region Segmentation Using Traditional Techniques
During the segmentation process, we encountered a challenge where the background of the face crop images varied in intensity—sometimes appearing darker than the face mask and other times lighter. The appropriate thresholding method depends on this variation: **cv2.THRESH_BINARY** is suitable when the background is lighter, while **cv2.THRESH_BINARY_INV** is required when the background is darker. 

To address this, we calculate the **average intensity** of the image. If the average intensity exceeds **127**, the background is classified as **light**; otherwise, it is classified as **dark**. Based on this classification, the appropriate thresholding technique is selected to ensure accurate segmentation.

Using this segmentation process, the **Dice coefficient score achieved was 46.08%**. To improve this accuracy, we implemented **U-Net**, a deep learning-based segmentation model. The process of training and evaluating U-Net is detailed below.

## vii. How to Run the Code
Run the Python notebooks, or view the results that are visible from our runs.

## Contributors
- IMT2022100 : Tahir Khadarabad
- IMT2022505 : Anirudh Pathaneni
- IMT2022545 : Chaitanya Tadikonda

