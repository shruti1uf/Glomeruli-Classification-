# Glomeruli Classification Project

## Overview

This project aims to classify glomeruli patches from kidney biopsy images into two categories: globally sclerotic glomeruli (sclerotic) and non-globally sclerotic glomeruli (non-sclerotic). The dataset is derived from a large medical consortium, focusing on glomeruli detection and classification.

## Approach

### Dataset

The dataset consists of two folders:
- `globally_sclerotic_glomeruli`
- `non_globally_sclerotic_glomeruli`

Additionally, a `public.csv` file contains the names of each patch and their corresponding label.

### Model Selection

Several pre-trained models were evaluated based on their performance in terms of F1-score. The models tested include:
- AlexNet
- VGG19
- ResNet50
- GoogleNet

#### Performance of Different Models

- **AlexNet**: 
  - Test Accuracy: 0.9479
  - Precision: 0.8095
  - Recall: 0.9128
  - F1 Score: 0.8580

- **GoogleNet**:
  - Test Accuracy: 0.9792
  - Precision: 0.9645
  - Recall: 0.9128
  - F1 Score: 0.9379

- **VGG19**:
 - Training was attempted but the model took too long to train fully within the given constraints.
  
- **ResNet50**:
 - Training was attempted but the model took too long to train fully within the given constraints.

GoogleNet provided the best performance with the highest F1-score of apprpximately 94%, and accuracy of 98%. Therefore, it was chosen as the final model for this project. AlexNet was fast but the maximum F1-score and accuracy it had was 87% and 96% respectively.

### Data Preprocessing

Images were preprocessed using the following transformations:
- Resize to 224x224 pixels
- Convert to tensor
- Normalize using ImageNet mean and standard deviation

### Training and Validation

The dataset was split into training, validation, and test sets. The training process involved fine-tuning the last layer of the pre-trained GoogleNet model for binary classification. Various optimizers, schedulers, and loss functions were tested to achieve the best performance:

#### Optimizers

- **Adam**: Combines the advantages of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems. It computes adaptive learning rates for each parameter.
- **AdamW**: A variant of Adam that decouples weight decay from the gradient update, which can improve generalization.
- **SGD (Stochastic Gradient Descent)**: A basic optimizer that updates the parameters in the direction of the negative gradient. It is known for its simplicity and effectiveness, particularly in large datasets.
- **RMSProp**: An adaptive learning rate method designed to deal with the problems of Adagrad by using a moving average of squared gradients to normalize the gradient. **RMSProp provided the best results** in terms of model performance.

#### Schedulers

- **ReduceLROnPlateau**: Reduces the learning rate when a metric has stopped improving. This scheduler can help to converge faster when the learning rate becomes too large.
- **OneCycleLR**: A learning rate policy that anneals the learning rate from an initial value up to a maximum value and then back down to a minimum value. It helps in avoiding local minima and can result in better performance. **OneCycleLR provided the best results**.
- **StepLR**: Decays the learning rate by a factor every few epochs, which can be useful in reducing the learning rate systematically.

#### Loss Functions

- **BCEWithLogitsLoss**: Combines a Sigmoid layer and the binary cross-entropy loss in one single class. This is numerically more stable than using a plain Sigmoid followed by a binary cross-entropy loss. Since we are performing a binary classification, this loss function is suitable.

### Training Configuration

- **Optimizer**: RMSProp
- **Scheduler**: OneCycleLR
- **Loss Function**: BCEWithLogitsLoss

### Theoretical Justification for Component Choices

- **GoogleNet**: Chosen due to its efficiency and effectiveness in learning from a relatively small dataset while providing a high level of accuracy. GoogleNet, with its inception modules, balances computational efficiency and model performance, making it suitable for our task.

- **RMSProp**: RMSProp was chosen as it is an adaptive learning rate method designed to handle non-stationary objectives, and it worked best for this binary classification task. RMSProp adjusts the learning rate based on the average of recent gradients, which helps in converging quickly and effectively.

- **OneCycleLR**: OneCycleLR was used due to its ability to cyclically adjust the learning rate, which helps in finding a better minimum and avoids local minima. This scheduler adjusts the learning rate in a cyclical manner, which can lead to improved performance and faster convergence.

- **BCEWithLogitsLoss**: Combines a Sigmoid layer and the binary cross-entropy loss in one single class, making it more stable for binary classification tasks. This loss function is suitable for binary classification as it ensures numerical stability and efficient computation of gradients.

## ML Pipeline and its implementation.
- This repository contains the implementation of a machine learning pipeline for the classification of glomeruli as globally sclerotic or non-globally sclerotic using deep learning techniques.    - The pipeline consists of several essential steps, including data preprocessing, model training, and evaluation.

### Dataset and Preprocessing Procedure:
- The dataset consists of histopathological images of glomeruli labeled as globally sclerotic (1) or non-globally sclerotic (0). The images are organized into two sub-folders within the `Glomeruli_Classification` directory: `globally_sclerotic_glomeruli` and `non_globally_sclerotic_glomeruli`. Additionally, image annotations are provided in the public.csv file, containing image names and their corresponding labels.

#### Data Preprocessing Steps:
- **Data Loading**: The annotations from the public.csv file are read into a Pandas DataFrame to facilitate further processing.
- **Image Augmentation**: Image augmentation techniques such as random horizontal and vertical flips, random rotation, color jitter, and random grayscale are applied to increase the diversity of the training dataset and enhance model generalization.
- **Image Normalization**: Images are resized to 224x224 pixels and normalized using mean and standard deviation values of [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] respectively, as recommended for models pretrained on ImageNet.
- **Dataset Splitting**: The dataset is split into training (65%), validation (20%), and test sets (15%) using stratified splitting to preserve the class distribution in each set.

### Training and Validation Procedure:
- For this task, the GoogLeNet architecture, pretrained on ImageNet, is used as the base model. The final fully connected layer of the GoogLeNet model is modified to output binary classifications. The training process involves optimizing the model parameters using the RMSprop optimizer with a learning rate of 0.0001 and weight decay of 1e-4.

#### Training Loop Steps:
- **Data Loading**: Training and validation datasets are loaded using DataLoader instances, which provide efficient batching and data shuffling.
- **Model Initialization**: The modified GoogLeNet model is initialized and moved to the appropriate device (GPU if available).
- **Loss Function**: Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss) is used as the loss function, suitable for binary classification tasks.
- **Learning Rate Scheduler**: A OneCycleLR scheduler is employed to adjust the learning rate dynamically during training, enhancing convergence and generalization.

### Evaluation Metrics:  
- The trained model is evaluated on a separate test set to assess its performance using various metrics, including accuracy, precision, recall, and F1 score. Additionally, visualizations such as confusion matrix and ROC curve are generated to provide insights into the model's behavior.
    - **Accuracy**: Ratio of correctly predicted observations to the total observations.
    - **Precision**: Ratio of true positive predictions to the total predicted positives.
    - **Recall**: Ratio of true positive predictions to the total actual positives.
    - **F1 Score**: Harmonic mean of precision and recall, indicating a balance between the two metrics.

## Performance Metrics
The model's performance is evaluated based on accuracy, precision, recall, and F1 score. These metrics provide a comprehensive understanding of the model's ability to classify sclerotic and non-sclerotic glomeruli accurately.

- **Test Accuracy**: 0.9792
- **Test Precision**: 0.9645
- **Test Recall**: 0.9128
- **Test F1 Score**: 0.9379

## Environment Setup

Ensure you have Conda installed. Clone the project repository, then create and activate the project environment using:

```bash
conda env create -f environment.yml
conda activate Gomeruli_Classification
```

## Model Training

To train the model, execute the following command:

```bash
python googlenet.py
```
## Model Evaluation

After training the model, you can evaluate it on a new set of images using the evaluation.py script. The script requires the path to a folder containing glomeruli image patches as input and outputs a CSV file with the model's predictions.
Run the script as follows:
```bash
python evaluation.py <path_to_image_folder> model/model.pth evaluation.csv
```
Replace <path_to_image_folder> with the path to your image folder.
