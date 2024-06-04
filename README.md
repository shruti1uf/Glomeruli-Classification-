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

## Theoretical Justification for Component Choices

### Model Selection

- **GoogleNet**: Chosen due to its efficiency and effectiveness in learning from a relatively small dataset while providing a high level of accuracy. GoogleNet, with its inception modules, balances computational efficiency and model performance, making it suitable for our task.

### Optimizer Selection

- **RMSProp**: RMSProp was chosen as it is an adaptive learning rate method designed to handle non-stationary objectives, and it worked best for this binary classification task. RMSProp adjusts the learning rate based on the average of recent gradients, which helps in converging quickly and effectively.

### Scheduler Selection

- **OneCycleLR**: OneCycleLR was used due to its ability to cyclically adjust the learning rate, which helps in finding a better minimum and avoids local minima. This scheduler adjusts the learning rate in a cyclical manner, which can lead to improved performance and faster convergence.

### Loss Function

- **BCEWithLogitsLoss**: Combines a Sigmoid layer and the binary cross-entropy loss in one single class, making it more stable for binary classification tasks. This loss function is suitable for binary classification as it ensures numerical stability and efficient computation of gradients.




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
conda activate glomeruli_classification
