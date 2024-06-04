# Importing necessary liberaries for data manipulation, visualization, and deep learning.
import os # We use this module provide a way to interact with the operating system, it allows for file and directory manipulation, and other OS-level interactions.
import pandas as pd # We use this as its a powerful data manipulation, handling and analysis library for python.
import matplotlib.pyplot as plt # It's a plotting library and pyplot is a state-based interface.
import seaborn as sns # It is a statistical data visulization library based on matplotlib
import torch # This is the core library of pytorch, which is a deep learning framework. It provides data structures for multi-dimensional tensors and operation on tensors.
import torch.nn as nn # It provides neural network layers and functions for building and training purposes.
import torch.optim as optim # It contains all the optimization algorithms that update the weights of the neural networks based on the gradients.
from torchvision import transforms, models # They are submodules of torchvision providing common image transformations for data augumentation and preprocessing, and containing pre-trained models for image classification and other tasks respectively.
from torch.utils.data import Dataset, DataLoader # These procides tool for dataloading and batching.
from torch.optim.lr_scheduler import OneCycleLR # We use it to adjust the learning rate during training.
from PIL import Image # This provides the function for opening, manupulating, and saving many different image file formats. 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Provides function to calculate common evaluation metrics.
from sklearn.model_selection import train_test_split # Provides tools for splitting the dataset into training, validation, and test sets.
from tqdm import tqdm # It creates progress bars in python, useful for tracking long-running operations.

# Setting the paths and Hyperparameters
data_dir = 'Glomeruli_Classification'  # Directory containing the dataset
output_dir = 'Plots'  # Directory for saving results
root_path = data_dir  # Root path for the images
learning_rate = 0.0001  # Learning rate for the model
batch_size = 46  # The size of batches for data loading
num_epochs = 25  # Number of training epochs

# Ensuring that the results directory exists 
os.makedirs(output_dir, exist_ok=True)

# Loading the data from the csv file
# We will use try except for data quality and flow-control
try: 
    data_df = pd.read_csv('public.csv') # Loading the csv file data into a dataframe
except Exception as e:
    print(f"Error reading CSV file: {e}") # Print an error if loading the CSV file fails
    exit()
    
# Printing the summary statistics of the data 
print(data_df.describe())
print("\n Catergory Distribution: ")
print(data_df['ground truth'].value_counts())
    
# Analyzing the image size distribution.
image_dimensions = {'height':[], 'width':[]} # Initializing dictionaries to store image dimensions
for idx in range(len(data_df)):
    img_label = data_df.iloc[idx, 1] 
    subdir = "C:/Users/shrut/Downloads/Glomeruli_Classification/non_globally_sclerotic_glomeruli" if img_label == 0 else "C:/Users/shrut/Downloads/Glomeruli_Classification/globally_sclerotic_glomeruli"
    img_name = data_df.iloc[idx, 0] 
    img_path = os.path.join(root_path, subdir, img_name) 
    try:
        with Image.open(img_path) as img:
            image_dimensions['height'].append(img.height)
            image_dimensions['width'].append(img.width)
    except FileNotFoundError:
        print(f"Warning: Image {img_name} not found. Skipping.")
        continue
        
# Preparing the dataset for training
class SclGloDataset(Dataset):
    def __init__(self, annotations_df, root_dir, transform=None):
        self.annotations = annotations_df # DataFrame with annotations
        self.root_dir = root_dir # Root directory containing the images
        self.transform = transform # Transformations that are to be applied on images
        
    def __len__(self):
        return len(self.annotations) # Returning the number of samples in the dataset
    
    def __getitem__(self, idx):
        img_label = self.annotations.iloc[idx,1] # Getting the image label
        subdir = "C:/Users/shrut/Downloads/Glomeruli_Classification/non_globally_sclerotic_glomeruli" if img_label == 0 else "C:/Users/shrut/Downloads/Glomeruli_Classification/globally_sclerotic_glomeruli"
        img_name = os.path.join(self.root_dir, subdir, self.annotations.iloc[idx, 0])
        # Performing this in try except to handle missing images.
        try: 
            image = Image.open(img_name).convert('HSV') # Opening the image and converting it into HSV
        except FileNotFoundError:
            print(f"Warning: Image {img_name} not found, hence skipping it.") 
            return None # Skipping the sample image if missing
        
        if self.transform:
            image = self.transform(image) # This to apply any transformations on image that are defined.
            return image, int(img_label) # Returning the image and the label.
        
        
# Data transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing images to 224x224
    transforms.RandomHorizontalFlip(),  # Applying horizontal flip augmentation
    transforms.RandomVerticalFlip(), # Applying vertical flip augmentation
    transforms.RandomRotation(30),  # Applying random rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # Applying Color jitter
    transforms.RandomGrayscale(p=0.2), # Applying randome grayscale with a probablity of 0.2
    transforms.ToTensor(),  # Converting the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing the images
])

# Creating the Dataset Instances
dataset = SclGloDataset(annotations_df=data_df, root_dir=root_path, transform=transform_pipeline)

# Splitting data into training, validation, and test sets
train_val_data, test_data = train_test_split(dataset, test_size=0.15, random_state=42)  # Using 15% for testing
train_data, val_data = train_test_split(train_val_data, test_size=0.20, random_state=42)  # Using 20% of remaining for validation

# Creating data loaders for training, validations, and test sets
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("Dataset preparation complete.")

# Define the model architecture (GoogLeNet)
googlenet_model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
googlenet_model.fc = nn.Linear(googlenet_model.fc.in_features, 1) # Modifying the final fully connected layer

# Moving the model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
googlenet_model = googlenet_model.to(device)

# Defining the loss function 
criterion = nn.BCEWithLogitsLoss() # We use it for binary classification tasks.
# This function is appropriate since we are performing a binary classification problem (globally sclerotic vs. non-globally sclerotic glomeruli).
# We use this as it combines the sigmoid layer and the BCELoss in one single class and it is numerically more stable.

# Defining the optimizer
optimizer = optim.RMSprop(googlenet_model.parameters(), lr=learning_rate, weight_decay=1e-4)
# RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm designed to handle the vanishing gradient problem.
# It adjusts the learning rate dynamically for each parameter, which helps in faster convergence.
# `learning_rate` is set to a small value (0.0001) to ensure gradual learning.
# `weight_decay` (1e-4) is a form of regularization that helps prevent overfitting by penalizing large weights.

# Defining the scheduler (OneCycleLR)
scheduler = OneCycleLR(optimizer,
                       max_lr=learning_rate,
                       total_steps=25, 
                       pct_start=0.3, # The percentage of the cycle spent increasing the learning rate
                       anneal_strategy='cos', # Cosine annealing strategy to decrease the learning rate
                       div_factor=10.0, # Initial learning rate is max_lr / div_factor
                       final_div_factor=100.0, # Minimum learning rate is max_lr / final_div_factor
                       cycle_momentum=True, # Adjust momentum during the cycle 
                       base_momentum=0.85,
                       max_momentum=0.95)
# `OneCycleLR` is a learning rate scheduler that adjusts the learning rate dynamically according to the 1cycle policy.
# This policy helps in achieving better performance and faster convergence.
# `max_lr` is set to the learning rate defined earlier, representing the peak learning rate.
# `total_steps` indicates the total number of training iterations.
# `pct_start` determines the percentage of the total steps where the learning rate increases from its initial value to the maximum value.
# `anneal_strategy` specifies the strategy to decrease the learning rate, with 'cos' indicating cosine annealing.
# `div_factor` and `final_div_factor` control the range of the learning rate, helping in smoother transitions.
# `cycle_momentum` toggles momentum adjustment; adjusting momentum can help in achieving smoother convergence.
# `base_momentum` and `max_momentum` define the range for momentum values.

# Training loop 
for epoch in range(num_epochs):
    # Training phase
    googlenet_model.train() # Setting the model to training mode, enabling features like batch normalization.
    running_loss = 0.0 # Variable to accumulate the loss over the epoch
    correct_preds = 0 # Variable to count correct predictions
    total_preds = 0 # Variable to count total predictions
    print(f"Epoch {epoch + 1}/{num_epochs} - Training started.") # Printing the start of the epoch
    # Looping through batches of training data
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
        if data is None:
            continue # Skipping the iteration if data is None
        inputs, labels = data # Unpacking inputs and labels from the batch
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1) # Moving inputs and labels to the appropriate device and reshaping labels
        optimizer.zero_grad() # Clearing the gradients from the previous step, so that more gradients can be loaded
        outputs = googlenet_model(inputs) # Forward pass through the model
        loss = criterion(outputs, labels) # Calculating the loss
        loss.backward() # Backward pass to calculate gradients
        optimizer.step() # Updating the model parameters
        
        running_loss += loss.item() * inputs.size(0) # Accumulating the loss
        predictions = torch.sigmoid(outputs) > 0.5 # Converting model outputs to binary predictions
        correct_preds += (predictions == labels).sum().item() # Counting correct predictions
        total_preds += labels.size(0) # Counting total predictions
    
    train_loss = running_loss / len(train_loader.dataset)  # Calculating average training loss
    train_accuracy = correct_preds / total_preds  # Calculating training accuracy
    print(f"Epoch {epoch + 1}/{num_epochs} - Training completed. Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Validation phase
    googlenet_model.eval() # Setting the model to evaluation mode, disabling features like batch normalization
    val_running_loss = 0.0 # Variable to accumulate validation loss
    val_correct_preds = 0 # Variable to count correct predictions in validation
    val_total_preds = 0 # Variable to count total predictions in validation
    val_preds = []  # List to store validation predictions
    val_labels = [] # List to store true validation labels
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation started.") # Printing the start of the validation phase

    with torch.no_grad():  # Disabling gradient computation for validation
        for batch_idx, data in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")):
            if data is None:
                continue # Skipping the iteration if data is None
            inputs, labels = data # Unpacking inputs and labels from the batch
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1) # Moving inputs and labels to the appropriate device and reshaping labels
            outputs = googlenet_model(inputs) # Forward pass through the model
            val_loss = criterion(outputs, labels) # Calculating the validation loss
            val_running_loss += val_loss.item() * inputs.size(0) # Accumulating the validation loss
            predictions = torch.sigmoid(outputs) > 0.5 # Converting model outputs to binary predictions
            val_preds.extend(predictions.cpu().numpy()) # Storing predictions for later evaluation
            val_labels.extend(labels.cpu().numpy())  # Storing true labels for later evaluation
            val_correct_preds += (predictions == labels).sum().item() # Counting correct predictions
            val_total_preds += labels.size(0) # Counting total predictions
    
    val_loss = val_running_loss / len(val_loader.dataset) # Calculating average validation loss
    val_accuracy = val_correct_preds / val_total_preds # Calculating validation accuracy

    print(f"Epoch {epoch + 1}/{num_epochs} - Validation completed. Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    # Stepping the scheduler
    scheduler.step() # Adjusting the learning rate according to the scheduler

print("Training Complete.")

# Evaluating on Test Set
googlenet_model.eval()  # Setting the model to evaluation mode, disabling features like dropout and batch normalization
test_running_loss = 0.0  # Variable to accumulate the test loss
test_correct_preds = 0  # Variable to count correct predictions
test_total_preds = 0  # Variable to count total predictions
test_preds = []  # List to store test predictions
test_labels = []  # List to store true test labels
print("Testing started.")  # Printing the start of the testing phase

# Disable gradient calculation for the test phase
with torch.no_grad():
    # Looping through batches of test data
    for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing")):
        if data is None:
            continue  # Skipping the iteration if data is None
        inputs, labels = data  # Unpacking inputs and labels from the batch
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # Moving inputs and labels to the appropriate device and reshaping labels
        outputs = googlenet_model(inputs)  # Forward pass through the model
        test_loss = criterion(outputs, labels)  # Calculating the test loss
        test_running_loss += test_loss.item() * inputs.size(0)  # Accumulating the test loss
        predictions = torch.sigmoid(outputs) > 0.5  # Converting model outputs to binary predictions
        test_preds.extend(predictions.cpu().numpy())  # Storing predictions for later evaluation
        test_labels.extend(labels.cpu().numpy())  # Storing true labels for later evaluation
        test_correct_preds += (predictions == labels).sum().item()  # Counting correct predictions
        test_total_preds += labels.size(0)  # Counting total predictions

# Calculating average test loss
test_loss = test_running_loss / len(test_loader.dataset)
# Calculating test accuracy
test_accuracy = test_correct_preds / test_total_preds
# Calculating test F1 score
test_f1 = f1_score(test_labels, test_preds)

# Calculating additional metrics

# Accuracy: Ratio of correctly predicted observations to the total observations
accuracy = accuracy_score(test_labels, test_preds)

# Precision: Ratio of true positive predictions to the total predicted positives
precision = precision_score(test_labels, test_preds, average='binary')

# Recall: Ratio of true positive predictions to the total actual positives
recall = recall_score(test_labels, test_preds, average='binary')

# F1 Score: Harmonic mean of precision and recall, indicating a balance between precision and recall
f1 = f1_score(test_labels, test_preds, average='binary')

# Printing the metrics
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')
print("Testing completed.")  # Indicating that the testing process is complete

# Saving the model
model_save_path = 'googlenet_model.pth'  # Path to save the model
torch.save(googlenet_model.state_dict(), model_save_path)  # Saving the model's state dictionary
print(f"Model saved to {model_save_path}")  # Printing the save confirmation


# Plotting the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, test_preds)
# The confusion matrix is a table that shows the number of correct and incorrect predictions made by a classification model.
# Rows represent the actual classes, and columns represent the predicted classes.

plt.figure(figsize=(9, 6)) # creates a new figure for the plot with a width of 9 and height of 6.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False) # Visualizing as a heatmap, 'annot=true' - displays the actual values in each cell of the heatmap, 'fmt='d'' - specifies the format code for annotation here its integers, 'cmap='Blues'' - defines color of the map, and 'cbar=False' - hides the colobar legend since the values are already displayed.
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.show()

# Plotting the ROC Curve and AUC
from sklearn.metrics import roc_curve, roc_auc_score

# Calculating ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_preds)
roc_auc = roc_auc_score(test_labels, test_preds)

# roc_curve(test_labels, test_preds) calculates the ROC curve.
# The ROC curve is a graphical tool used to evaluate the performance of binary classification models.
# It plots the True Positive Rate (TPR) on the y-axis against the False Positive Rate (FPR) on the x-axis for different classification thresholds.
# - fpr: False Positive Rate (FPR) at various thresholds.
# - tpr: True Positive Rate (TPR) at various thresholds.
# - _: Discarded threshold values used to calculate FPR and TPR.

# roc_auc_score(test_labels, test_preds) calculates the Area Under the ROC Curve (AUC).
# AUC is a single numerical metric that summarizes the performance of the ROC curve.

# Plot ROC curve
plt.figure(figsize=(9, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}') # plotting the actual ROC curve with a blue line and label including the calculated AUC score.
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') # plotting a dashed gray line representing a perfect classifier (AUC = 1).
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.show()

# Breakdown on how to interpret the plots.
# 1. A good classifier will have an ROC curve that stays close to the top-left corner of the graph, indicating high TPR (correctly classifying positive cases) and low FPR (incorrectly classifying negative cases).
# 2. An AUC of 1 indicates perfect performance, while an AUC of 0.5 represents a random guess. It represents the probability that the model will rank a randomly chosen positive instance higher than a randomly chosen negative instance.
# 3. TPR, also known as recall or sensitivity, is the proportion of actual positive cases that the model correctly identifies. In other words, it tells you how good the model is at finding true positives.
# 4. FPR is the proportion of actual negative cases that the model incorrectly classifies as positive. In other words, it tells you how often the model makes a mistake by identifying a negative case as positive.

"""
        Output:
            
        ground truth
 count   5758.000000
 mean       0.183050
 std        0.386741
 min        0.000000
 25%        0.000000
 50%        0.000000
 75%        0.000000
 max        1.000000

  Catergory Distribution: 
 ground truth
 0    4704
 1    1054
 Name: count, dtype: int64
 Dataset preparation complete.
 Epoch 1/25 - Training started.
 Training Epoch 1/25: 100%|██████████| 86/86 [09:10<00:00,  6.40s/it]
 Epoch 1/25 - Training completed. Loss: 0.2272, Accuracy: 0.9055
 Epoch 1/25 - Validation started.
 Validation Epoch 1/25: 100%|██████████| 22/22 [00:51<00:00,  2.35s/it]
 Epoch 1/25 - Validation completed. Loss: 0.1417, Accuracy: 0.9581
 Epoch 2/25 - Training started.
 Training Epoch 2/25: 100%|██████████| 86/86 [06:43<00:00,  4.70s/it]
 Epoch 2/25 - Training completed. Loss: 0.0947, Accuracy: 0.9650
 Epoch 2/25 - Validation started.
 Validation Epoch 2/25: 100%|██████████| 22/22 [00:47<00:00,  2.16s/it]
 Epoch 2/25 - Validation completed. Loss: 0.0937, Accuracy: 0.9694
 Epoch 3/25 - Training started.
 Training Epoch 3/25: 100%|██████████| 86/86 [06:39<00:00,  4.65s/it]
 Epoch 3/25 - Training completed. Loss: 0.0625, Accuracy: 0.9773
 Epoch 3/25 - Validation started.
 Validation Epoch 3/25: 100%|██████████| 22/22 [00:47<00:00,  2.18s/it]
 Epoch 3/25 - Validation completed. Loss: 0.1238, Accuracy: 0.9591
 Epoch 4/25 - Training started.
 Training Epoch 4/25: 100%|██████████| 86/86 [06:41<00:00,  4.67s/it]
 Epoch 4/25 - Training completed. Loss: 0.0498, Accuracy: 0.9837
 Epoch 4/25 - Validation started.
 Validation Epoch 4/25: 100%|██████████| 22/22 [00:47<00:00,  2.18s/it]
 Epoch 4/25 - Validation completed. Loss: 0.2568, Accuracy: 0.9285
 Epoch 5/25 - Training started.
 Training Epoch 5/25: 100%|██████████| 86/86 [05:48<00:00,  4.05s/it]
 Epoch 5/25 - Training completed. Loss: 0.0679, Accuracy: 0.9750
 Epoch 5/25 - Validation started.
 Validation Epoch 5/25: 100%|██████████| 22/22 [00:27<00:00,  1.23s/it]
 Epoch 5/25 - Validation completed. Loss: 0.1156, Accuracy: 0.9653
 Epoch 6/25 - Training started.
 Training Epoch 6/25: 100%|██████████| 86/86 [04:20<00:00,  3.03s/it]
 Epoch 6/25 - Training completed. Loss: 0.0562, Accuracy: 0.9798
 Epoch 6/25 - Validation started.
 Validation Epoch 6/25: 100%|██████████| 22/22 [00:27<00:00,  1.24s/it]
 Epoch 6/25 - Validation completed. Loss: 0.0813, Accuracy: 0.9796
 Epoch 7/25 - Training started.
 Training Epoch 7/25: 100%|██████████| 86/86 [04:10<00:00,  2.91s/it]
 Epoch 7/25 - Training completed. Loss: 0.0380, Accuracy: 0.9875
 Epoch 7/25 - Validation started.
 Validation Epoch 7/25: 100%|██████████| 22/22 [00:27<00:00,  1.24s/it]
 Epoch 7/25 - Validation completed. Loss: 0.2754, Accuracy: 0.9428
 Epoch 8/25 - Training started.
 Training Epoch 8/25: 100%|██████████| 86/86 [04:10<00:00,  2.91s/it]
 Epoch 8/25 - Training completed. Loss: 0.0674, Accuracy: 0.9752
 Epoch 8/25 - Validation started.
 Validation Epoch 8/25: 100%|██████████| 22/22 [00:27<00:00,  1.25s/it]
 Epoch 8/25 - Validation completed. Loss: 0.1140, Accuracy: 0.9632
 Epoch 9/25 - Training started.
 Training Epoch 9/25: 100%|██████████| 86/86 [04:11<00:00,  2.92s/it]
 Epoch 9/25 - Training completed. Loss: 0.0293, Accuracy: 0.9888
 Epoch 9/25 - Validation started.
 Validation Epoch 9/25: 100%|██████████| 22/22 [00:27<00:00,  1.25s/it]
 Epoch 9/25 - Validation completed. Loss: 0.1184, Accuracy: 0.9673
 Epoch 10/25 - Training started.
 Training Epoch 10/25: 100%|██████████| 86/86 [04:11<00:00,  2.93s/it]
 Epoch 10/25 - Training completed. Loss: 0.0478, Accuracy: 0.9829
 Epoch 10/25 - Validation started.
 Validation Epoch 10/25: 100%|██████████| 22/22 [00:27<00:00,  1.25s/it]
 Epoch 10/25 - Validation completed. Loss: 0.1190, Accuracy: 0.9632
 Epoch 11/25 - Training started.
 Training Epoch 11/25: 100%|██████████| 86/86 [04:11<00:00,  2.93s/it]
 Epoch 11/25 - Training completed. Loss: 0.0395, Accuracy: 0.9849
 Epoch 11/25 - Validation started.
 Validation Epoch 11/25: 100%|██████████| 22/22 [00:27<00:00,  1.24s/it]
 Epoch 11/25 - Validation completed. Loss: 0.1303, Accuracy: 0.9602
 Epoch 12/25 - Training started.
 Training Epoch 12/25: 100%|██████████| 86/86 [04:15<00:00,  2.97s/it]
 Epoch 12/25 - Training completed. Loss: 0.0410, Accuracy: 0.9842
 Epoch 12/25 - Validation started.
 Validation Epoch 12/25: 100%|██████████| 22/22 [00:41<00:00,  1.90s/it]
 Epoch 12/25 - Validation completed. Loss: 0.1059, Accuracy: 0.9622
 Epoch 13/25 - Training started.
 Training Epoch 13/25: 100%|██████████| 86/86 [04:12<00:00,  2.93s/it]
 Epoch 13/25 - Training completed. Loss: 0.0480, Accuracy: 0.9844
 Epoch 13/25 - Validation started.
 Validation Epoch 13/25: 100%|██████████| 22/22 [00:27<00:00,  1.25s/it]
 Epoch 13/25 - Validation completed. Loss: 0.0934, Accuracy: 0.9704
 Epoch 14/25 - Training started.
 Training Epoch 14/25: 100%|██████████| 86/86 [04:12<00:00,  2.93s/it]
 Epoch 14/25 - Training completed. Loss: 0.0088, Accuracy: 0.9969
 Epoch 14/25 - Validation started.
 Validation Epoch 14/25: 100%|██████████| 22/22 [00:27<00:00,  1.25s/it]
 Epoch 14/25 - Validation completed. Loss: 0.1202, Accuracy: 0.9632
 Epoch 15/25 - Training started.
 Training Epoch 15/25: 100%|██████████| 86/86 [04:12<00:00,  2.93s/it]
 Epoch 15/25 - Training completed. Loss: 0.0165, Accuracy: 0.9951
 Epoch 15/25 - Validation started.
 Validation Epoch 15/25: 100%|██████████| 22/22 [00:27<00:00,  1.25s/it]
 Epoch 15/25 - Validation completed. Loss: 0.2277, Accuracy: 0.9571
 Epoch 16/25 - Training started.
 Training Epoch 16/25: 100%|██████████| 86/86 [04:12<00:00,  2.93s/it]
 Epoch 16/25 - Training completed. Loss: 0.0343, Accuracy: 0.9875
 Epoch 16/25 - Validation started.
 Validation Epoch 16/25: 100%|██████████| 22/22 [00:27<00:00,  1.25s/it]
 Epoch 16/25 - Validation completed. Loss: 0.1019, Accuracy: 0.9704
 Epoch 17/25 - Training started.
 Training Epoch 17/25: 100%|██████████| 86/86 [04:12<00:00,  2.93s/it]
 Epoch 17/25 - Training completed. Loss: 0.0074, Accuracy: 0.9980
 Epoch 17/25 - Validation started.
 Validation Epoch 17/25: 100%|██████████| 22/22 [00:27<00:00,  1.25s/it]
 Epoch 17/25 - Validation completed. Loss: 0.0938, Accuracy: 0.9724
 Epoch 18/25 - Training started.
 Training Epoch 18/25: 100%|██████████| 86/86 [08:55<00:00,  6.23s/it]
 Epoch 18/25 - Training completed. Loss: 0.0021, Accuracy: 0.9997
 Epoch 18/25 - Validation started.
 Validation Epoch 18/25: 100%|██████████| 22/22 [01:35<00:00,  4.32s/it]
 Epoch 18/25 - Validation completed. Loss: 0.1072, Accuracy: 0.9694
 Epoch 19/25 - Training started.
 Training Epoch 19/25: 100%|██████████| 86/86 [09:37<00:00,  6.72s/it]
 Epoch 19/25 - Training completed. Loss: 0.0243, Accuracy: 0.9918
 Epoch 19/25 - Validation started.
 Validation Epoch 19/25: 100%|██████████| 22/22 [00:44<00:00,  2.03s/it]
 Epoch 19/25 - Validation completed. Loss: 0.1094, Accuracy: 0.9653
 Epoch 20/25 - Training started.
 Training Epoch 20/25: 100%|██████████| 86/86 [06:40<00:00,  4.65s/it]
 Epoch 20/25 - Training completed. Loss: 0.0105, Accuracy: 0.9980
 Epoch 20/25 - Validation started.
 Validation Epoch 20/25: 100%|██████████| 22/22 [00:43<00:00,  1.99s/it]
 Epoch 20/25 - Validation completed. Loss: 0.0880, Accuracy: 0.9704
 Epoch 21/25 - Training started.
 Training Epoch 21/25: 100%|██████████| 86/86 [06:40<00:00,  4.66s/it]
 Epoch 21/25 - Training completed. Loss: 0.0036, Accuracy: 0.9992
 Epoch 21/25 - Validation started.
 Validation Epoch 21/25: 100%|██████████| 22/22 [00:44<00:00,  2.02s/it]
 Epoch 21/25 - Validation completed. Loss: 0.0860, Accuracy: 0.9745
 Epoch 22/25 - Training started.
 Training Epoch 22/25: 100%|██████████| 86/86 [06:39<00:00,  4.64s/it]
 Epoch 22/25 - Training completed. Loss: 0.0019, Accuracy: 0.9995
 Epoch 22/25 - Validation started.
 Validation Epoch 22/25: 100%|██████████| 22/22 [00:44<00:00,  2.01s/it]
 Epoch 22/25 - Validation completed. Loss: 0.1049, Accuracy: 0.9714
 Epoch 23/25 - Training started.
 Training Epoch 23/25: 100%|██████████| 86/86 [05:42<00:00,  3.98s/it]
 Epoch 23/25 - Training completed. Loss: 0.0075, Accuracy: 0.9995
 Epoch 23/25 - Validation started.
 Validation Epoch 23/25: 100%|██████████| 22/22 [00:26<00:00,  1.23s/it]
 Epoch 23/25 - Validation completed. Loss: 0.1277, Accuracy: 0.9653
 Epoch 24/25 - Training started.
 Training Epoch 24/25: 100%|██████████| 86/86 [06:35<00:00,  4.60s/it]
 Epoch 24/25 - Training completed. Loss: 0.0007, Accuracy: 1.0000
 Epoch 24/25 - Validation started.
 Validation Epoch 24/25: 100%|██████████| 22/22 [00:44<00:00,  2.02s/it]
 Epoch 24/25 - Validation completed. Loss: 0.1166, Accuracy: 0.9683
 Epoch 25/25 - Training started.
 Training Epoch 25/25: 100%|██████████| 86/86 [06:46<00:00,  4.73s/it]
 Epoch 25/25 - Training completed. Loss: 0.0009, Accuracy: 1.0000
 Epoch 25/25 - Validation started.
 Validation Epoch 25/25: 100%|██████████| 22/22 [00:44<00:00,  2.02s/it]
 Epoch 25/25 - Validation completed. Loss: 0.1274, Accuracy: 0.9653
 Training Complete.
 Testing started.
 Testing: 100%|██████████| 19/19 [00:49<00:00,  2.63s/it]
 Test Accuracy: 0.9792
 Test Precision: 0.9645
 Test Recall: 0.9128
 Test F1 Score: 0.9379
 Testing completed.
 Model saved to googlenet_model.pth

  
 Important
 Figures are displayed in the Plots pane by default. To make them also appear inline in the console, you need to uncheck "Mute inline plotting" under the options menu of Plots.
 
 """