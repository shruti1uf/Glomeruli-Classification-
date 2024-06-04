# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:56:39 2024

@author: shrut
"""
import os # We use this module provide a way to interact with the operating system, it allows for file and directory manipulation, and other OS-level interactions.
import sys # It provides functions and variables which are used to manipulate different parts of the Python Runtime Environment. It lets us access system-specific parameters and functions.
import pandas as pd # We use this as its a powerful data manipulation, handling and analysis library for python.
import torch # This is the core library of pytorch, which is a deep learning framework. It provides data structures for multi-dimensional tensors and operation on tensors.
from torchvision import transforms, models # They are submodules of torchvision providing common image transformations for data augumentation and preprocessing, and containing pre-trained models for image classification and other tasks respectively.
from torch.utils.data import Dataset, DataLoader # These procides tool for dataloading and batching.
from PIL import Image # This provides the function for opening, manupulating, and saving many different image file formats.
from tqdm import tqdm # It creates progress bars in python, useful for tracking long-running operations.

# Define the dataset class
class GlomeruliDataset(Dataset):
# Glomeruli Dataset Class defines how to load and process glomeruli images for the model.
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Traversing the directories and collecting image paths and labels
        for label, subdir in enumerate(['non_globally_sclerotic_glomeruli', 'globally_sclerotic_glomeruli']):
            subdir_path = os.path.join(root_dir, subdir)
            for img_name in os.listdir(subdir_path):
                if os.path.isfile(os.path.join(subdir_path, img_name)):
                    self.image_paths.append(os.path.join(subdir_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths) # Returns the total number of images in the dataset.

    def __getitem__(self, idx):
        img_path = self.image_paths[idx] # Retrieves an image and its corresponding label at the specified index.
        image = Image.open(img_path).convert('HSV') # Opening th eimage and converting it into HSV.
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], os.path.basename(img_path) # A tuple containing the preprocessed image, its label, and the filename.

# Define the image transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)), # Resizing to 224x224
    transforms.ToTensor(), # Converting the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizing the pixel values
])

# Loading the model, that is, pre-trained GoogleNet model and modifying it for the binary classification task.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
googlenet_model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
googlenet_model.fc = torch.nn.Linear(googlenet_model.fc.in_features, 1) # Replacing the final layer for binary classification.
googlenet_model.load_state_dict(torch.load('googlenet_model.pth', map_location=device)) # Loading the saved model weights
googlenet_model = googlenet_model.to(device)
googlenet_model.eval() #  Setting the model to evaluation mode (disabling dropout layers)

# Defining the function to run predictions and save the results
def evaluate_model(root_dir, output_csv):
    dataset = GlomeruliDataset(root_dir=root_dir, transform=transform_pipeline)
    dataloader = DataLoader(dataset, batch_size=46, shuffle=True)

    results = []
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = googlenet_model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            predictions = predictions.cpu().numpy().astype(int).flatten()
            for filename, prediction in zip(filenames, predictions):
                results.append([filename, prediction])

    results_df = pd.DataFrame(results, columns=['Image Name', 'Predicted Class'])
    results_df.to_csv(output_csv, index=False)
    print(f"\n Results saved to {output_csv}")

if __name__ == "__main__":
    # Hardcoded path for testing
    root_folder_path = 'C:/Users/shrut/Downloads/Glomeruli_Classification'
    if not os.path.isdir(root_folder_path):
        print(f"\n Error: {root_folder_path} is not a valid directory")
        sys.exit(1)

    output_csv_path = 'evaluation.csv'
    evaluate_model(root_folder_path, output_csv_path)
    
# Verification of the csv file, if generated properly.
try:
    eval_df = pd.read_csv(output_csv_path)
    print("\n Sample of the generated evaluation.csv:")
    print(eval_df.head())
except Exception as e:
    print(f"\n Error reading evaluation.csv: {e}")
    sys.exit(1)

"""
    Output:
    
    Evaluating: 100%|██████████| 126/126 [14:32<00:00,  6.93s/it]
     Results saved to evaluation.csv

     Sample of the generated evaluation.csv:
                                              Image Name  Predicted Class
    0  S-2006-002244_PAS_2of2_645432ad435c92704a3859d...                0
    1  S-1908-009877_PAS_2of3_64551cf7435c92704a3c663...                0
    2  S-2006-002185_PAS_2of2_645432a3435c92704a3851c...                0
    3  S-2103-004858_PAS_1of2_64552871435c92704a3feb2...                0
    4  S-1905-017785_PAS_1of2_64551c7b435c92704a3b8a2...                1
        
"""
