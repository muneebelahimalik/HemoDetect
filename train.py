import yaml
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
from google.colab import drive
from data_loader import load_and_preprocess_dataset
from data_utils import count_files_in_directory, get_image_paths, get_label_paths, get_image_resolution, count_class_distribution, plot_class_distribution, display_images_with_bounding_boxes
from model import create_model, compile_and_train_model, evaluate_model

# Mount Google Drive
drive.mount('/content/drive')

# Specify the path to your data folders on Google Drive
data_folder = '/content/drive/MyDrive/Blood_Cell_Dataset'
images_folder = Path('/content/drive/MyDrive/Blood_Cell_Dataset/train/images')
labels_folder = Path('/content/drive/MyDrive/Blood_Cell_Dataset/train/labels')
training_dir = '/content/drive/MyDrive/Blood_Cell_Dataset/train/images'
validation_dir = '/content/drive/MyDrive/Blood_Cell_Dataset/valid/images'
training_dir_1 = '/content/drive/MyDrive/Blood_Cell_Dataset/train'
validation_dir_1 = '/content/drive/MyDrive/Blood_Cell_Dataset/valid'

# Access the data from the YAML file
yaml_file_path = '/content/drive/MyDrive/Blood_Cell_Dataset/data.yaml'
with open(yaml_file_path, 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)

# Update YAML data paths
yaml_data['train'] = '/content/drive/MyDrive/Blood_Cell_Dataset/train/images'
yaml_data['val'] = '/content/drive/MyDrive/Blood_Cell_Dataset/valid/images'

# Save updated YAML data
with open('data.yaml', 'w') as f:
    yaml.dump(yaml_data, f)

# Load class names
class_names = yaml_data['names']

# Define the image preprocessing options
transform = T.ToTensor()  # Convert the images to tensors

# Create image data generator and load/preprocess the training dataset
train_dataset = load_and_preprocess_dataset(training_dir_1, target_size=(416, 416), batch_size=16)

# Create image data generator and load/preprocess the validation dataset
validation_dataset = load_and_preprocess_dataset(validation_dir_1, target_size=(416, 416), batch_size=16)

# Count the number of training files
file_count = count_files_in_directory(training_dir)
print(f"Training Images: {file_count}")

# Count the number of validation files
file_count = count_files_in_directory(validation_dir)
print(f"Validation Images: {file_count}")

# Get image paths and label paths
image_paths = get_image_paths(images_folder)
label_paths = get_label_paths(labels_folder)

# Display unique image resolutions
resolutions = get_image_resolution(image_paths)
unique_resolutions = set(resolutions)
print("Unique resolutions:", unique_resolutions)

# Count class distribution
class_counts = count_class_distribution(label_paths, class_names)
plot_class_distribution(class_counts)

# Display training images with bounding boxes
num_images = 5
image_files = random.sample(image_paths, num_images)
display_images_with_bounding_boxes(image_files, labels_folder)

# Define the Faster R-CNN model
num_classes = 3  # Number of disease classes
model = create_model(num_classes)

# Define the training parameters
learning_rate = 0.001
num_epochs = 25
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Compile and train the model
compile_and_train_model(model, train_dataset, validation_dataset, loss_fn, learning_rate, num_epochs)

# Evaluate the trained model
evaluate_model(model, validation_dataset)
