import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_image_data_generator(rescale):
    return ImageDataGenerator(rescale=rescale)

def load_and_preprocess_dataset(directory, target_size, batch_size):
    image_data_generator = create_image_data_generator(1.0/255.0)
    
    train_dataset = image_data_generator.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse'
    )
    
    return train_dataset

def load_image(file_path):
    return cv2.imread(file_path)
