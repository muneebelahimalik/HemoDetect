import os
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def count_files_in_directory(directory):
    return len(os.listdir(directory))

def get_image_paths(images_folder):
    return list(images_folder.glob("*.jpg"))

def get_label_paths(labels_folder):
    return sorted(labels_folder.glob("*.txt"))

def get_image_resolution(image_path):
    img = cv2.imread(str(image_path))
    h, w, _ = img.shape
    return w, h

def count_class_distribution(label_paths, class_names):
    class_counts = Counter()
    for label_file in label_paths:
        with open(label_file, "r") as file:
            lines = file.readlines()
        class_counts.update(Counter([int(line.split()[0]) for line in lines]))
    
    class_counts_names = {class_names[int(class_id)]: count for class_id, count in class_counts.items()}
    
    return class_counts_names

def plot_class_distribution(class_counts_names):
    df = pd.DataFrame.from_dict(class_counts_names, orient="index", columns=["count"])
    ax = df.plot(kind="bar")
    plt.xlabel("Classes")
    plt.ylabel("Number of Instances")
    plt.title("Class Distribution")
    plt.show()

def display_images_with_bounding_boxes(selected_image_files, labels_folder, class_list, colors):
    for selected_image_file in selected_image_files:
        demo_image = selected_image_file
        demo_label = pathlib.Path(labels_folder) / f"{selected_image_file.stem}.txt"

        image = cv2.imread(str(demo_image))
        height, width, _ = image.shape
        T = []

        with open(demo_label, "r") as file1:
            for line in file1.readlines():
                split = line.split(" ")
                class_id = int(split[0])
                color = colors[class_id]
                clazz = class_list[class_id]
                x, y, w, h = float(split[1]), float(split[2]), float(split[3]), float(split[4])
                box = [int((x - 0.5*w)* width), int((y - 0.5*h) * height), int(w*width), int(h*height)]
                cv2.rectangle(image, box, color, 2)
                cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(image, class_list[class_id], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = cv2.resize(image, (600, 600))
            plt.show()
