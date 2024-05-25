import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_images_from_directory(directory, label, img_size=(32, 32)):
    images = []
    labels = []
    for img_name in os.listdir(directory):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(directory, img_name)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels


def load_dataset_npz(file_path, test_size=0.2):
    # Load the dataset from the .npz file
    data = np.load(file_path)
    x, y = data['x'], data['y']

    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    return x_train, x_test, y_train, y_test

