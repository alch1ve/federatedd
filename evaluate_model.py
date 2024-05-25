import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import dataset  # Ensure this module contains the function to load the dataset
from model import create_model

# Path to the saved global model
model_path = "C:/Users/aldri/federatedd/model/final_global_model.h5"

# Path to your .npz dataset
npz_path = r"C:\Users\aldri\federatedd\dataset\CpE_Faculty_Members.npz"

# Load dataset
x_train, x_test, y_train, y_test = dataset.load_dataset_from_npz(npz_path, test_size=0.2)

# Encode labels as integers
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
