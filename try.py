import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Load the saved model
saved_model_path = "C:/Users/aldri/federatedd/model/final_global_model.h5"
loaded_model = tf.keras.models.load_model(saved_model_path)

# Compile the model (ensure the compile configuration matches the one used during training)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    img = img.resize(target_size)  # Resize the image to the target size
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = img_array / 255.0  # Normalize the image array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

input_shape = (None,512)  # Update to match the input shape of your model during training
# Path to the image you want to test
image_path = "C:/Users/aldri/OneDrive/Pictures/Screenshots/Screenshot 2024-05-26 113732.png"

# Preprocess the image
preprocessed_image = preprocess_image(image_path, input_shape[:2])  # Pass only (512, 512) for resizing

# Predict the label
predicted_probabilities = loaded_model.predict(preprocessed_image)
predicted_class_index = np.argmax(predicted_probabilities, axis=1)

# Decode the predicted class index
predicted_label = label_encoder.inverse_transform(predicted_class_index)

print("Predicted Label:", predicted_label[0])
