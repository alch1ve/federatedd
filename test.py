import tensorflow as tf
import dataset
from sklearn.preprocessing import LabelEncoder

# Load the saved model
saved_model_path = "C:/Users/aldri/federatedd/model/final_global_model.h5"
loaded_model = tf.keras.models.load_model(saved_model_path)

# Compile the model (ensure the compile configuration matches the one used during training)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load the test dataset
npz_path = r"C:\Users\aldri\federatedd\dataset\Client_1.npz"
x_train, x_test, y_train, y_test = dataset.load_dataset_from_npz(npz_path, test_size=0.2)

# Encode labels if necessary (same as in client.py)
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_test_encoded = tf.keras.utils.to_categorical(y_test_encoded, num_classes=5)  # Convert to one-hot encoding

# Evaluate the model on the test data
loss, accuracy = loaded_model.evaluate(x_test, y_test_encoded)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

loaded_model.summary()

import cv2 as cv
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

# Initialize FaceNet model
embedder = FaceNet()

# Load the single image
image_path = "C:/Users/aldri/OneDrive/Pictures/Screenshots/Screenshot 2024-05-26 121439.png"
image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Detect faces in the image using MTCNN or any other face detection algorithm
detector = MTCNN()
faces = detector.detect_faces(image_rgb)

# If faces are detected, proceed with prediction
if faces:
    # Extract the bounding box coordinates of the first detected face
    x, y, w, h = faces[0]['box']
    
    # Crop the face from the image
    cropped_face = image_rgb[y:y+h, x:x+w]
    
    # Resize the cropped face to match the input shape expected by FaceNet
    resized_face = cv.resize(cropped_face, (160, 160))
    
    # Normalize the image to match FaceNet's expected input format
    resized_face = np.around(resized_face / 255.0, decimals=12)

    # Generate embedding for the resized face using FaceNet
    embedding = embedder.embeddings([resized_face])[0]  # Assuming embedder.embeddings() expects a batch of images
    
    # Ensure the embedding is in the correct shape
    input_data = np.expand_dims(embedding, axis=0)  # Shape should be (1, embedding_size)
    labels_npz = np.load(npz_path)
    labels = labels_npz
    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)
    
    # Get the predicted class label
    predicted_class = np.argmax(predictions)
    labels = labels_npz['arr_1']  # Assuming labels are stored in 'arr_1'

# Assuming you have already obtained the predicted class index
predicted_class_index = predicted_class

# Get the label corresponding to the predicted class index
predicted_label = labels[predicted_class_index]

# Print the predicted label
print("Predicted Label:", predicted_label)
