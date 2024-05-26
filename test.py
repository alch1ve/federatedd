import tensorflow as tf
import numpy as np
import cv2 as cv
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
import dataset


#lmao


# Load the saved model
saved_model_path = "C:/Users/aldri/federatedd/model/final_global_model.h5"
loaded_model = tf.keras.models.load_model(saved_model_path)

# Compile the model (ensure the compile configuration matches the one used during training)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the test dataset
npz_path = r"C:/Users/aldri/federatedd/dataset/CpE_Faculty_Members.npz"
data = np.load(npz_path)
x_train, x_test, y_train, y_test = dataset.load_dataset_from_npz(npz_path, test_size=0.2)

# Encode labels if necessary (same as in client.py)
label_encoder = LabelEncoder()
label_encoder.fit(y_train)  # Fit on all possible class names
y_test_encoded = label_encoder.transform(y_test)

# Evaluate the model on the test data
loss, accuracy = loaded_model.evaluate(x_test, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

loaded_model.summary()

# Initialize FaceNet model
embedder = FaceNet()

# Load the single image
image_path = "C:/Users/aldri/Downloads/Princess_Cyril_Malabanan/77e700aa-d9e3-4590-b5c6-d8bf65153389.jpg"
image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Detect faces in the image using MTCNN
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
    
    # Generate embedding for the resized face using FaceNet
    embedding = embedder.embeddings([resized_face])[0]  # FaceNet expects a batch of images
    
    # Reshape the embedding to match the input shape expected by the loaded model
    input_data = np.expand_dims(embedding, axis=0)
    
    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    
    # Decode the predicted class index to get the class name
    predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
    
    # Print the predicted class name
    print("Predicted Class:", predicted_class_name)
else:
    print("No faces detected in the image.")
