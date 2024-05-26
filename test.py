import tensorflow as tf
import dataset

# Load the saved model
saved_model_path = "C:/Users/aldri/federatedd/model/final_global_model.h5"
loaded_model = tf.keras.models.load_model(saved_model_path)

# Compile the model (ensure the compile configuration matches the one used during training)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the test dataset
npz_path = r"C:\Users\aldri\federatedd\dataset\Class_1.npz"
x_train, x_test, y_train, y_test = dataset.load_dataset_from_npz(npz_path, test_size=0.2)  # Load the test set

# Encode labels if necessary (same as in client.py)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

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
    
    # Resize the cropped face to match the input shape expected by the loaded model
    resized_face = cv.resize(cropped_face, (160, 160))
    
    # Generate embedding for the resized face using FaceNet
    embedding = embedder.embeddings(resized_face)[0]  # Assuming embedder.embeddings() returns a list of embeddings
    
    # Reshape the embedding to match the input shape expected by the loaded model
    input_data = embedding.reshape(1, -1)
    
    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)
    
    # Get the predicted class label
    predicted_class = np.argmax(predictions)
    
    # Print the predicted class label
    print("Predicted Class:", predicted_class)
else:
    print("No faces detected in the image.")
