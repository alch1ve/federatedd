import tensorflow as tf
import dataset
from sklearn.preprocessing import LabelEncoder

# Load the saved model
saved_model_path = "C:/Users/aldri/federatedd/global model/final_global_model.keras"
loaded_model = tf.keras.models.load_model(saved_model_path)

# Compile the model (ensure the compile configuration matches the one used during training)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load the test dataset
npz_path = r"C:\Users\aldri\federatedd\dataset\Client_2.npz"
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
image_path = "C:/Users/aldri/Downloads/Princess_Cyril_Malabanan/c081d25d-1c0a-459f-81d8-55518f82c9e1.jpg"
image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert image to RGB format

# Create a copy of the original image
original_image = np.copy(image_rgb)

# Detect faces in the image using MTCNN
detector = MTCNN()
faces = detector.detect_faces(image_rgb)

# If faces are detected, proceed with prediction
if faces:
    # Extract the bounding box coordinates of the first detected face
    x, y, w, h = faces[0]['box']

    # Crop the face from the original image
    cropped_face = original_image[y:y+h, x:x+w]

    # Resize the cropped face to match the input shape expected by FaceNet
    resized_face = cv.resize(cropped_face, (160, 160))

    # Generate embedding for the resized face using FaceNet
    embedding = embedder.embeddings([resized_face])[0]  # FaceNet expects a batch of images

    # Reshape the embedding to match the input shape expected by the loaded model
    input_data = np.expand_dims(embedding, axis=0)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)

    # Get the predicted class index and accuracy
    predicted_class_index = np.argmax(predictions)
    accuracy = np.max(predictions)

    # Decode the predicted class index to get the class name
    predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]

    # Draw bounding box around the detected face (color: green) on the original image
    cv.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Put text with predicted class name and accuracy on the original image
    text = f"{predicted_class_name} (Confidence: {accuracy:.2f})"
    cv.putText(original_image, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert image back to BGR for displaying with OpenCV
    original_image_bgr = cv.cvtColor(original_image, cv.COLOR_RGB2BGR)

    # Display the original image with bounding box and accuracy using cv.imshow
    cv.imshow('Face Recognition', original_image_bgr)

    # Wait until a key is pressed
    cv.waitKey(0)

    # Close all OpenCV windows
    cv.destroyAllWindows()
else:
    print("No faces detected in the image.")
