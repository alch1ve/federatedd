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
