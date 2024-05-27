import argparse
import numpy as np
from flwr.client import ClientApp, NumPyClient
import tensorflow as tf
import dataset
import model as model_module
import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load dataset
npz_path = r"C:\Users\aldri\federatedd\dataset\Class_1.npz"
x_train, x_test, y_train, y_test = dataset.load_dataset_from_npz(npz_path, test_size=0.2)

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, model, global_num_classes):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model = model
        self.global_num_classes = global_num_classes

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # One-hot encode the labels to match the global model's output shape
        y_train_global = tf.keras.utils.to_categorical(self.y_train, self.global_num_classes)
        y_val_global = tf.keras.utils.to_categorical(self.y_val, self.global_num_classes)
        
        self.model.fit(
            self.x_train,
            y_train_global,
            validation_data=(self.x_val, y_val_global),
            epochs=10,
            batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        y_val_global = tf.keras.utils.to_categorical(self.y_val, self.global_num_classes)
        loss, accuracy = self.model.evaluate(self.x_val, y_val_global)
        return loss, len(self.x_val), {"accuracy": accuracy}

def client_fn(cid: str):
    # Define number of classes for the global model
    global_num_classes = 6
    num_classes = len(np.unique(y_train))  # Local number of classes (3 in this case)
    input_shape = x_train.shape[1]
    model = model_module.create_model(input_shape, global_num_classes)  # Global model with 6 classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return FlowerClient(x_train, y_train, x_test, y_test, model, global_num_classes).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, help="Client ID")
    args = parser.parse_args()

    start_client(
        server_address="172.16.197.173:8080",
        client=client_fn(args.client_id),
    )
