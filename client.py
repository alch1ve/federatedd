import argparse
import numpy as np
from flwr.client import ClientApp, NumPyClient
import tensorflow as tf
import dataset
import model as model_module

# Make TensorFlow log less verbose
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define the path to your .npz dataset
npz_path = r"C:\Users\aldri\federatedd\partitions\partition_2.npz"

# Load dataset
x_train, x_test, y_train, y_test = dataset.load_dataset_from_npz(npz_path, test_size=0.2)

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, model):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model = model

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=10,
            batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"accuracy": accuracy}

def client_fn(cid: str, partition_id: int):
    """Create and return an instance of Flower `Client`."""
    # Create model instance for the client
    num_classes = len(np.unique(y_train_encoded))  # Number of unique classes (i.e., number of persons)
    input_shape = x_train.shape[1]  # Number of features
    dense_model = model_module.create_model(input_shape, num_classes)
    dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return FlowerClient(x_train, y_train_encoded, x_test, y_test_encoded, dense_model).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, help="Client ID")
    parser.add_argument("--partition_id", type=int, help="Partition ID")
    args = parser.parse_args()

    start_client(
        server_address="192.168.1.10:8080",
        client=client_fn(args.client_id, args.partition_id),
    )
