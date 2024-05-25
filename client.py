import argparse
import numpy as np
from flwr.client import ClientApp, NumPyClient
import tensorflow as tf
import dataset
import model as model_module

# Make TensorFlow log less verbose
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define the base path to your dataset
dataset_base_path = r"C:\Users\dgzhi\OneDrive\Desktop\Federated\dataset\cpe_faculty"

# Load dataset
x_train, x_test, y_train, y_test = dataset.load_dataset(dataset_base_path, test_size=0.2)

# Partition the dataset
def partition_data(x, y, num_partitions):
    partition_size = len(x) // num_partitions
    partitions = []
    for i in range(num_partitions):
        start = i * partition_size
        end = start + partition_size
        partitions.append((x[start:end], y[start:end]))
    return partitions

train_partitions = partition_data(x_train, y_train, 2)
test_partitions = partition_data(x_test, y_test, 2)

# Print dataset shapes after partitioning
print("Training data shapes after partitioning:", [(x.shape, y.shape) for x, y in train_partitions])
print("Testing data shapes after partitioning:", [(x.shape, y.shape) for x, y in test_partitions])

# Verify the content of the partitions (optional, for detailed verification)
print("Training partition 0 labels:", np.unique(train_partitions[0][1], return_counts=True))
print("Training partition 1 labels:", np.unique(train_partitions[1][1], return_counts=True))
print("Testing partition 0 labels:", np.unique(test_partitions[0][1], return_counts=True))
print("Testing partition 1 labels:", np.unique(test_partitions[1][1], return_counts=True))

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
    # Load the partitioned data based on the partition ID
    x_partition, y_partition = train_partitions[partition_id]
    x_val, y_val = test_partitions[partition_id]
    # Create model instance for the client
    num_classes = len(np.unique(y_train))  # Number of unique classes (i.e., number of persons)
    input_shape = x_train[0].shape
    cnn_model = model_module.create_model(input_shape, num_classes)
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return FlowerClient(x_partition, y_partition, x_val, y_val, cnn_model).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, help="Client ID")
    parser.add_argument("--partition_id", type=int, help="Partition ID")
    args = parser.parse_args()

    start_client(
        server_address="192.168.1.10:8080",
        client=client_fn(args.client_id, args.partition_id), 
    )

