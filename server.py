from typing import List, Tuple
import argparse
import numpy as np
import tensorflow as tf
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, Weights
import dataset
import model as model_module


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define initial model training function
def train_initial_model(x_train, y_train, input_shape, num_classes, initial_epochs=3, batch_size=32):
    model = model_module.create_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=initial_epochs, batch_size=batch_size)
    return model.get_weights()


# Define config
config = ServerConfig(num_rounds=3)

# Define strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    # Load the dataset
    npz_path = r"C:\Users\aldri\federatedd\dataset\CpE_Faculty_Members.npz"
    x_train, _, y_train, _ = dataset.load_dataset_from_npz(npz_path, test_size=0.2)
    num_classes = len(np.unique(y_train))  # Number of unique classes (i.e., number of persons)
    input_shape = x_train.shape[1]  # Number of features

    # Train initial model
    initial_model_weights = train_initial_model(x_train, y_train, input_shape, num_classes)

    # Start server with initial model weights
    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
        initial_model=Weights(initial_model_weights),
    )
