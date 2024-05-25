from typing import List, Tuple
from flwr.server import ServerApp, ServerConfig
from flwr.common import Parameters, Metrics
import flwr as fl
from model import create_model
import tensorflow as tf
import dataset
import numpy as np


npz_path = r"C:\Users\aldri\federatedd\partitions\partition_1.npz"
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, global_model: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.num_rounds = kwargs.get('num_rounds', 3)  # Default to 3 if not provided

    def aggregate_fit(self, rnd: int, results: List[Tuple[str, Parameters, Metrics]], failures: List[str]):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if rnd == self.num_rounds:
            self.save_model(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

    def save_model(self, parameters: Parameters):
        weights = fl.common.parameters_to_ndarrays(parameters)
        self.global_model.set_weights(weights)
        self.global_model.save("C:/Users/aldri/federatedd/model/final_global_model.h5")
        print("Model saved as final_global_model.h5")

    def aggregate_evaluate(self, rnd: int, results: List[Tuple[str, Metrics]], failures: List[str]) -> Metrics:
        aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        accuracies = [res[1].metrics['accuracy'] for res in results if res[1].metrics is not None]
        if accuracies:
            average_accuracy = sum(accuracies) / len(accuracies)
            print(f"Round {rnd}: Average accuracy: {average_accuracy:.3f}")
        else:
            print(f"Round {rnd}: No accuracies to average.")
        return aggregated_metrics

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Load your data and pre-train the global model
x_train, x_test, y_train, y_test = dataset.load_dataset_from_npz(npz_path, test_size=0.2)
num_classes = len(np.unique(y_train))
input_shape = x_train.shape[1]
global_model = create_model(input_shape, num_classes)
global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
global_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Define strategy with pre-trained global model
strategy = CustomFedAvg(global_model=global_model, evaluate_metrics_aggregation_fn=weighted_average)

# Define config
config = ServerConfig(num_rounds=3)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )
