import os
from typing import List, Tuple
from flwr.server import ServerApp, ServerConfig
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, Metrics
import flwr as fl
from model import create_model

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, save_path=None, **kwargs):
        super().__init__(**kwargs)
        self.global_model = None
        self.num_rounds = kwargs.get('num_rounds', 3)
        self.save_path = save_path

    def initialize_model(self, parameters: Parameters):
        print("Initializing global model...")
        if self.global_model is None:
            print("Global model is None")
            weights = fl.common.parameters_to_ndarrays(parameters)
            print("Weights:", weights)
            input_shape = weights[0].shape[0]
            print("Input shape:", input_shape)
            self.num_classes = sum(weights[-1])  # Sum up the number of classes from all clients
            print("Number of classes:", self.num_classes)
            self.global_model = create_model(input_shape, self.num_classes)
            print("Global model created:", self.global_model)
            self.global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.global_model.set_weights(weights)


    def aggregate_fit(self, rnd: int, results: List[Tuple[str, FitRes]], failures: List[BaseException]) -> Tuple[Parameters, Scalar]:
        print(f"Aggregate fit for round {rnd}, {len(results)} results received")
        if rnd == 0 and self.global_model is None:
            # Initialize global model with parameters from the first client's result
            if results:
                print("Initializing global model...")
                parameters = results[0][1].parameters
                print("Parameters received from the first client:", parameters)
                self.initialize_model(parameters)
            else:
                print("No results received for the first round. Cannot initialize global model.")
                return Parameters([]), 0.0
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if rnd == self.num_rounds:
            # Save the global model after the final round
            if self.global_model is not None:  # Check if global model is initialized
                print("Saving final global model...")
                self.save_model(aggregated_parameters)
            else:
                print("Global model is not initialized!")
        return aggregated_parameters, aggregated_metrics


    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        accuracies = [res[1].metrics['accuracy'] for res in results if res[1].metrics is not None]
        if accuracies:
            average_accuracy = sum(accuracies) / len(accuracies)
            print(f"Round {rnd}: Average accuracy: {average_accuracy:.3f}")
        else:
            print(f"Round {rnd}: No accuracies to average.")
        return aggregated

    def save_model(self):
        if self.save_path is not None:
            self.global_model.save(self.save_path)
            print(f"Global model saved to {self.save_path}")

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = CustomFedAvg(save_path="C:/Users/aldri/federatedd/model/final_global_model.h5", evaluate_metrics_aggregation_fn=weighted_average)

config = ServerConfig(num_rounds=3)

app = ServerApp(
    config=config,
    strategy=strategy,
)

if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )
