from typing import List, Tuple
from flwr.server import ServerApp, ServerConfig
from flwr.common import Metrics
import flwr as fl
from model import create_model

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_model = create_model(512, 6)  # Global model with 6 classes
        self.num_rounds = kwargs.get('num_rounds', 3)

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if rnd == self.num_rounds:
            self.save_model(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

    def save_model(self, parameters):
        weights = fl.common.parameters_to_ndarrays(parameters)
        self.global_model.set_weights(weights)
        self.global_model.save("C:/Users/aldri/federatedd/model/final_global_model.h5")
        print("Model saved as final_global_model.h5")

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        accuracies = [res[1].metrics['accuracy'] for res in results if res[1].metrics is not None]
        if accuracies:
            average_accuracy = sum(accuracies) / len(accuracies)
            print(f"Round {rnd}: Average accuracy: {average_accuracy:.3f}")
        else:
            print(f"Round {rnd}: No accuracies to average.")
        return aggregated

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average)

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
