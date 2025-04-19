import flwr as fl

# Federated Averaging strategy with basic settings
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,              # All clients participate in training
    min_fit_clients=2,             # At least 2 clients needed to train
    min_available_clients=2        # Wait until 2 clients are connected
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
