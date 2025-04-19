import flwr as fl
import os
import random
from client import FederatedClient
from flwr.common import Context
from flwr.server.strategy import FedAvg  # Import the FedAvg strategy
import numpy as np
from flwr.common import ndarrays_to_parameters
from model import create_model_complex, create_transformer_model

os.environ["RAY_memory_usage_threshold"] = "0.99"
os.environ["RAY_memory_monitor_refresh_ms"] = "100"

# --- Load available client file paths ---
def get_client_paths(folder, limit=None):
    files = [f for f in os.listdir(folder) if f.startswith("client_") and f.endswith(".csv")]
    files = sorted(files)
    if limit:
        files = files[:limit]
    return [os.path.join(folder, f) for f in files]

# --- ClientManager: Wrap client initialization into new API with random selection ---
class FlowerClientManager(fl.client.NumPyClient):
    def __init__(self, client_paths):
        self.client_paths = client_paths

    def get_properties(self, config):
        return {}

    def get_parameters(self, config):
        client = self._load_client()
        return client.get_parameters(config)

    def fit(self, parameters, config):
        client = self._load_client()
        return client.fit(parameters, config)

    def evaluate(self, parameters, config):
        client = self._load_client()
        return client.evaluate(parameters, config)

    def _load_client(self):
        # Randomly select a client file for each round
        path = random.choice(self.client_paths)
        return FederatedClient(path_to_data=path)

# --- Simulation Entry Point ---
if __name__ == "__main__":
    data_folder = "data/clients"
    client_paths = get_client_paths(data_folder, limit=5)

    def client_fn(context: Context) -> fl.client.Client:
        # Create the client using the updated manager that randomly selects a CSV each time
        return FlowerClientManager(client_paths).to_client()
    
    from flwr.server.strategy import FedOpt

    # Replace this with the actual structure of your model's weights
    model = create_transformer_model(input_shape=(9,))  # Adjust input shape as needed
    initial_weights = model.get_weights()  # Get the actual weights from the model
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = FedOpt(
        initial_parameters=initial_parameters,
        min_fit_clients=5,
        min_available_clients=5,
        eta=0.01,  # Server learning rate
        eta_l=0.1,  # Client learning rate
        tau=0.9,  # Momentum parameter
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_paths),
        config=fl.server.ServerConfig(num_rounds=25),
        client_resources={"num_cpus": 6, "num_gpus": 1},
        ray_init_args={"num_cpus": 12, "num_gpus": 1},
        strategy=strategy,
    )
