import flwr as fl
import os
import pandas as pd
from client import FederatedClient
import json
from flwr.common import ndarrays_to_parameters, Parameters
import numpy as np
from flwr.common import Context

import os
os.environ["RAY_memory_usage_threshold"] = "0.99"
os.environ["RAY_memory_monitor_refresh_ms"] = "100"

# --- Load available client file paths ---
def get_client_paths(folder, limit=None):
    files = [f for f in os.listdir(folder) if f.startswith("client_") and f.endswith(".csv")]
    files = sorted(files)
    if limit:
        files = files[:limit]
    return [os.path.join(folder, f) for f in files]

# --- Load station mapping ---
with open("data/station_mapping.json", "r") as f:
    station_to_index = json.load(f)
num_stations = len(station_to_index)

# --- ClientManager: Wrap client initialization into new API ---
class FlowerClientManager(fl.client.NumPyClient):
    def __init__(self, client_id, client_paths):
        self.client_id = int(client_id)
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
        path = self.client_paths[self.client_id]
        return FederatedClient(path_to_data=path, num_stations=num_stations)

# --- Simulation Entry Point ---
if __name__ == "__main__":
    data_folder = "data/clients"
    client_paths = get_client_paths(data_folder, limit=10)

    def client_fn(context: Context) -> fl.client.Client:
        return FlowerClientManager(context.run_id, client_paths).to_client()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_paths),
        config=fl.server.ServerConfig(num_rounds=10),
        client_resources={"num_cpus": 1},
        ray_init_args={"num_cpus": 4}
    )
