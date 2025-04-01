import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model

import json

# Load station label mapping
with open("data/station_mapping.json", "r") as f:
    station_to_index = json.load(f)

def load_client_data(path):
    df = pd.read_csv(path)

    # Features
    X = df[["Start_hour", "Start_dayofweek", "Start_lat", "Start_lon"]]

    # Targets
    y_duration = df["Trip_duration_minutes"]
    y_station = df["End station number"].astype(str).map(station_to_index)

    # Split data
    X_train, X_val, y_dur_train, y_dur_val = train_test_split(X, y_duration, test_size=0.2, random_state=42)
    _, _, y_dest_train, y_dest_val = train_test_split(X, y_station, test_size=0.2, random_state=42)

    return X_train, X_val, y_dur_train, y_dur_val, y_dest_train, y_dest_val



# --- Flower Client Class ---
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, path_to_data, num_stations):
        X_train, X_val, y_dur_train, y_dur_val, y_dest_train, y_dest_val = load_client_data(path_to_data)
        self.X_train, self.X_val = X_train.values, X_val.values
        self.y_dur_train, self.y_dur_val = y_dur_train.values, y_dur_val.values
        self.y_dest_train, self.y_dest_val = y_dest_train.values, y_dest_val.values

        self.model = create_model(input_shape=(self.X_train.shape[1],), num_stations=num_stations)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.X_train,
            {"duration": self.y_dur_train, "destination": self.y_dest_train},
            epochs=1, batch_size=32, verbose=0
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        results = self.model.evaluate(
            self.X_val,
            {"duration": self.y_dur_val, "destination": self.y_dest_val},
            verbose=0
        )
        print(f"[CLIENT] Eval loss={results[0]:.4f}, MAE={results[1]:.4f}, Acc={results[2]:.4f}")
        return results[0], len(self.X_val), {"mae": results[1], "accuracy": results[2]}


# --- Entry point to run this client individually (for testing only) ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python client.py <client_data.csv> <num_stations>")
        exit(1)

    path = sys.argv[1]
    num_stations = int(sys.argv[2])
    fl.client.start_numpy_client("localhost:8080", client=FederatedClient(path, num_stations))
