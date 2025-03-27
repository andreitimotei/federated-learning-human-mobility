import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model
import logging

logging.basicConfig(filename="client_log.txt", level=logging.INFO)


# Load local client dataset
def load_client_data(path):
    df = pd.read_csv(path)

    features = df[[
        "Start_hour", "Start_dayofweek",
        "Start_lat", "Start_lon"
    ]]
    labels = df["Trip_duration_minutes"]

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train.values, y_train.values, X_val.values, y_val.values

# Create Flower client
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, client_data_path):
        self.X_train, self.y_train, self.X_val, self.y_val = load_client_data(client_data_path)
        self.model = create_model(input_shape=(self.X_train.shape[1],))

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.X_val, self.y_val, verbose=0)

        # Print nicely formatted metrics
        print(f"[CLIENT] Evaluation -> Loss: {loss:.4f}, MAE: {mae:.4f} (Samples: {len(self.y_val)})")
        logging.info(f"Evaluation: Loss={loss:.4f}, MAE={mae:.4f}, Samples={len(self.y_val)}")


        return loss, len(self.X_val), {"mae": mae}


# Run as a standalone client
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fl_model/client.py <client_data_path>")
        exit(1)

    client_data_path = sys.argv[1]
    fl.client.start_numpy_client(server_address="localhost:8080", client=FederatedClient(client_data_path))
