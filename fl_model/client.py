import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model, create_model_2, create_model_complex
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Optionally, enable GPU memory growth
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled on GPUs")
    except RuntimeError as e:
        print(e)

def load_client_data(path):
    df = pd.read_csv(path)

    # Use the same input features (you can also include additional features)
    X = df[["Start_date", "End_date", "Start_dayofweek", "End_dayofweek", "Start_lat", "Start_lon"]]

    # Targets:
    y_duration = df["Trip_duration_minutes"]
    y_lat = df["End_lat"]
    y_lon = df["End_lon"]

    # Split data into training and validation sets; now you have two target outputs (y_lat, y_lon)
    X_train, X_val, y_dur_train, y_dur_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val = train_test_split(
        X, y_duration, y_lat, y_lon, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_dur_train, y_dur_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val


# --- Flower Client Class ---
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, path_to_data):
        X_train, X_val, y_dur_train, y_dur_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val = load_client_data(path_to_data)
        self.X_train, self.X_val = X_train.values, X_val.values
        self.y_dur_train, self.y_dur_val = y_dur_train.values, y_dur_val.values
        self.y_lat_train, self.y_lat_val = y_lat_train.values, y_lat_val.values
        self.y_lon_train, self.y_lon_val = y_lon_train.values, y_lon_val.values

        self.model = create_model_complex(input_shape=(self.X_train.shape[1],))

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.X_train,
            {"duration": self.y_dur_train, "lat": self.y_lat_train, "lon": self.y_lon_train},
            epochs=1, batch_size=32, verbose=0
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        results = self.model.evaluate(
            self.X_val,
            {"duration": self.y_dur_val, "lat": self.y_lat_val, "lon": self.y_lon_val},
            verbose=0
        )
        print("Full evaluation results:", results)
        # Adjust print indices based on the order of outputs; for example, if the results order is:
        # [total_loss, duration_loss, lat_loss, lon_loss, duration_mae, lat_mae, lon_mae]
        print(f"[CLIENT] Duration MAE = {results[4]:.4f}, Lat MAE = {results[5]:.4f}, Lon MAE = {results[6]:.4f}")
        return results[0], len(self.X_val), {"duration_mae": results[4], "lat_mae": results[5], "lon_mae": results[6]}

# --- Entry point to run this client individually (for testing only) ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_data.csv>")
        exit(1)

    path = sys.argv[1]
    fl.client.start_numpy_client("localhost:8080", client=FederatedClient(path))
