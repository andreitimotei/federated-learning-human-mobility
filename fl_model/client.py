import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model_complex
import os
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Optionally, enable GPU memory growth
physical_gpus = tf.config.list_physical_devices('GPU')
if (physical_gpus):
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled on GPUs")
    except RuntimeError as e:
        print(e)

def load_client_data(path):
    df = pd.read_csv(path)

    # Drop rows with NaN values across all relevant columns
    df = df.dropna(subset=[
        "Start_date", "End_date", "Start_dayofweek", "End_dayofweek",
        "Start_lat", "Start_lon", "Trip_distance_km", "Trip_duration_minutes",
        "Bike number", "Bike model", "End_lat", "End_lon"
    ])

    # Use the same input features (you can also include additional features)
    X = df[[
        "Start_date", "End_date", "Start_dayofweek", "End_dayofweek",
        "Start_lat", "Start_lon", "Trip_distance_km", "Trip_duration_minutes"
    ]]

    # Convert all columns to numerical types
    X = X.apply(pd.to_numeric, errors="coerce")

    # Targets:
    y_lat = pd.to_numeric(df["End_lat"], errors="coerce")
    y_lon = pd.to_numeric(df["End_lon"], errors="coerce")

    # Split data into training and validation sets
    X_train, X_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val = train_test_split(
        X, y_lat, y_lon, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val


# --- Flower Client Class ---
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, path_to_data):
        X_train, X_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val = load_client_data(path_to_data)
        self.X_train, self.X_val = X_train.values, X_val.values
        self.y_lat_train, self.y_lat_val = y_lat_train.values, y_lat_val.values
        self.y_lon_train, self.y_lon_val = y_lon_train.values, y_lon_val.values

        self.model = create_model_complex(input_shape=(self.X_train.shape[1],))

    def get_parameters(self, config):
        return ndarrays_to_parameters(self.model.get_weights())

    def fit(self, parameters, config):
        # Handle both list and Parameters object formats
        if isinstance(parameters, list):
            weights = parameters  # Already a list of NumPy arrays
        else:
            weights = parameters_to_ndarrays(parameters)

        # Set model weights
        self.model.set_weights(weights)

        # Train the model
        history = self.model.fit(
            self.X_train,
            {"lat": self.y_lat_train, "lon": self.y_lon_train},
            epochs=10, batch_size=16, verbose=0
        )

        # Return updated weights, number of examples, and metrics
        updated_weights = self.model.get_weights()  # Return as a list of NumPy arrays
        num_examples = len(self.X_train)
        metrics = {"loss": history.history["loss"][-1]}  # Add additional metrics if needed

        print("Metrics:", metrics)

        return updated_weights, num_examples, metrics

    def evaluate(self, parameters, config):
        # Handle both list and Parameters object formats
        if isinstance(parameters, list):
            weights = parameters  # Already a list of NumPy arrays
        else:
            weights = parameters_to_ndarrays(parameters)

        # Set model weights
        self.model.set_weights(weights)

        # Convert data to float32 (if necessary)
        X_val = self.X_val.astype(np.float32)
        y_lat_val = self.y_lat_val.astype(np.float32)
        y_lon_val = self.y_lon_val.astype(np.float32)

        # Evaluate the model
        results = self.model.evaluate(
            X_val,
            {"lat": y_lat_val, "lon": y_lon_val},
            verbose=0
        )
        print("Full evaluation results:", results)
        print("Lat MAE:", results[3])
        print("Lon MAE:", results[4])
        return results[0], len(X_val), {"lat_mae": results[3], "lon_mae": results[4]}

# --- Entry point to run this client individually (for testing only) ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_data.csv>")
        exit(1)

    path = sys.argv[1]
    fl.client.start_numpy_client(server_address="localhost:8080", client=FederatedClient(path))
