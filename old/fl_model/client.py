import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_transformer_model
import os
from sklearn.preprocessing import StandardScaler
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

import pandas as pd
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6378137.0  # Radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_nearest_station(predicted_lat, predicted_lon, stations_df):

    distances = stations_df.apply(
        lambda row: haversine(predicted_lat, predicted_lon, row['latitude'], row['longitude']), axis=1
    )
    min_idx = distances.idxmin()
    nearest_station_name = stations_df.loc[min_idx, 'name']
    nearest_station_distance = distances[min_idx]
    return nearest_station_name


def load_client_data(path):
    df = pd.read_csv(path)

    # Drop rows with NaN values across all relevant columns
    df = df.dropna(subset=[
        "Start_date", "End_date", "Start_dayofweek", "End_dayofweek",
        "Start_lat", "Start_lon", "Trip_distance_m", "Trip_duration_ms", "Trip_duration_minutes",
        "Bike number", "Bike model", "End_lat", "End_lon"
    ])

    # Use some numeric input features; for example:
    X = df[["Start_date", "Start_dayofweek", "Start_lat", "Start_lon", "Bike number", "Start_geohash_enc","station_avg_dur","station_std_dur","station_top_dest_enc"]]
    X = X.apply(pd.to_numeric, errors="coerce")

    # Targets: Concatenate End_lat and End_lon into one DataFrame/array
    y_lat = pd.to_numeric(df["End_lat"], errors="coerce")
    y_lon = pd.to_numeric(df["End_lon"], errors="coerce")
    y_coords = pd.concat([y_lat, y_lon], axis=1)  # shape will be (num_samples, 2)

    # Assuming y_station (if needed) remains separate
    y_station = df["End station"]

    # Split into train and validation sets (adjust as needed)
    X_train, X_val, y_coords_train, y_coords_val, y_station_train, y_station_val = train_test_split(
        X, y_coords, y_station, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_coords_train, y_coords_val, y_station_train, y_station_val


# --- Flower Client Class ---
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, path_to_data):
        # Load client data
        X_train, X_val, y_coords_train, y_coords_val, y_station_train, y_station_val = load_client_data(path_to_data)
        
        # Assign training and validation data
        self.X_train, self.X_val = X_train.values, X_val.values
        self.y_coords_train, self.y_coords_val = y_coords_train.values, y_coords_val.values
        self.true_stations_train = y_station_train.values
        self.true_stations_val = y_station_val.values

        # Fit scalers on training target values (latitude and longitude)
        self.scaler_lat = StandardScaler().fit(self.y_coords_train[:, 0].reshape(-1, 1))
        self.scaler_lon = StandardScaler().fit(self.y_coords_train[:, 1].reshape(-1, 1))
        
        # Create the global model
        self.model = create_transformer_model(input_shape=(self.X_train.shape[1],))

    def get_parameters(self, config):
        return ndarrays_to_parameters(self.model.get_weights())
    
    def fit(self, parameters, config):
        # Handle both list and Parameters object formats
        if isinstance(parameters, list):
            weights = parameters  # Already a list of NumPy arrays
        else:
            weights = parameters_to_ndarrays(parameters)

        # Set global model weights
        self.model.set_weights(weights)

        # Train the global model on local data
        history = self.model.fit(
            self.X_train,
            self.y_coords_train,  # Provide y_coords_train directly as a single tensor
            epochs=20,  # Fewer epochs for local training
            batch_size=16,
            verbose=0
        )

        # Return updated weights, number of examples, and metrics
        updated_weights = self.model.get_weights()
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

        # Set global model weights
        self.model.set_weights(weights)

        # Check if evaluation set is empty to avoid returning a zero count
        if len(self.X_val) == 0:
            print("Warning: No evaluation samples available. Returning dummy evaluation metrics.")
            return 0.0, 1, {"geodesic_loss": 0.0, "station_acc": 0.0, "avg_distance_error": 0.0}

        # Convert data to float32 if necessary
        X_val = self.X_val.astype(np.float32)
        y_coords_val = self.y_coords_val.astype(np.float32)

        # Get predictions; model outputs a (batch, 2) tensor
        preds = self.model.predict(X_val)
        pred_lat = preds[:, 0]
        pred_lon = preds[:, 1]

        # Load station mapping data and build mapping dictionary
        stations_df = pd.read_csv("data/stations_geolocated.csv")
        stations_mapping = {row['name']: (row['latitude'], row['longitude']) for idx, row in stations_df.iterrows()}

        predicted_stations = []
        distance_errors = []
        true_stations = self.true_stations_val

        # For each sample, compute nearest station and haversine error compared to true station coordinates
        for i, (lat_val, lon_val, true_station) in enumerate(zip(pred_lat, pred_lon, true_stations)):
            predicted_station = find_nearest_station(lat_val, lon_val, stations_df)
            predicted_stations.append(predicted_station)
            if true_station in stations_mapping:
                true_coords = stations_mapping[true_station]
                error_distance = haversine(lat_val, lon_val, true_coords[0], true_coords[1])
                distance_errors.append(error_distance)
            else:
                distance_errors.append(np.nan)

        # Compute average distance error (ignoring NaNs)
        avg_distance_error = np.nanmean(distance_errors)

        # Calculate station prediction accuracy
        station_accuracy = np.mean(np.array(predicted_stations) == np.array(true_stations))

        # Evaluate the model
        results = self.model.evaluate(X_val, y_coords_val, verbose=0)
        geodesic_loss_value = results[0]  # Assuming the first value is the loss

        # Return the required tuple
        metrics = {
            "geodesic_loss": geodesic_loss_value,
            "station_acc": station_accuracy,
            "avg_distance_error": avg_distance_error,
        }
        print(f"Geodesic loss: {geodesic_loss_value}, Station accuracy: {station_accuracy}, Avg distance error: {avg_distance_error}")

        return geodesic_loss_value, len(self.X_val), metrics


# --- Entry point to run this client individually (for testing only) ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_data.csv>")
        exit(1)

    path = sys.argv[1]
    fl.client.start_numpy_client(server_address="localhost:8080", client=FederatedClient(path))
