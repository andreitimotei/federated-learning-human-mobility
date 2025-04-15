import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model_complex
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
    print(f"Finding nearest station for predicted coordinates: {predicted_lat}, {predicted_lon}")
    # stations_df should have columns for station id, latitude, and longitude
    distances = stations_df.apply(
        lambda row: haversine(predicted_lat, predicted_lon, row['latitude'], row['longitude']), axis=1
    )
    min_idx = distances.idxmin()
    nearest_station_name = stations_df.loc[min_idx, 'name']
    nearest_station_distance = distances[min_idx]
    print(f"Nearest station: {nearest_station_name}, Distance: {nearest_station_distance} meters")
    return nearest_station_name



def load_client_data(path):
    df = pd.read_csv(path)

    # Drop rows with NaN values across all relevant columns
    df = df.dropna(subset=[
        "Start_date", "End_date", "Start_dayofweek", "End_dayofweek",
        "Start_lat", "Start_lon", "Trip_distance_m", "Trip_duration_ms", "Trip_duration_minutes",
        "Bike number", "Bike model", "End_lat", "End_lon"
    ])

    # Use the same input features (you can also include additional features)
    X = df[["Start_date", "Start_dayofweek", "Start_lat", "Start_lon", "Bike number"]]

    # Convert all columns to numerical types
    X = X.apply(pd.to_numeric, errors="coerce")

    # Targets:
    y_lat = pd.to_numeric(df["End_lat"], errors="coerce")
    y_lon = pd.to_numeric(df["End_lon"], errors="coerce")

    y_station = df["End station"]  # Assuming this is the station ID

    # Split data into training and validation sets
    X_train, X_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val, y_station_train, y_station_val = train_test_split(
        X, y_lat, y_lon, y_station, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val, y_station_train, y_station_val


# --- Flower Client Class ---
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, path_to_data):
        X_train, X_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val, y_station_train, y_station_val = load_client_data(path_to_data)
        self.X_train, self.X_val = X_train.values, X_val.values
        self.y_lat_train, self.y_lat_val = y_lat_train.values, y_lat_val.values
        self.y_lon_train, self.y_lon_val = y_lon_train.values, y_lon_val.values
        self.true_stations_val = y_station_val.values

        # Fit scalers on training target values
        self.scaler_lat = StandardScaler().fit(self.y_lat_train.reshape(-1, 1))
        self.scaler_lon = StandardScaler().fit(self.y_lon_train.reshape(-1, 1))
        
        self.model = create_model_complex(input_shape=(self.X_train.shape[1],))

    def get_parameters(self, config):
        return ndarrays_to_parameters(self.model.get_weights())
    
    def get_true_stations(self):
        return self.true_stations_val

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
            epochs=20, batch_size=16, verbose=0
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

        # Check if evaluation set is empty to avoid returning a zero count.
        if len(self.X_val) == 0:
            print("Warning: No evaluation samples available. Returning dummy evaluation metrics.")
            return 0.0, 1, {"lat_mae": 0.0, "lon_mae": 0.0, "station_acc": 0.0, "avg_distance_error": 0.0}

        # Convert data to float32 if necessary
        X_val = self.X_val.astype(np.float32)
        y_lat_val = self.y_lat_val.astype(np.float32)
        y_lon_val = self.y_lon_val.astype(np.float32)

        # Get predictions (assume model outputs [lat, lon])
        preds = self.model.predict(X_val)
        pred_lat = preds[0].flatten()  # ensure 1D array
        pred_lon = preds[1].flatten()  # ensure 1D array

        # (We assume that the targets are unscaled; if not, remove inverse transformation if necessary)

        # Load station mapping data and build a mapping: station name -> (latitude, longitude)
        stations_df = pd.read_csv("data/stations_geolocated.csv")
        # Expected columns: 'name', 'latitude', 'longitude'
        stations_mapping = {row['name']: (row['latitude'], row['longitude']) for idx, row in stations_df.iterrows()}

        # Initialize lists to collect results
        predicted_stations = []
        distance_errors = []
        true_stations = self.get_true_stations()  # True station names (from your dataset)

        # For each prediction, compute the nearest station and the error distance compared to the true station
        for i, (pred_lat_val, pred_lon_val, true_station) in enumerate(zip(pred_lat, pred_lon, true_stations)):
            # Use your find_nearest_station function to get the predicted station name.
            predicted_station = find_nearest_station(pred_lat_val, pred_lon_val, stations_df)
            predicted_stations.append(predicted_station)

            # Look up the true station coordinates.
            if true_station in stations_mapping:
                true_coords = stations_mapping[true_station]
                error_distance = haversine(pred_lat_val, pred_lon_val, true_coords[0], true_coords[1])
                distance_errors.append(error_distance)
                print(f"Sample {i}: Predicted ({pred_lat_val:.4f}, {pred_lon_val:.4f}), "
                    f"Actual station {true_station} at ({true_coords[0]:.4f}, {true_coords[1]:.4f}), "
                    f"Distance error: {error_distance:.2f} meters")
            else:
                print(f"Warning: True station '{true_station}' not found in mapping!")
                distance_errors.append(np.nan)

        # Compute average distance error (ignoring any NaNs)
        avg_distance_error = np.nanmean(distance_errors)
        print(f"Average distance error: {avg_distance_error:.2f} meters")

        # Calculate station prediction accuracy by comparing predicted and true station names
        station_accuracy = np.mean(np.array(predicted_stations) == np.array(true_stations))
        print("Station prediction accuracy:", station_accuracy)

        # Optionally, compute the original regression metrics (losses) from the model evaluation.
        results = self.model.evaluate(X_val,
                                    {"lat": y_lat_val, "lon": y_lon_val},
                                    verbose=0)
        metrics = {"lat_mae": results[3], "lon_mae": results[4],
                "station_acc": station_accuracy, "avg_distance_error": avg_distance_error}
        return results[0], len(X_val), metrics


# --- Entry point to run this client individually (for testing only) ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_data.csv>")
        exit(1)

    path = sys.argv[1]
    fl.client.start_numpy_client(server_address="localhost:8080", client=FederatedClient(path))
