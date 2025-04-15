import pandas as pd
import numpy as np
import math

def haversine(lat1, lon1, lat2, lon2): 
    # Earth radius in meters 
    R = 6378137.0 
    phi1 = math.radians(lat1) 
    phi2 = math.radians(lat2) 
    dphi = math.radians(lat2 - lat1) 
    dlambda = math.radians(lon2 - lon1) 
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2 
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def engineer_features(input_csv, output_csv):
    # Load the merged dataset
    df = pd.read_csv(input_csv)

    # Convert dates to datetime objects
    df["Start date"] = pd.to_datetime(df["Start date"])
    df["End date"] = pd.to_datetime(df["End date"])

    # Extract features: hour of day and day of week
    df["Start_hour"] = df["Start date"].dt.hour
    df["Start_dayofweek"] = df["Start date"].dt.dayofweek  # Monday=0

    # Extract features: hour of day and day of week
    df["End_hour"] = df["End date"].dt.hour
    df["End_dayofweek"] = df["End date"].dt.dayofweek  # Monday=0


    # Compute trip duration in minutes if not already available
    df["Trip_duration_minutes"] = (df["End date"] - df["Start date"]).dt.total_seconds() / 60
    df["Trip_duration_ms"] = df["Total duration (ms)"]

    # Convert dates to datetime objects
    df["Start_date"] = pd.to_datetime(df["Start date"]).astype(np.int64) // 10**9  # Convert to Unix timestamp in seconds
    df["End_date"] = pd.to_datetime(df["End date"]).astype(np.int64) // 10**9  # Convert to Unix timestamp in seconds

    # Rename latitude and longitude columns for clarity if needed
    # (Assuming the merged file already has Start_lat, Start_lon from merge_geodata.py)

    # Compute trip distance if end station coordinates exist
    if "End_lat" in df.columns and "End_lon" in df.columns:
        df["Trip_distance_m"] = df.apply(
            lambda row: haversine(row["Start_lat"], row["Start_lon"], row["End_lat"], row["End_lon"]),
            axis=1
        )

    # Handle missing values: fill numeric columns with median
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Optionally, select a subset of useful columns
    cols_to_keep = [
        "Number", "Start_date", "End_date",
        "Start_hour", "Start_dayofweek", "End_hour", "End_dayofweek",
        "Start station number", "Start station",
        "End station number", "End station",
        "Bike number", "Bike model",
        "Trip_duration_minutes", "Trip_duration_ms",
        "Trip_distance_m",
        "Start_lat", "Start_lon"
    ]

    from sklearn.preprocessing import StandardScaler

    feature_cols = ["Start_date", "End_date", "Start_hour", "Start_dayofweek", "End_hour", "End_dayofweek", "Start_lat", "Start_lon", "Trip_distance_m", "Trip_duration_ms"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # If available, include end station coordinates and trip distance
    if "End_lat" in df.columns and "End_lon" in df.columns:
        cols_to_keep += ["End_lat", "End_lon", "Trip_distance_m"]
    df = df[cols_to_keep]

    # Save the processed dataset
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved engineered dataset to: {output_csv}")


if __name__ == "__main__":
    input_path = "data/bike_trips_with_geo.csv"
    output_path = "data/bike_features.csv"
    engineer_features(input_path, output_path)
