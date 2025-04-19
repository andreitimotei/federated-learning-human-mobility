import pandas as pd
import os
import json

def split_into_clients(input_csv, output_folder, min_samples=100):
    df = pd.read_csv(input_csv)

    # Load station index mapping
    with open("data/station_mapping.json", "r") as f:
        station_to_index = json.load(f)

    # Create proper columns for training
    df["duration"] = df["Trip_duration_minutes"]
    df["destination"] = df["End station number"].astype(str).map(station_to_index)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Group by Start station number
    grouped = df.groupby("Start station number")

    client_count = 0
    for station_id, group in grouped:
        if len(group) < min_samples:
            continue

        client_filename = f"client_{station_id}.csv"
        group.to_csv(os.path.join(output_folder, client_filename), index=False)
        client_count += 1

    print(f"[✓] Saved {client_count} client datasets to {output_folder}")

if __name__ == "__main__":
    input_path = "data/bike_features_filtered_percentile.csv"
    output_folder = "data/clients"
    split_into_clients(input_path, output_folder)
