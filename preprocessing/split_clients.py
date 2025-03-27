import pandas as pd
import os

def split_into_clients(input_csv, output_folder, min_samples=100):
    df = pd.read_csv(input_csv)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Group by Start station number
    grouped = df.groupby("Start station number")

    client_count = 0
    for station_id, group in grouped:
        if len(group) < min_samples:
            continue  # Skip stations with too few samples

        client_filename = f"client_{station_id}.csv"
        group.to_csv(os.path.join(output_folder, client_filename), index=False)
        client_count += 1

    print(f"[✓] Saved {client_count} client datasets to {output_folder}")

if __name__ == "__main__":
    input_path = "data/bike_features.csv"
    output_folder = "data/clients"
    split_into_clients(input_path, output_folder)
