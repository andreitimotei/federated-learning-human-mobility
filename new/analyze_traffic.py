import os
import pandas as pd

# Simulated upload environment: use /mnt/data for uploaded files
data_dir = "processed-data/clients"
station_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

traffic_data = []

# Parse uploaded station files
for path in station_files:
    try:
        df = pd.read_csv(path)
        total_traffic = df["num_arrivals"].sum() + df["num_departures"].sum()
        station_id = os.path.basename(path).split(".")[0]
        traffic_data.append((station_id, total_traffic))
    except Exception as e:
        print(f"Failed to process {path}: {e}")

# Sort and display top 10
top_stations = sorted(traffic_data, key=lambda x: x[1], reverse=True)[:10]
top_stations_df = pd.DataFrame(top_stations, columns=["Station", "Total Traffic"])
print("Top 10 Stations by Traffic")
print(top_stations_df)
