import os
import pandas as pd
import json

def generate_station_mapping(client_folder, output_json):
    station_ids = set()

    for fname in os.listdir(client_folder):
        if fname.startswith("client_") and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(client_folder, fname))
            station_ids.update(df["End station number"].unique())

    station_ids = sorted(int(s) for s in station_ids)
    mapping = {str(station_id): idx for idx, station_id in enumerate(station_ids)}

    with open(output_json, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[✓] Saved station mapping to {output_json} ({len(mapping)} entries)")

if __name__ == "__main__":
    generate_station_mapping("data/clients", "data/station_mapping.json")
