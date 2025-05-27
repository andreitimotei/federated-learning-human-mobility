import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Directory containing client CSV files
CLIENT_DIR = "processed-data/clients"
GLOBAL_VAL_PATH = "processed-data/global_validation.csv"

# Fraction of data to reserve for global validation
GLOBAL_VAL_FRACTION = 0.1

REQUIRED_COLUMNS = [
    "is_arrival_spike", "is_departure_spike",
    "rolling_arrivals_3h", "rolling_departures_3h"
]

def create_global_validation():
    global_val_data = []

    # Iterate through all client CSV files
    for file_name in os.listdir(CLIENT_DIR):
        if file_name.endswith(".csv"):
            file_path = os.path.join(CLIENT_DIR, file_name)
            print(f"Processing {file_name}...")

            # Load client dataset
            df = pd.read_csv(file_path)

            # Add missing engineered features with default values
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    print(f"[WARN] {col} missing in {file_name}, filling with 0.")
                    df[col] = 0

            # Handle NaN values in the client dataset
            df = df.dropna()

            if len(df) < 2:
                print(f"Skipping {file_name} (not enough rows to split).")
                continue

            # Split into global validation and remaining data
            train_data, val_data = train_test_split(df, test_size=GLOBAL_VAL_FRACTION, random_state=42)

            # Save the remaining data back to the client file
            train_data.to_csv(file_path, index=False)

            # Append the reserved validation data to the global validation list
            global_val_data.append(val_data)

    # Combine all reserved validation data into a single DataFrame
    if global_val_data:
        global_val_df = pd.concat(global_val_data, ignore_index=True)
        global_val_df.to_csv(GLOBAL_VAL_PATH, index=False)
        print(f"Global validation dataset saved to {GLOBAL_VAL_PATH}")
    else:
        print("No global validation data created (no valid rows found).")

def clean_global_validation():
    df = pd.read_csv(GLOBAL_VAL_PATH)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["datetime"] = df["datetime"].apply(lambda x: x.timestamp() if pd.notnull(x) else None)

    if "station_name" in df.columns:
        df = df.drop(columns=["station_name"])

    df = df.dropna()
    df.to_csv(GLOBAL_VAL_PATH, index=False)
    print(f"Cleaned dataset saved to {GLOBAL_VAL_PATH}")

if __name__ == "__main__":
    create_global_validation()
    clean_global_validation()
