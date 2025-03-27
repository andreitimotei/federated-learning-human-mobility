import pandas as pd

def engineer_features(input_csv, output_csv):
    # Load the merged dataset
    df = pd.read_csv(input_csv)

    # Convert dates to datetime objects
    df["Start date"] = pd.to_datetime(df["Start date"])
    df["End date"] = pd.to_datetime(df["End date"])

    # Feature: hour of day and day of week
    df["Start_hour"] = df["Start date"].dt.hour
    df["Start_dayofweek"] = df["Start date"].dt.dayofweek  # Monday = 0

    # Feature: trip duration in minutes
    df["Trip_duration_minutes"] = (df["End date"] - df["Start date"]).dt.total_seconds() / 60

    # Rename lat/lon columns for clarity
    df = df.rename(columns={
        "latitude": "Start_lat",
        "longitude": "Start_lon"
    })

    # Keep all useful columns
    df = df[[
        "Number",
        "Start date", "End date",
        "Start station number", "Start station",
        "End station number", "End station",
        "Bike number", "Bike model",
        "Total duration", "Total duration (ms)",
        "Start_lat", "Start_lon",
        "Start_hour", "Start_dayofweek",
        "Trip_duration_minutes"
    ]]

    # Save the processed dataset
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved engineered dataset to: {output_csv}")

if __name__ == "__main__":
    input_path = "data/bike_trips_with_geo.csv"
    output_path = "data/bike_features.csv"
    engineer_features(input_path, output_path)
