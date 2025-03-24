import pandas as pd

def merge_station_geodata(trip_csv, station_csv, output_csv):
    # Load datasets
    df_trips = pd.read_csv(trip_csv)
    df_stations = pd.read_csv(station_csv)

    # Merge on start station name
    df_merged = pd.merge(
        df_trips,
        df_stations,
        how="left",
        left_on="Start station",
        right_on="name"
    )

    # Drop redundant column
    df_merged.drop(columns=["name"], inplace=True)

    # Save to file
    df_merged.to_csv(output_csv, index=False)
    print(f"[✓] Saved merged dataset to: {output_csv}")

if __name__ == "__main__":
    trip_csv = "data/LondonBikeJourneyAug2023.csv"
    station_csv = "data/stations_geolocated.csv"
    output_csv = "data/bike_trips_with_geo.csv"

    merge_station_geodata(trip_csv, station_csv, output_csv)
