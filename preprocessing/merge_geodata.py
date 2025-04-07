import pandas as pd

def merge_station_geodata(trip_csv, station_csv, output_csv):
    # Load datasets
    df_trips = pd.read_csv(trip_csv)
    df_stations = pd.read_csv(station_csv)

    # Merge geodata for the start station
    # Prefix station columns with "Start_"
    df_stations_start = df_stations.rename(columns={
        "name": "Start_station_name",
        "latitude": "Start_lat",
        "longitude": "Start_lon"
    })
    df_trips = pd.merge(
        df_trips, df_stations_start,
        how="left", left_on="Start station", right_on="Start_station_name"
    )

    # Merge geodata for the end station
    # Prefix station columns with "End_"
    df_stations_end = df_stations.rename(columns={
        "name": "End_station_name",
        "latitude": "End_lat",
        "longitude": "End_lon"
    })
    df_trips = pd.merge(
        df_trips, df_stations_end,
        how="left", left_on="End station", right_on="End_station_name"
    )

    # Optionally, drop the redundant station name columns
    df_trips.drop(columns=["Start_station_name", "End_station_name"], inplace=True)

    # Save merged dataset
    df_trips.to_csv(output_csv, index=False)
    print(f"[✓] Saved merged dataset to: {output_csv}")


if __name__ == "__main__":
    trip_csv = "data/LondonBikeJourneyAug2023.csv"
    station_csv = "data/stations_geolocated.csv"
    output_csv = "data/bike_trips_with_geo.csv"

    merge_station_geodata(trip_csv, station_csv, output_csv)
