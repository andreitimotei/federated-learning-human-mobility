import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_station(file_path, holiday_list=None):
    """
    Preprocess a single station CSV for federated learning:
      - Parses dates, reindexes hourly (imputes missing timestamps)
      - Imputes static and weather features
      - Generates temporal features (hour, dow, weekend, holiday)
      - Creates lag features for num_departures & num_arrivals
      - Normalizes weather variables
      - Converts datetime to Unix time
      - Drops non-numeric columns like station_name
    Args:
      file_path (str): Path to station CSV
      holiday_list (list[str], optional): List of dates (YYYY-MM-DD) to flag as holidays
    Returns:
      pd.DataFrame: Feature-engineered dataframe ready for modeling
    """
    # Load and index
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    
    # Ensure continuous hourly index
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    df = df.reindex(idx)
    
    # Impute static columns
    static_cols = ['terminal', 'station_name', 'lat', 'lon', 'nbDocks']
    for col in static_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Define weather columns
    weather_cols = [
        'temperature', 'min_temp', 'max_temp', 'mean_temp',
        'sun_duration', 'snow_depth', 'precipitation',
        'solar_radiation', 'pressure', 'humidity', 'cloud_cover'
    ]
    # Impute weather with interpolation + ffill/bfill
    df[weather_cols] = df[weather_cols].interpolate().ffill().bfill()
    
    # Fill missing counts with zero
    count_cols = ['num_departures', 'num_arrivals']
    df[count_cols] = df[count_cols].fillna(0)

    # Mark spike hours
    df["is_arrival_spike"] = (df["num_arrivals"] > df["num_arrivals"].quantile(0.95)).astype(int)
    df["is_departure_spike"] = (df["num_departures"] > df["num_departures"].quantile(0.95)).astype(int)


    df["rolling_arrivals_3h"] = df["num_arrivals"].rolling(3).sum()
    df["rolling_departures_3h"] = df["num_departures"].rolling(3).sum()
    df[["rolling_arrivals_3h", "rolling_departures_3h"]] = df[["rolling_arrivals_3h", "rolling_departures_3h"]].fillna(0)

    
    # Temporal features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    if holiday_list:
        holidays = pd.to_datetime(holiday_list)
        df['is_holiday'] = df.index.normalize().isin(holidays).astype(int)
    else:
        df['is_holiday'] = 0
    
    # Lagged counts
    lags = [1, 2, 3, 24, 168]
    for lag in lags:
        for col in count_cols:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Drop early rows missing lag data
    df = df.dropna(subset=[f'{col}_lag1' for col in count_cols])
    
    # Normalize weather features
    scaler = StandardScaler()
    df[weather_cols] = scaler.fit_transform(df[weather_cols])
    
    # Convert datetime to Unix time
    df = df.reset_index().rename(columns={'index': 'datetime'})
    df['datetime'] = df['datetime'].apply(lambda x: x.timestamp() if pd.notnull(x) else None)
    
    # Drop non-numeric columns like station_name
    if 'station_name' in df.columns:
        df = df.drop(columns=['station_name'])
    
    return df

def preprocess_all_stations(client_dir, holiday_list=None):
    """
    Preprocess all client files in the specified directory.
    Args:
      client_dir (str): Path to the directory containing client CSV files.
      holiday_list (list[str], optional): List of dates (YYYY-MM-DD) to flag as holidays.
    """
    # Ensure the directory exists
    if not os.path.exists(client_dir):
        print(f"Directory {client_dir} does not exist.")
        return

    # Process each client file
    for file_name in os.listdir(client_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(client_dir, file_name)
            print(f"Processing {file_name}...")
            try:
                # Preprocess the station data
                processed_df = preprocess_station(file_path, holiday_list)

                # Save the processed file with the same name
                processed_df.to_csv(file_path, index=False)
                print(f"Processed and saved: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    # Directory containing client files
    CLIENT_DIR = "processed-data/clients"

    # Optional: List of holidays
    HOLIDAY_LIST = ["2023-01-02", "2023-01-02", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-29", "2023-08-28", "2023-12-25", "2023-12-26"]  # Example holidays

    # Preprocess all client files
    preprocess_all_stations(CLIENT_DIR, HOLIDAY_LIST)

