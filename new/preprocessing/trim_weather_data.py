import pandas as pd
import os

def trim_and_clean_weather_data(input_path: str, output_path: str) -> None:
    """
    Load the raw weather CSV, rename columns to human-friendly names,
    filter to only rows in 2023, impute suspect values (q==1) via interpolation,
    convert temperatures from 0.1°C to °C, and save to output_path.

    Parameters:
    - input_path: path to the full weather CSV file (1970-2023)
    - output_path: path to write the processed 2023 weather CSV
    """
    # Read input CSV
    df = pd.read_csv(input_path)

    # Rename data columns
    rename_map = {
        'DATE': 'date',               # Date in YYYYMMDD
        'TX': 'max_temp',             # Daily max temperature (0.1°C)
        'TN': 'min_temp',             # Daily min temperature (0.1°C)
        'TG': 'mean_temp',            # Daily mean temperature (0.1°C)
        'SS': 'sun_duration',         # Sunshine duration (0.1 h)
        'SD': 'snow_depth',           # Snow depth (cm)
        'RR': 'precipitation',        # Precipitation amount (0.1 mm)
        'QQ': 'solar_radiation',      # Global radiation (W/m²)
        'PP': 'pressure',             # Sea level pressure (0.1 hPa)
        'HU': 'humidity',             # Relative humidity (%)
        'CC': 'cloud_cover'           # Cloud cover (oktas)
    }
    # Rename quality flags
    quality_map = {f"Q_{k}": f"q_{v}" for k, v in rename_map.items() if k != 'DATE'}
    df.rename(columns={**rename_map, **quality_map}, inplace=True)

    # Parse the date column and filter to 2023
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df_2023 = df[df['date'].dt.year == 2023].copy()

    # Sort by date and set as index for interpolation
    df_2023.sort_values('date', inplace=True)
    df_2023.set_index('date', inplace=True)

    # Impute suspect data (q == 1) by linear interpolation
    metrics = [
        'max_temp','min_temp','mean_temp','sun_duration','snow_depth',
        'precipitation','solar_radiation','pressure','humidity','cloud_cover'
    ]
    for col in metrics:
        q_col = 'q_' + col
        if q_col in df_2023.columns:
            # Mask suspect values
            df_2023.loc[df_2023[q_col] == 1, col] = pd.NA
            # Interpolate missing days (forward/backward)
            df_2023[col] = df_2023[col].interpolate(method='linear', limit_direction='both')

    # Convert temperatures from 0.1°C to °C
    for temp_col in ['max_temp', 'min_temp', 'mean_temp']:
        if temp_col in df_2023.columns:
            df_2023[temp_col] = df_2023[temp_col] / 10.0

    # Reset index
    df_2023.reset_index(inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed data
    df_2023.to_csv(output_path, index=False)
    print(f"Processed and saved {len(df_2023)} daily records for 2023 to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_csv = os.path.join(base_dir, 'raw-data', 'london_weather_data_1979_to_2023.csv')
    output_csv = os.path.join(base_dir, 'processed-data', 'london_weather_2023.csv')
    trim_and_clean_weather_data(input_csv, output_csv)
