import pandas as pd
import numpy as np
import os
from datetime import timedelta

def hourly_temperature(hour: int, t_min: float, t_max: float, t_mean: float) -> float:
    """
    Estimate hourly temperature based on daily min, max, and mean.
    Uses a sinusoidal model to ensure daily mean is honored,
    peaking at mid-afternoon and trough at early morning.

    Parameters:
    - hour: int hour of day [0-23]
    - t_min: float daily minimum temperature (°C)
    - t_max: float daily maximum temperature (°C)
    - t_mean: float daily mean temperature (°C)

    Returns:
    - Estimated temperature at given hour
    """
    # Sinusoidal parameters
    amplitude = (t_max - t_min) / 2.0
    # Phase shift to align peak at 15:00
    # We want sin(2π*(h - phase)/24) == 1 at h = max_hour
    max_hour = 15
    phase = max_hour - 6  # because sin reaches 1 at π/2 => (2π/24)*(h - phase) = π/2 => h - phase = 6

    # Compute angle
    angle = 2 * np.pi * (hour - phase) / 24
    temp = t_mean + amplitude * np.sin(angle)
    return temp


def expand_daily_to_hourly(input_path: str, output_path: str) -> None:
    """
    Read daily weather CSV, expand each row into 24 hourly entries,
    estimating temperature per hour based on daily min, max, and mean,
    retain daily min, max, mean, drop any quality columns,
    and carry other daily metrics unchanged.

    Parameters:
    - input_path: path to daily weather CSV for 2023
    - output_path: path to write expanded hourly weather CSV
    """
    # Read daily data
    df = pd.read_csv(input_path, parse_dates=['date'])

    # Drop any quality flag columns
    quality_cols = [c for c in df.columns if c.startswith('q_')]
    df.drop(columns=quality_cols, inplace=True)

    records = []
    for _, row in df.iterrows():
        base_date = row['date']
        t_min = row['min_temp']
        t_max = row['max_temp']
        t_mean = row['mean_temp']
        # Prepare other daily metrics once
        daily_metrics = row.drop(labels=['date', 'min_temp', 'max_temp', 'mean_temp']).to_dict()

        for hour in range(24):
            dt = base_date + timedelta(hours=hour)
            temp = hourly_temperature(hour, t_min, t_max, t_mean)
            record = {
                'datetime': dt,
                'temperature': temp,
                'min_temp': t_min,
                'max_temp': t_max,
                'mean_temp': t_mean
            }
            # include other daily metrics
            record.update(daily_metrics)
            records.append(record)

    # Build DataFrame
    df_hourly = pd.DataFrame(records)
    df_hourly.sort_values('datetime', inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_hourly.to_csv(output_path, index=False)
    print(f"Expanded to {len(df_hourly)} hourly records and saved to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_csv = os.path.join(base_dir, 'processed-data', 'london_weather_2023.csv')
    output_csv = os.path.join(base_dir, 'processed-data', 'london_weather_2023_hourly.csv')
    expand_daily_to_hourly(input_csv, output_csv)
