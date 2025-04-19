import pandas as pd
import os
import glob

def merge_journey_data(input_dir: str, output_path: str) -> None:
    """
    Merge multiple TfL journey CSV extracts into one sorted dataset.

    - Reads all CSV files in input_dir matching '*JourneyDataExtract*.csv'
    - Detects and parses start/end date columns
    - Concatenates into a single DataFrame
    - Sorts by the start datetime
    - Outputs to output_path

    Parameters:
    - input_dir: directory containing individual journey CSV files
    - output_path: path to write the merged, sorted CSV
    """
    # Build file list
    pattern = os.path.join(input_dir, '*JourneyDataExtract*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")

    merged_dfs = []
    for file in files:
        print(f"Reading {file}")
        # Read with low_memory=False to avoid mixed-type dtype warnings
        df = pd.read_csv(file, low_memory=False)

        # Detect start/end date columns by name
        cols_lower = {c.lower(): c for c in df.columns}
        start_col = None
        end_col = None
        for lower, orig in cols_lower.items():
            if 'start' in lower and 'date' in lower:
                start_col = orig
            elif 'end' in lower and 'date' in lower:
                end_col = orig
        if not start_col:
            raise ValueError(f"Start date column not found in {file}. Found columns: {df.columns.tolist()}")
        if not end_col:
            raise ValueError(f"End date column not found in {file}. Found columns: {df.columns.tolist()}")

        # Parse datetime columns
        df[start_col] = pd.to_datetime(df[start_col])
        df[end_col] = pd.to_datetime(df[end_col])

        merged_dfs.append(df)

    # Concatenate all
    merged = pd.concat(merged_dfs, ignore_index=True)

    # Sort by start date
    merged.sort_values(start_col, inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save merged dataset
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(files)} files into {len(merged)} records at {output_path}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_dir = os.path.join(base_dir, 'raw-data')
    output_csv = os.path.join(base_dir, 'processed-data', 'journey_data_2023.csv')
    merge_journey_data(input_dir, output_csv)
