import subprocess
import sys

def run_script(script_name):
    print(f"Running {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)
    print(f"{script_name} completed successfully.\n")

if __name__ == "__main__":
    # 1. Merge geodata to create bike_trips_with_geo.csv
    run_script("preprocessing/merge_geodata.py")

    # 2. Engineer features to create bike_features.csv (includes recalculating Trip_duration_minutes)
    run_script("preprocessing/features.py")

    # 3. Run outlier filtering (produces bike_features_filtered_percentile.csv)
    run_script("preprocessing/feature_engineering.py")

    # 4. Split the filtered dataset into client files (data/clients folder)
    run_script("preprocessing/split_clients.py")

    # 5. Generate station mapping from client files (data/station_mapping.json)
    run_script("preprocessing/generate_station_mapping.py")

    print("All preprocessing steps completed successfully!")
