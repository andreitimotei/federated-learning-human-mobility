import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def inspect_client_data(client_data_path):
    print(f"\nInspecting client data at: {client_data_path}")
    df = pd.read_csv(client_data_path)

    print("\nDataframe shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())

    if 'duration' in df.columns:
        print("\nTrip Duration Stats:")
        print(df['duration'].describe())

        plt.figure(figsize=(10, 4))
        sns.histplot(df['duration'], bins=50, kde=True)
        plt.title("Trip Duration Distribution")
        plt.xlabel("Duration (seconds)")
        plt.savefig("duration_distribution.png")
        plt.close()

    if 'destination' in df.columns:
        print("\nDestination Frequency:")
        print(df['destination'].value_counts().head(10))

        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x='destination', order=df['destination'].value_counts().index[:20])
        plt.title("Top 20 Destination Stations")
        plt.xlabel("Destination Station ID")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("destination_distribution.png")
        plt.close()

if __name__ == "__main__":
    data_folder = "data/clients"
    client_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith("client_") and f.endswith(".csv")])

    # Only inspect the first 3 clients for now
    for path in client_files[:3]:
        inspect_client_data(path)

    print("\n✅ Done. Check generated PNGs for distribution plots.")
