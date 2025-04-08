import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def inspect_client_data():

    df = pd.read_csv("data/bike_features.csv")


    print("\nDataframe head:\n", df.head())
    print("\nDataframe describe:\n", df.describe())
    print("\nDataframe info:\n", df.info())

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

    inspect_client_data()

    print("\n✅ Done. Check generated PNGs for distribution plots.")
