import pandas as pd

# Load your dataset
df = pd.read_csv("data/bike_features.csv")

# Extract the trip durations
durations = df["Trip_duration_minutes"]

# Calculate the 99th percentile threshold for trip duration
upper_threshold = durations.quantile(0.99)
print(f"99th percentile (upper threshold): {upper_threshold}")

# Filter out trips that exceed this threshold
df_filtered = df[durations <= upper_threshold]

# Save or inspect the filtered dataset
df_filtered.to_csv("data/bike_features_filtered_percentile.csv", index=False)
print(f"Original dataset size: {df.shape[0]}, Filtered dataset size: {df_filtered.shape[0]}")
