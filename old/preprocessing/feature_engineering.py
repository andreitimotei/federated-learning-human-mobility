import pandas as pd
import geohash2 as gh
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load your dataset
df = pd.read_csv("data/bike_features.csv")

# Extract the trip durations
durations = df["Trip_duration_ms"]

# Calculate the 99th percentile threshold for trip duration
upper_threshold = durations.quantile(0.99)

print(f"99th percentile (upper threshold): {upper_threshold}")

# Filter out trips that exceed this threshold
df = df[durations <= upper_threshold]



# 1) Geohash for start coords (precision=6 gives ~610 m grid)
df['Start_geohash'] = df.apply(
    lambda row: gh.encode(row['Start_lat'], row['Start_lon'], precision=6), axis=1
)
# label‐encode geohashes to ints
geohash_le = LabelEncoder()
df['Start_geohash_enc'] = geohash_le.fit_transform(df['Start_geohash'])

# 2) Per‐station historical stats
station_stats = df.groupby('Start station')['Trip_duration_minutes'] \
                    .agg(['mean','std']) \
                    .rename(columns={'mean':'station_avg_dur','std':'station_std_dur'})
df = df.merge(station_stats, left_on='Start station', right_index=True)

# 3) Most common destination per start station
top_dest = df.groupby('Start station')['End station'] \
                .agg(lambda x: x.value_counts().idxmax()) \
                .rename('station_top_dest')
df = df.merge(top_dest, left_on='Start station', right_index=True)
dest_le = LabelEncoder()
df['station_top_dest_enc'] = dest_le.fit_transform(df['station_top_dest'])

# 4) Add them to your feature list
feature_cols = [
    "Start_date", "End_date", "Start_hour", "Start_dayofweek", "End_hour", "End_dayofweek", "Start_lat", "Start_lon", "Trip_distance_m", "Trip_duration_ms",
    # … your old features …,
    'Start_geohash_enc',
    'station_avg_dur',
    'station_std_dur',
    'station_top_dest_enc',
]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 5) Keep them in final schema
cols_to_keep = [
    "Number", "Start_date", "End_date",
    "Start_hour", "Start_dayofweek", "End_hour", "End_dayofweek",
    "Start station number", "Start station",
    "End station number", "End station",
    "Bike number", "Bike model",
    "Trip_duration_minutes", "Trip_duration_ms",
    "Trip_distance_m",
    "Start_lat", "Start_lon",
    "End_lat", "End_lon", "Trip_distance_m",
    # … your old cols …,
    'Start_geohash_enc',
    'station_avg_dur',
    'station_std_dur',
    'station_top_dest_enc',
]
# plus your existing ones
df = df[cols_to_keep]


# Save the filtered dataset
df = df.reset_index(drop=True)

# Save or inspect the filtered dataset
df.to_csv("data/bike_features_filtered_percentile.csv", index=False)
print(f"Original dataset size: {df.shape[0]}, Filtered dataset size: {df.shape[0]}")

# Save the label encoders for geohash and destination
import pickle

with open("data/geohash_le.pkl", "wb") as f:
    pickle.dump(geohash_le, f)

with open("data/dest_le.pkl", "wb") as f:
    pickle.dump(dest_le, f)
