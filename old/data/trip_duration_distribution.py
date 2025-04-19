import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/bike_features.csv")
durations = df["Trip_duration_minutes"].dropna()

# Compute summary statistics
mean_val = durations.mean()
median_val = durations.median()
std_val = durations.std()

# Create a figure with two subplots (histogram on top, boxplot below)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Top subplot: Histogram using a logarithmic x-axis
ax1.hist(durations, bins=50, edgecolor='k', alpha=0.7)
ax1.set_xscale('log')
ax1.set_xlabel("Trip Duration (minutes, log scale)")
ax1.set_ylabel("Frequency")
ax1.set_title("Histogram of Trip Durations (Log Scale)")
ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_val:.2f}")
ax1.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f"Median: {median_val:.2f}")
ax1.legend()

# Bottom subplot: Boxplot using a logarithmic x-axis
ax2.boxplot(durations, vert=False)
ax2.set_xscale('log')
ax2.set_xlabel("Trip Duration (minutes, log scale)")
ax2.set_title("Boxplot of Trip Durations (Log Scale)")

plt.tight_layout()
plt.savefig("trip_duration_plot.png")
plt.close()
