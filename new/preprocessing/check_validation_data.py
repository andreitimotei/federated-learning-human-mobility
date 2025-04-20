import pandas as pd

# Load the dataset
global_val_csv = "processed-data/global_validation.csv"
df = pd.read_csv(global_val_csv)

# Check the data types of each column
print(df.dtypes)

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
print(f"Non-numeric columns: {non_numeric_cols}")