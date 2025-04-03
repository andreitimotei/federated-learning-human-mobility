import re
import pandas as pd

with open("output.txt", "r") as file:
    text = file.read()

# Extract metrics using regex
pattern = re.compile(
    r"round: (\d+) - duration_loss: ([\d.]+) - duration_mae: ([\d.]+) - destination_loss: ([\d.]+) - destination_accuracy: ([\d.]+)"
)
matches = pattern.findall(text)

# Save to CSV
df = pd.DataFrame(matches, columns=["round", "duration_loss", "duration_mae", "destination_loss", "destination_accuracy"])
df = df.astype({
    "round": int,
    "duration_loss": float,
    "duration_mae": float,
    "destination_loss": float,
    "destination_accuracy": float
})
df.to_csv("federated_training_metrics.csv", index=False)
print("✅ Metrics saved to federated_training_metrics.csv")
