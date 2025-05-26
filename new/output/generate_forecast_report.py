import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
from model import FedMLPLSTM
from client import StationDataset

def load_model(model_path, static_dim, lag_seq_len, device='cpu'):
    model = FedMLPLSTM(static_dim=static_dim, exog_dim=0, lag_seq_len=lag_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def plot_predictions(df, predictions, day, station_id):
    actual = df[df["date"] == day][["num_arrivals", "num_departures"]].values
    predicted = predictions

    hours = list(range(len(actual)))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hours, actual[:, 0], label="Actual Arrivals")
    plt.plot(hours, predicted[:, 0], label="Predicted Arrivals")
    plt.title(f"Arrivals - Station {station_id} on {day}")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hours, actual[:, 1], label="Actual Departures")
    plt.plot(hours, predicted[:, 1], label="Predicted Departures")
    plt.title(f"Departures - Station {station_id} on {day}")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(csv_path, model_path, day, station_id, device='cpu'):
    ds = StationDataset(csv_path)
    model = load_model(model_path, ds.X_static_exog.shape[1], ds.X_lag_seq.shape[1], device=device)

    inputs = []
    labels = []
    timestamps = []

    for i in range(len(ds)):
        (static_exog, lag_seq), y = ds[i]
        static_exog = torch.tensor(static_exog).unsqueeze(0).to(device)
        lag_seq = torch.tensor(lag_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(static_exog, lag_seq).cpu().numpy().flatten()
            inputs.append(pred)
            labels.append(y)
            timestamps.append(ds.X_static_exog[i][0])  # datetime timestamp

    results_df = pd.DataFrame(inputs, columns=["pred_arrivals", "pred_departures"])
    results_df["actual_arrivals"] = [v[0] for v in labels]
    results_df["actual_departures"] = [v[1] for v in labels]
    results_df["timestamp"] = pd.to_datetime(timestamps, unit='s')
    results_df["date"] = results_df["timestamp"].dt.date
    results_df["hour"] = results_df["timestamp"].dt.hour

    # Filter for specified day
    day_str = pd.to_datetime(day).date()
    day_df = results_df[results_df["date"] == day_str]

    # Compute and print error metrics
    mae_arr = (day_df["pred_arrivals"] - day_df["actual_arrivals"]).abs().mean()
    mae_dep = (day_df["pred_departures"] - day_df["actual_departures"]).abs().mean()
    print(f"MAE on {day} for Station {station_id}: Arrivals = {mae_arr:.2f}, Departures = {mae_dep:.2f}")

    plot_predictions(day_df, day_df[["pred_arrivals", "pred_departures"]].values, day_str, station_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to station CSV file")
    parser.add_argument("--model", required=True, help="Path to trained model .pt file")
    parser.add_argument("--day", required=True, help="Date (YYYY-MM-DD) to evaluate")
    parser.add_argument("--station", required=True, help="Station ID")
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()

    main(args.csv, args.model, args.day, args.station, device=args.device)
