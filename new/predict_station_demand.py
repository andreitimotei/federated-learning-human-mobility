import argparse
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
from model.model import FedMLPLSTM
from client.client import StationDataset

def load_model(model_path, static_dim, lag_seq_len, device):
    model = FedMLPLSTM(static_dim=static_dim, exog_dim=0, lag_seq_len=lag_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_station_info(csv_path):
    # Try to extract station id from filename
    basename = os.path.basename(csv_path)
    station_id = basename.split("_")[1].split(".")[0] if "station_" in basename else "Unknown"
    # Try to extract station name from CSV if available
    try:
        df = pd.read_csv(csv_path, nrows=1)
        if "station_name" in df.columns:
            station_name = df["station_name"].iloc[0]
        else:
            station_name = "Unknown"
    except Exception:
        station_name = "Unknown"
    return station_id, station_name

def predict_for_day(model, dataset, target_date, device):
    timestamps, preds, trues = [], [], []
    for i in range(len(dataset)):
        (static_exog, lag_seq), y = dataset[i]
        dt = pd.to_datetime(dataset.X_static_exog[i][0] * 1000000000, unit="s")
        target_dt = pd.to_datetime(target_date).date()
        if dt.date() != target_dt:
            continue

        with torch.no_grad():
            static_exog = torch.tensor(static_exog).unsqueeze(0).to(device)
            lag_seq = torch.tensor(lag_seq).unsqueeze(0).to(device)
            y_tensor = torch.tensor(y).unsqueeze(0).to(device)

            pred = model(static_exog, lag_seq).squeeze().cpu().numpy()
            pred = np.round(pred).astype(int)
            target = y_tensor.squeeze().cpu().numpy()

        timestamps.append(dt)
        preds.append(pred)
        trues.append(target)
    return timestamps, preds, trues

def plot_predictions(timestamps, preds, trues, target_date, station_id, station_name):
    if not preds or not trues:
        print(f"[WARNING] No predictions found for {target_date}. Check the date format or dataset coverage.")
        return

    pred_arr, pred_dep = zip(*preds)
    true_arr, true_dep = zip(*trues)

    timestamps = pd.to_datetime(timestamps)

    plt.figure(figsize=(14, 6))
    plt.suptitle(f"Station {station_id} - {station_name} | {target_date}", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(timestamps, true_arr, label="Actual Arrivals")
    plt.plot(timestamps, pred_arr, label="Predicted Arrivals")
    plt.title("Arrivals")
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(timestamps, true_dep, label="Actual Departures")
    plt.plot(timestamps, pred_dep, label="Predicted Departures")
    plt.title("Departures")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"prediction_plot_{station_id}_{target_date}_new.png")
    print(f"[INFO] Plot saved to prediction_plot_{station_id}_{target_date}_new.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to station CSV")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--day", required=True, help="Date to evaluate (YYYY-MM-DD)")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    args = parser.parse_args()

    ds = StationDataset(args.csv)
    static_dim = ds.X_static_exog.shape[1]
    lag_seq_len = ds.X_lag_seq.shape[1]

    station_id, station_name = extract_station_info(args.csv)

    model = load_model(args.model, static_dim, lag_seq_len, args.device)
    timestamps, preds, trues = predict_for_day(model, ds, args.day, args.device)
    plot_predictions(timestamps, preds, trues, args.day, station_id, station_name)

if __name__ == "__main__":
    main()
