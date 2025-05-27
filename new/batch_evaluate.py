# batch_evaluate_and_plot.py

import matplotlib
matplotlib.use("Agg")  # Add this line at the very top, before importing pyplot

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model.model import FedMLPLSTM
from client.client import StationDataset


def load_model(model_path, static_dim, lag_seq_len, device):
    model = FedMLPLSTM(static_dim=static_dim, exog_dim=0, lag_seq_len=lag_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_day(model, dataset, date, device):
    timestamps, preds, trues = [], [], []
    date = pd.to_datetime(date).date()
    for i in range(len(dataset)):
        (static_exog, lag_seq), y = dataset[i]
        dt = pd.to_datetime(dataset.X_static_exog[i][0] * 1000000000, unit="s", errors="coerce").date()
        if dt != date:
            continue
        with torch.no_grad():
            static_exog = torch.tensor(static_exog).unsqueeze(0).to(device)
            lag_seq = torch.tensor(lag_seq).unsqueeze(0).to(device)
            y_tensor = torch.tensor(y).unsqueeze(0).to(device)
            pred = model(static_exog, lag_seq).squeeze().cpu().numpy()
            # pred = np.round(pred).astype(int)  # <-- Add this line to round predictions
            actual = y_tensor.squeeze().cpu().numpy()
        timestamps.append(dt)
        preds.append(pred)
        trues.append(actual)

    if not preds:
        return None, None, None

    mae = sum(abs(p - t).mean() for p, t in zip(preds, trues)) / len(preds)
    return mae, preds, trues

def plot_predictions(preds, trues, station_id, date):
    pred_arr, pred_dep = zip(*preds)
    true_arr, true_dep = zip(*trues)
    n = len(pred_arr)
    hours = list(range(n))  # X axis: hour indices

    # Save only if at least one interval has demand > 5
    if not (
        any(x > 5 for x in pred_arr) or
        any(x > 5 for x in pred_dep) or
        any(x > 5 for x in true_arr) or
        any(x > 5 for x in true_dep)
    ):
        return  # Skip saving this plot

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hours, true_arr, label="True Arrivals")
    plt.plot(hours, pred_arr, label="Pred Arrivals")
    plt.legend()
    plt.title("Arrivals")
    plt.xlabel("Hour of Day")
    plt.xticks(hours if n <= 24 else range(0, n, max(1, n // 12)))  # reasonable ticks

    plt.subplot(1, 2, 2)
    plt.plot(hours, true_dep, label="True Departures")
    plt.plot(hours, pred_dep, label="Pred Departures")
    plt.legend()
    plt.title("Departures")
    plt.xlabel("Hour of Day")
    plt.xticks(hours if n <= 24 else range(0, n, max(1, n // 12)))

    plt.suptitle(f"Station {station_id} | {date}")
    fname = f"prediction_plot/accurate_plot_{station_id}_{date}.png"
    plt.savefig(fname)
    plt.close()
    print(f"[SAVED] {fname}")

def get_all_dates_from_csvs(csv_dir):
    all_dates = set()
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            path = os.path.join(csv_dir, file)
            try:
                df = pd.read_csv(path, usecols=["datetime"])
                # If datetime is already a UNIX timestamp, use unit="s"
                df["datetime"] = pd.to_datetime(df["datetime"], unit="s", errors="coerce")
                df = df.dropna(subset=["datetime"])
                unique_dates = df["datetime"].dt.date.unique()
                for d in unique_dates:
                    dt = pd.Timestamp(d)
                    unix_time = int(dt.timestamp())
                    # print(f"Date: {d} -> UNIX time: {unix_time}")
                all_dates.update(unique_dates)
            except Exception as e:
                print(f"Skipped {file}: {e}")
    return sorted(list(all_dates))

def batch_evaluate(model_path, csv_dir, mae_threshold=1.0, device="cuda"):
    all_dates = get_all_dates_from_csvs(csv_dir)
    for file in os.listdir(csv_dir):
        if not file.endswith(".csv"):
            continue
        station_path = os.path.join(csv_dir, file)
        station_id = os.path.splitext(file)[0]
        try:
            dataset = StationDataset(station_path)
            static_dim = dataset.X_static_exog.shape[1]
            lag_seq_len = dataset.X_lag_seq.shape[1]
            model = load_model(model_path, static_dim, lag_seq_len, device)

            for day in all_dates:
                mae, preds, trues = evaluate_day(model, dataset, day, device)
                print(f"[EVALUATED] {station_id} on {day}: MAE = {mae}")
                if mae is not None and mae < mae_threshold:
                    plot_predictions(preds, trues, station_id, day)
        except Exception as e:
            print(f"[ERROR] Failed on {file}: {e}")

if __name__ == "__main__":
    batch_evaluate(
        model_path="global_models/global_model_round_15.pt",
        csv_dir="processed-data/clients",
        mae_threshold=0.4,
        device="cuda"
    )
