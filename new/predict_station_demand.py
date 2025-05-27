import os
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from model.model import FedMLPLSTM
from client.client import StationDataset

# ==== Set your parameters here ====
STATION_CSV = "processed-data/clients/station_1075.csv"
MODEL_PATH = "global_models/global_model_round_24.pt"
TARGET_DATE = "2023-10-14"
DEVICE = "cuda"  # or "cpu"
# ==================================

def load_model(model_path, static_dim, lag_seq_len, device):
    model = FedMLPLSTM(static_dim=static_dim, exog_dim=0, lag_seq_len=lag_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_station_info(csv_path):
    basename = os.path.basename(csv_path)
    station_id = basename.split("_")[1].split(".")[0] if "station_" in basename else "Unknown"
    try:
        df = pd.read_csv(csv_path, nrows=1)
        if "station_name" in df.columns:
            station_name = df["station_name"].iloc[0]
        else:
            station_name = "Unknown"
    except Exception:
        station_name = "Unknown"
    return station_id, station_name

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
            pred = np.round(pred).astype(int)
            pred = np.clip(pred, 0, None)
            actual = y_tensor.squeeze().cpu().numpy()
        timestamps.append(dt)
        preds.append(pred)
        trues.append(actual)
    if not preds:
        print(f"[WARNING] No predictions found for {date}. Check the date format or dataset coverage.")
        return None, None, None
    mae = sum(abs(p - t).mean() for p, t in zip(preds, trues)) / len(preds)
    return mae, preds, trues

def plot_predictions(preds, trues, station_id, station_name, date):
    if not preds or not trues:
        print(f"[WARNING] No predictions to plot for {date}.")
        return

    pred_arr, pred_dep = zip(*preds)
    true_arr, true_dep = zip(*trues)
    hours = list(range(24))
    def pad_to_24(x):
        x = list(x)
        if len(x) < 24:
            x += [0] * (24 - len(x))
        return x[:24]
    pred_arr = pad_to_24(pred_arr)
    pred_dep = pad_to_24(pred_dep)
    true_arr = pad_to_24(true_arr)
    true_dep = pad_to_24(true_dep)

    hour_labels = [f"{h:02d}:00" if h % 3 == 0 else "" for h in hours]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hours, true_arr, label="Actual Arrivals")
    plt.plot(hours, pred_arr, label="Predicted Arrivals")
    plt.legend()
    plt.title("Arrivals")
    plt.xlabel("Hour of Day")
    plt.xticks(hours, hour_labels, rotation=45)

    plt.subplot(1, 2, 2)
    plt.plot(hours, true_dep, label="Actual Departures")
    plt.plot(hours, pred_dep, label="Predicted Departures")
    plt.legend()
    plt.title("Departures")
    plt.xlabel("Hour of Day")
    plt.xticks(hours, hour_labels, rotation=45)

    plt.suptitle(f"Station {station_id} - {station_name} | {date}")
    fname = f"prediction_plot_{station_id}_{date}_new.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fname)
    plt.close()
    print(f"[INFO] Plot saved to {fname}")

def main():
    ds = StationDataset(STATION_CSV)
    static_dim = ds.X_static_exog.shape[1]
    lag_seq_len = ds.X_lag_seq.shape[1]
    station_id, station_name = extract_station_info(STATION_CSV)
    model = load_model(MODEL_PATH, static_dim, lag_seq_len, DEVICE)
    mae, preds, trues = evaluate_day(model, ds, TARGET_DATE, DEVICE)
    print(f"[EVALUATED] {station_id} on {TARGET_DATE}: MAE = {mae}")
    plot_predictions(preds, trues, station_id, station_name, TARGET_DATE)

if __name__ == "__main__":
    main()
