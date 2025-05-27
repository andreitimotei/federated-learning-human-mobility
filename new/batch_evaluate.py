# batch_evaluate_and_plot.py

import matplotlib
matplotlib.use("Agg")

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model.model import FedMLPLSTM
from client.client import StationDataset
import xml.etree.ElementTree as ET

def load_model(model_path, static_dim, lag_seq_len, device):
    model = FedMLPLSTM(static_dim=static_dim, exog_dim=0, lag_seq_len=lag_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_day(model, dataset, date, device):
    timestamps, preds, trues, hours = [], [], [], []
    date = pd.to_datetime(date).date()
    found_any = False
    for i in range(len(dataset)):
        (static_exog, lag_seq), y = dataset[i]
        dt = pd.to_datetime(dataset.X_static_exog[i][0] * 1000000000, unit="s", errors="coerce")
        if dt.date() != date:
            continue
        found_any = True
        hour = dt.hour
        with torch.no_grad():
            static_exog = torch.tensor(static_exog).unsqueeze(0).to(device)
            lag_seq = torch.tensor(lag_seq).unsqueeze(0).to(device)
            y_tensor = torch.tensor(y).unsqueeze(0).to(device)
            pred = model(static_exog, lag_seq).squeeze().cpu().numpy()
            pred = np.round(pred).astype(int)
            pred = np.clip(pred, 0, None)
            actual = y_tensor.squeeze().cpu().numpy()
        timestamps.append(dt)
        hours.append(hour)
        preds.append(pred)
        trues.append(actual)

    if not found_any:
        print(f"[DEBUG] No data found for {date} in this station!")
    if not preds:
        return None, None, None

    mae = sum(abs(p - t).mean() for p, t in zip(preds, trues)) / len(preds)
    return mae, list(zip(hours, preds)), list(zip(hours, trues))

def load_station_names(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    terminal_to_name = {}
    for st in root.findall('station'):
        terminal = st.find('terminalName').text
        name = st.find('name').text
        terminal_to_name[terminal] = name
    return terminal_to_name

STATION_XML = "raw-data/stations.xml"
TERMINAL_TO_NAME = load_station_names(STATION_XML)

def get_station_name_from_id(station_id):
    if isinstance(station_id, str) and station_id.startswith("station_"):
        terminal = station_id.split("_")[1]
    else:
        terminal = str(station_id)
    # Pad to 6 digits for terminalName lookup
    terminal_padded = terminal.zfill(6)
    return TERMINAL_TO_NAME.get(terminal_padded, "Unknown")

def plot_predictions(preds, trues, station_id, date):
    import matplotlib.ticker as ticker

    station_name = get_station_name_from_id(station_id)

    # Initialize 24-hour zero arrays
    pred_arr, pred_dep = [0]*24, [0]*24
    true_arr, true_dep = [0]*24, [0]*24

    for hour, (a, d) in preds:
        pred_arr[hour] = a
        pred_dep[hour] = d
    for hour, (a, d) in trues:
        true_arr[hour] = a
        true_dep[hour] = d

    if not (
        any(x > 10 for x in pred_arr) or
        any(x > 10 for x in pred_dep) or
        any(x > 10 for x in true_arr) or
        any(x > 10 for x in true_dep)
    ):
        return

    hours = list(range(24))
    hour_labels = [f"{h:02d}:00" if h % 3 == 0 else "" for h in hours]

    all_vals = pred_arr + pred_dep + true_arr + true_dep
    max_val = max(all_vals)
    ylim = 60 if max_val > 40 else 40 if max_val > 20 else 20

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hours, true_arr, label="True Arrivals")
    plt.plot(hours, pred_arr, label="Pred Arrivals")
    plt.legend()
    plt.title("Arrivals")
    plt.xlabel("Hour of Day")
    plt.xticks(hours, hour_labels, rotation=45)
    plt.ylim(0, ylim)

    plt.subplot(1, 2, 2)
    plt.plot(hours, true_dep, label="True Departures")
    plt.plot(hours, pred_dep, label="Pred Departures")
    plt.legend()
    plt.title("Departures")
    plt.xlabel("Hour of Day")
    plt.xticks(hours, hour_labels, rotation=45)
    plt.ylim(0, ylim)

    plt.suptitle(f"Station {station_name} | {date}")
    fname = f"prediction_plot_2/accurate_plot_{station_id}_{date}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
                df["datetime"] = pd.to_datetime(df["datetime"], unit="s", errors="coerce")
                df = df.dropna(subset=["datetime"])
                unique_dates = df["datetime"].dt.date.unique()
                all_dates.update(unique_dates)
            except Exception as e:
                print(f"Skipped {file}: {e}")
    return sorted(list(all_dates))

def batch_evaluate(model_path, csv_dir, mae_threshold=1.0, device="cuda"):
    all_dates = get_all_dates_from_csvs(csv_dir)
    traffic_scores = []
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            path = os.path.join(csv_dir, file)
            try:
                df = pd.read_csv(path, usecols=["num_arrivals", "num_departures"])
                score = df["num_arrivals"].sum() + df["num_departures"].sum()
                traffic_scores.append((file, score))
            except Exception as e:
                print(f"Skipped {file} for traffic scoring: {e}")

    sorted_files = [f for f, _ in sorted(traffic_scores, key=lambda x: x[1], reverse=True)]

    for file in sorted_files:
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
        model_path="global_models/global_model_round_25.pt",
        csv_dir="processed-data/clients",
        mae_threshold=1.0,
        device="cuda"
    )
