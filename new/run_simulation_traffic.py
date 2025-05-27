# run_simulation.py

import os
import sys
import torch
import glob
import flwr as fl
from torch.utils.data import DataLoader, random_split
from flwr.common import Context
import logging
import csv
import random
import pandas as pd

logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

METRICS_CSV = "global_metrics.csv"

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DATA_DIR = "processed-data/clients"
CLIENTS_PER_ROUND = 10
ROUNDS = 25
BATCH_SIZE = 64
LOCAL_EPOCHS = 10
LEARNING_RATE = 1e-3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from client.client import StationClient, StationDataset
from model.model import FedMLPLSTM

class ClientManager:
    def __init__(self, client_paths):
        self.client_paths = client_paths

    def get_client(self, cid: int) -> StationClient:
        if cid < 0 or cid >= len(self.client_paths):
            raise ValueError(f"Invalid client ID: {cid}.")
        csv_path = self.client_paths[cid]
        ds = StationDataset(csv_path)
        if len(ds) < 2:
            raise ValueError(f"Insufficient data for client {cid}")
        n_train = int(0.8 * len(ds))
        train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        return StationClient(train_loader=train_loader, val_loader=val_loader, device=DEVICE)

def main():
    pattern = os.path.join(DATA_DIR, "station_*.csv")
    csv_paths = sorted(glob.glob(pattern))

    # Calculate total traffic and sort clients by traffic
    traffic_scores = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            score = df["num_arrivals"].sum() + df["num_departures"].sum()
            traffic_scores.append((path, score))
        except Exception as e:
            print(f"Skipping {path}: {e}")

    traffic_scores.sort(key=lambda x: x[1], reverse=True)
    top_50_paths = [p for p, _ in traffic_scores[:200] if len(StationDataset(p)) > 1]
    print(f"Using top {len(top_50_paths)} clients for round rotation.")

    global_val_csv = "processed-data/global_validation.csv"
    global global_val_loader
    global_val_ds = StationDataset(global_val_csv)
    global_val_loader = DataLoader(global_val_ds, batch_size=BATCH_SIZE)

    if not os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "global_mae"])

    def save_final_model(parameters, save_path="final_model.pt"):
        model = FedMLPLSTM(13, 0, 5).to(DEVICE)
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict)
        torch.save(model.state_dict(), save_path)
        logging.info(f"[Model Saved] Final global model saved to {save_path}")

    def evaluate_global_model(server_round, parameters, config):
        static_dim, exog_dim, lag_seq_len = 13, 0, 5
        model = FedMLPLSTM(static_dim, exog_dim, lag_seq_len).to(DEVICE)
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict)
        model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for (static_exog, lag_seq), y in global_val_loader:
                static_exog, lag_seq, y = static_exog.to(DEVICE), lag_seq.to(DEVICE), y.to(DEVICE)
                preds = model(static_exog, lag_seq)
                total_loss += torch.nn.L1Loss()(preds, y).item() * y.size(0)

        avg_loss = total_loss / len(global_val_loader.dataset)
        logging.info(f"[Global Evaluation] Round {server_round} | MAE: {avg_loss:.4f}")
        with open(METRICS_CSV, "a", newline="") as f:
            csv.writer(f).writerow([server_round, avg_loss])
        save_final_model(parameters, save_path=f"global_models/global_model_round_{server_round}.pt")
        return avg_loss, {"loss": avg_loss}

    class RotatingClientManager:
        def __init__(self, all_paths):
            self.all_paths = all_paths
            self.round = 0

        def get_round_clients(self, round_num):
            random.seed(round_num)
            selected = random.sample(self.all_paths, CLIENTS_PER_ROUND)
            return ClientManager(selected)

    rotator = RotatingClientManager(top_50_paths)
    current_round = {"num": 0}

    def client_fn(context: Context) -> fl.client.NumPyClient:
        round_num = current_round["num"]
        global client_manager
        client_manager = rotator.get_round_clients(round_num)
        cid = (context.node_id - 1) % CLIENTS_PER_ROUND
        return client_manager.get_client(cid).to_client()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CLIENTS_PER_ROUND,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=CLIENTS_PER_ROUND,
            min_available_clients=CLIENTS_PER_ROUND,
            evaluate_fn=evaluate_global_model
        ),
        client_resources={"num_cpus": 1, "num_gpus": 1},
    )

    print("Simulation complete!")
    print(f"Final metrics: {history.metrics_centralized}")

if __name__ == "__main__":
    main()
