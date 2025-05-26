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
ROUNDS = 15
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
    valid_csv_paths = [p for p in csv_paths if len(StationDataset(p)) > 1]
    num_clients = len(valid_csv_paths)
    print(f"Found {num_clients} valid clients.")

    global client_manager
    client_manager = ClientManager(valid_csv_paths)

    global node_id_to_client_index
    node_id_to_client_index = {node_id: idx for idx, node_id in enumerate(range(1, num_clients + 1))}

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

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CLIENTS_PER_ROUND / num_clients,
        min_fit_clients=CLIENTS_PER_ROUND,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_global_model
    )

    def client_fn(context: Context) -> fl.client.NumPyClient:
        cid = (context.node_id - 1) % len(valid_csv_paths)
        return client_manager.get_client(cid).to_client()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1},
    )

    print("Simulation complete!")
    print(f"Final metrics: {history.metrics_centralized}")


if __name__ == "__main__":
    main()
