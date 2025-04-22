# simulate_flower.py

"""
Run this script to simulate a federated learning experiment
for hourly bike arrivals/departures forecasting across all TfL stations.
Simply execute:
    python3 simulate_flower.py
and the simulation will start with the settings below.
"""
import os
import sys
# ─── GPU CONFIGURATION ──────────────────────────────────────────────────────────
# Select the GPU device(s) to use (e.g., "0" for the first GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# PyTorch will automatically allocate memory as needed on the GPU.
# Detect and report device
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("CUDA not available, using CPU")

import glob
import flwr as fl
from torch.utils.data import DataLoader, random_split
from flwr.common import Context  # Import Context

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
# Path to folder containing all preprocessed station CSVs
DATA_DIR = "processed-data/clients"
# How many clients to sample each round
CLIENTS_PER_ROUND = 10
# Total number of federated rounds
ROUNDS = 50
# Local training hyperparameters
BATCH_SIZE = 32
LOCAL_EPOCHS = 10
LEARNING_RATE = 1e-3

# ─── IMPORT YOUR CLIENT AND DATASET ─────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from client.client import StationClient, StationDataset  # adjust if your module path differs
from model.model import FedMLPLSTM  # adjust if your module path differs

class ClientManager:
    def __init__(self, client_paths):
        self.client_paths = client_paths

    def get_client(self, cid: int) -> StationClient:
        if cid < 0 or cid >= len(self.client_paths):
            raise ValueError(f"Invalid client ID: {cid}. Must be in range [0, {len(self.client_paths) - 1}].")
        
        csv_path = self.client_paths[cid]
        ds = StationDataset(csv_path)
        
        # Check if the dataset is empty
        if len(ds) == 0:
            raise ValueError(f"Dataset for client {cid} is empty. Path: {csv_path}")
        
        n_train = int(0.8 * len(ds))
        n_val = len(ds) - n_train
        if n_train == 0 or n_val == 0:
            raise ValueError(f"Insufficient data for splitting. Client {cid} has {len(ds)} rows. Path: {csv_path}")
        
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        return StationClient(train_loader=train_loader, val_loader=val_loader, device=DEVICE)

def main():
    # Gather all station CSVs
    pattern = os.path.join(DATA_DIR, "station_*.csv")
    csv_paths = sorted(glob.glob(pattern))
    
    # Filter out empty or small datasets
    valid_csv_paths = []
    for path in csv_paths:
        ds = StationDataset(path)
        if len(ds) > 1:  # Ensure at least 2 rows for splitting
            valid_csv_paths.append(path)
        else:
            print(f"Skipping empty or small dataset: {path}")
    
    num_clients = len(valid_csv_paths)
    if num_clients == 0:
        raise RuntimeError(f"No valid station CSVs found in {DATA_DIR}")
    
    print(f"Found {num_clients} valid clients. Sampling {CLIENTS_PER_ROUND} per round.")
    
    # Initialize the client manager
    global client_manager
    client_manager = ClientManager(valid_csv_paths)
    
    # Create a mapping from node_id to client index
    global node_id_to_client_index
    node_id_to_client_index = {node_id: idx for idx, node_id in enumerate(range(1, num_clients + 1))}

    # 2. Define global validation DataLoader
    global_val_csv = "processed-data/global_validation.csv"  # Path to your global validation CSV
    global global_val_loader  # Declare as global so it can be used in evaluate_global_model
    global_val_ds = StationDataset(global_val_csv)
    global_val_loader = DataLoader(global_val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Define the global evaluation function
    def evaluate_global_model(server_round: int, parameters: fl.common.Parameters, config: dict):
        # Correct static_dim and exog_dim based on the dataset
        static_dim = 13  # Update this to match the number of static features in your dataset
        exog_dim = 0     # Update this if you have additional exogenous features
        lag_seq_len = 5  # Number of lagged time steps

        # Load the global model with the given parameters
        model = FedMLPLSTM(static_dim=static_dim, exog_dim=exog_dim, lag_seq_len=lag_seq_len).to(DEVICE)  # Move model to DEVICE
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in zip(model.state_dict().keys(), parameters)}  # Move parameters to DEVICE
        model.load_state_dict(state_dict)
        model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for (static_exog, lag_seq), y in global_val_loader:
                static_exog = static_exog.to(DEVICE)  # Move input tensors to DEVICE
                lag_seq = lag_seq.to(DEVICE)
                y = y.to(DEVICE)
                preds = model(static_exog, lag_seq)
                total_loss += torch.nn.MSELoss()(preds, y).item() * y.size(0)

        avg_loss = total_loss / len(global_val_loader.dataset)
        print(f"[Global Evaluation] Round {server_round}, Average Loss: {avg_loss:.4f}")
        return avg_loss, {"loss": avg_loss}

    # 4. Configure FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CLIENTS_PER_ROUND / num_clients,
        min_fit_clients=CLIENTS_PER_ROUND,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_global_model  # Use the updated function
    )

    # 5. Define client factory for simulation
    def client_fn(context: Context) -> fl.client.NumPyClient:
        # Validate node_id
        if context.node_id < 1:
            raise ValueError(f"Invalid node_id: {context.node_id}. Must be greater than 0.")
        
        # Map node_id to client index using modulo
        cid = (context.node_id - 1) % len(valid_csv_paths)
        return client_manager.get_client(cid).to_client()

    # 6. Start federated simulation
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
