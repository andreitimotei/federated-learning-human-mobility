# client.py

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import flwr as fl
import sys
import os

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from new.model.model import FedMLPLSTM  # your federated model from model/model.py

# ======== Dataset Definition ========
class StationDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Convert datetime to Unix time
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["datetime"] = df["datetime"].apply(lambda x: x.timestamp() if pd.notnull(x) else None)

        # Drop rows with NaN values after conversion
        df = df.dropna()

        # Check for NaN or Inf in the dataset
        if df.isnull().values.any():
            raise ValueError(f"Dataset {csv_path} contains NaN values")
        if not np.isfinite(df.values).all():
            raise ValueError(f"Dataset {csv_path} contains Inf values")

        # Targets
        self.y = df[["num_arrivals", "num_departures"]].values.astype("float32")
        
        # Features
        static_exog_cols = [
            "datetime",  # Include the converted datetime column
            "hour", "day_of_week", "month", "is_weekend", "is_holiday",
            "lat", "lon", "nbDocks",
            "temperature", "precipitation", "humidity", "pressure",
        ]
        lag_cols = [
            "num_arrivals_lag1", "num_arrivals_lag2", "num_arrivals_lag3",
            "num_arrivals_lag24", "num_arrivals_lag168",
            "num_departures_lag1", "num_departures_lag2", "num_departures_lag3",
            "num_departures_lag24", "num_departures_lag168",
        ]
        self.X_static_exog = df[static_exog_cols].values.astype("float32")
        n_lags = len(lag_cols) // 2
        arr_lags = df[[c for c in lag_cols if "arrivals" in c]].values.reshape(-1, n_lags)
        dep_lags = df[[c for c in lag_cols if "departures" in c]].values.reshape(-1, n_lags)
        self.X_lag_seq = np.stack([arr_lags, dep_lags], axis=-1).astype("float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_static_exog[idx], self.X_lag_seq[idx]), self.y[idx]

# ======== Flower Client ========
class StationClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, device="cpu"):
        self.device = torch.device(device)

        # Access the original dataset from the Subset object
        original_dataset = train_loader.dataset.dataset

        # Adjust these dims if you change feature lists
        static_dim = original_dataset.X_static_exog.shape[1]
        exog_dim = 0  # already included in static_exog_cols
        lag_seq_len = original_dataset.X_lag_seq.shape[1]

        self.model = FedMLPLSTM(
            static_dim=static_dim,
            exog_dim=exog_dim,
            lag_seq_len=lag_seq_len
        ).to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config=None):
        """
        Return the model parameters as a list of NumPy arrays.
        The `config` argument is included to match Flower's expected signature.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Load global weights
        self.set_parameters(parameters)
        # Read hyperparams
        epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 32))
        lr = float(config.get("lr", 1e-3))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        
        # Local training
        self.model.train()
        for _ in range(epochs):
            for (static_exog, lag_seq), y in self.train_loader:
                static_exog = static_exog.to(self.device)
                lag_seq = lag_seq.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(static_exog, lag_seq)
                loss = self.criterion(preds, y)
                loss.backward()

                for p in self.model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            print("NaN or Inf detected in gradients")

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for (static_exog, lag_seq), y in self.val_loader:
                # Check for NaN or Inf in the input data
                if torch.isnan(static_exog).any() or torch.isinf(static_exog).any():
                    print("NaN or Inf detected in static_exog")
                if torch.isnan(lag_seq).any() or torch.isinf(lag_seq).any():
                    print("NaN or Inf detected in lag_seq")
                if torch.isnan(y).any() or torch.isinf(y).any():
                    print("NaN or Inf detected in targets (y)")

                static_exog = static_exog.to(self.device)
                lag_seq = lag_seq.to(self.device)
                y = y.to(self.device)
                preds = self.model(static_exog, lag_seq)

                # Check for NaN or Inf in predictions
                if torch.isnan(preds).any() or torch.isinf(preds).any():
                    print("NaN or Inf detected in predictions")

                preds = torch.clamp(preds, min=-1e6, max=1e6)

                total_loss += self.criterion(preds, y).item() * y.size(0)

        avg_loss = total_loss / len(self.val_loader.dataset)
        print(f"[Client Evaluation] Average Loss: {avg_loss:.4f}")
        return float(avg_loss), len(self.val_loader.dataset), {"mse": float(avg_loss)}

# ======== Main Entrypoint ========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to processed station CSV")
    parser.add_argument("--server_address", type=str, default="localhost:8080")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()

    # Load dataset
    ds = StationDataset(args.csv)
    # 80/20 split
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Start federated client
    client = StationClient(train_loader, val_loader, device=args.device)
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
        config={"local_epochs": args.local_epochs, "batch_size": args.batch_size, "lr": 1e-3}
    )

if __name__ == "__main__":
    main()

