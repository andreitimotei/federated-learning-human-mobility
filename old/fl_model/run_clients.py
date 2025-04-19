import os
import subprocess

def launch_clients(data_folder, n_clients=None):
    client_files = sorted([
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.startswith("client_") and f.endswith(".csv")
    ])

    if n_clients:
        client_files = client_files[:n_clients]

    processes = []
    for client_file in client_files:
        print(f"Launching client for: {os.path.basename(client_file)}")
        proc = subprocess.Popen(
            ["python3", "fl_model/client.py", client_file]
        )
        processes.append(proc)

    for proc in processes:
        proc.wait()

if __name__ == "__main__":
    launch_clients("data/clients", n_clients=10)  # Set None to run all
