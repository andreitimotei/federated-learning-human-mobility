import torch
import torch.nn as nn

class FedMLPLSTM(nn.Module):
    def __init__(self, 
                 static_dim,      # e.g., nbDocks + lat + lon + temporal flags
                 exog_dim,        # e.g., num weather features
                 lag_seq_len,     # number of lagged time steps (e.g., 5)
                 hidden_mlp=64,
                 hidden_lstm=32,
                 lstm_layers=1):
        super(FedMLPLSTM, self).__init__()
        # MLP branch
        self.mlp = nn.Sequential(
            nn.Linear(static_dim + exog_dim, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, hidden_mlp // 2),
            nn.ReLU()
        )
        # LSTM branch
        self.lstm = nn.LSTM(input_size=2,  # departures+arrivals per lag
                            hidden_size=hidden_lstm,
                            num_layers=lstm_layers,
                            batch_first=True)
        # Fusion & output
        fusion_dim = (hidden_mlp // 2) + hidden_lstm
        self.fc_out = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 2)  # predict arrivals, departures
        )
    
    def forward(self, static_exog, lag_seq):
        """
        static_exog: [batch, static_dim+exog_dim]
        lag_seq:     [batch, lag_seq_len, 2]
        """
        mlp_feat = self.mlp(static_exog)                       # [batch, hidden_mlp//2]
        _, (h_n, _) = self.lstm(lag_seq)                       # h_n: [layers, batch, hidden_lstm]
        lstm_feat = h_n[-1]                                     # [batch, hidden_lstm]
        combined = torch.cat([mlp_feat, lstm_feat], dim=1)      # [batch, fusion_dim]
        return self.fc_out(combined)                            # [batch, 2]
