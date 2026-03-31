import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error


# class GRUPredictor(nn.Module):
#     def __init__(self, input_dim=1, hidden_dim=64):
#         super().__init__()
#         self.gru = nn.GRU(
#             input_dim,
#             hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         # x: (B, T, 1)
#         out, _ = self.gru(x)
#         y_hat = self.fc(out)
#         return y_hat


class CNNLSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, cnn_dim=32, horizon=10):
        super().__init__()

        self.horizon = horizon
        self.input_dim = input_dim

        # ===== CNN =====
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ===== LSTM =====
        self.lstm = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # ⭐ 多步输出
        self.fc = nn.Linear(hidden_dim, input_dim * horizon)

    def forward(self, x):
        # x: (B, T, 1)

        B = x.shape[0]

        # CNN
        x = x.permute(0, 2, 1)      # (B, 1, T)
        x = self.conv(x)            # (B, cnn_dim, T)

        # LSTM
        x = x.permute(0, 2, 1)      # (B, T, cnn_dim)
        _, (h_n, _) = self.lstm(x)  # (1, B, hidden)

        h = h_n.squeeze(0)          # (B, hidden_dim)

        # ⭐ 预测未来 horizon 步
        y_hat = self.fc(h)          # (B, horizon * dim)
        y_hat = y_hat.view(B, self.horizon, self.input_dim)

        return y_hat

def predictive_score_metrics(
    ori_data,
    generated_data,
    device,
    horizon=10,
    iterations=2000,
    batch_size=128,
    lr=1e-3,
):

    # ===== tensor =====
    ori_data = ori_data.to(device=device, dtype=torch.float32)
    generated_data = generated_data.to(device=device, dtype=torch.float32)

    N, T, dim = generated_data.shape
    assert T > horizon, "Sequence length must be > horizon"

    model = CNNLSTMPredictor(
        input_dim=dim,
        hidden_dim=64,
        cnn_dim=32,
        horizon=horizon
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # ==========================
    # Train on generated data
    # ==========================
    model.train()

    for _ in range(iterations):

        idx = torch.randperm(N)[:batch_size]
        batch = generated_data[idx]   # (B, T, 1)

        X = batch[:, :-horizon, :]    # (B, T-H, 1)
        Y = batch[:, -horizon:, :]    # ⭐ 未来 H 步

        pred = model(X)               # (B, H, 1)

        loss = loss_fn(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ======================
    # Test on real data
    # ======================
    model.eval()

    with torch.no_grad():

        X = ori_data[:, :-horizon, :]
        Y = ori_data[:, -horizon:, :]

        pred = model(X)

        predictive_score = loss_fn(pred, Y).item()

    return predictive_score





