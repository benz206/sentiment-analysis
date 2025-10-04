"""
Stock-feature-only LSTM training using prices from stocks.sqlite.
Reads OHLCV for a ticker, builds sequences, trains, and evaluates.
"""

import math
import os
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

"""
OHLCV: Open, High, Low, Close, Volume (per time step)
LSTM: Long Short-Term Memory
lookback: Number of past timesteps fed into the model.
horizon: How many steps ahead to predict (just doing 1 for 1 step ahead)
num_layers: Stacked LSTM layers
dropout: Regularization probability between layers
lr: Learning rate for the optimizer
MSE: Mean Squared Error (training loss here)
RMSE: Root Mean Squared Error (reported on validation; same units as price)
MAE: Mean Absolute Error (validation; price units)
val: Validation (held-out) split
std/mean: Standard deviation/mean (for feature standardization)
cuda: NVIDIA GPU backend; else CPU
ckpt: Checkpoint file saved to output/lstm_<TICKER>.pt
TSLA: Tesla ticker symbol
"""

DB_PATH = os.path.join(os.getcwd(), "stocks.sqlite")


def fetch_prices(ticker: str) -> np.ndarray:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ts, open, high, low, close, volume
        FROM prices
        WHERE ticker = ?
        ORDER BY ts ASC
        """,
        (ticker,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        raise RuntimeError(f"No price rows for ticker {ticker}")
    data = np.array(
        [[r["open"], r["high"], r["low"], r["close"], r["volume"]] for r in rows],
        dtype=np.float32,
    )
    return data


def train_val_split(arr: np.ndarray, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(arr)
    split = max(1, int(n * (1 - val_ratio)))
    return arr[:split], arr[split:]


def standardize(
    train: np.ndarray, val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (val - mean) / std, mean, std


def make_sequences(
    arr: np.ndarray, lookback: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in range(lookback, len(arr) - horizon + 1):
        window = arr[i - lookback : i]
        target_close = arr[i + horizon - 1, 3]
        xs.append(window)
        ys.append([target_close])
    return np.stack(xs), np.stack(ys)


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class PriceLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


@dataclass
class Config:
    ticker: str = "TSLA"
    lookback: int = 20
    horizon: int = 1
    val_ratio: float = 0.2
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    raw = fetch_prices(cfg.ticker)
    train_raw, val_raw = train_val_split(raw, cfg.val_ratio)
    train_norm, val_norm, mean, std = standardize(train_raw, val_raw)
    x_train, y_train = make_sequences(train_norm, cfg.lookback, cfg.horizon)
    x_val, y_val = make_sequences(val_norm, cfg.lookback, cfg.horizon)
    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError(
            "Insufficient data to form sequences; adjust lookback/horizon"
        )

    train_ds = SeqDataset(x_train, y_train)
    val_ds = SeqDataset(x_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = PriceLSTM(
        input_size=train_norm.shape[1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    os.makedirs("output", exist_ok=True)
    ckpt_path = os.path.join("output", f"lstm_{cfg.ticker}.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: List[float] = []
        for xb, yb in train_dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"epoch {epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "mean": mean,
                    "std": std,
                },
                ckpt_path,
            )

    print(f"saved best checkpoint to {ckpt_path}")

    model.eval()
    preds: List[float] = []
    trues: List[float] = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            p = model(xb)
            preds.extend(p.squeeze(-1).cpu().numpy().tolist())
            trues.extend(yb.squeeze(-1).cpu().numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)
    close_idx = 3
    denorm_preds = preds * float(std[0, close_idx]) + float(mean[0, close_idx])
    denorm_trues = trues * float(std[0, close_idx]) + float(mean[0, close_idx])

    rmse = float(np.sqrt(np.mean((denorm_preds - denorm_trues) ** 2)))
    mae = float(np.mean(np.abs(denorm_preds - denorm_trues)))
    print(f"val_rmse={rmse:.4f} val_mae={mae:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(denorm_trues, label="actual")
    plt.plot(denorm_preds, label="pred")
    plt.title(f"{cfg.ticker} close: actual vs pred")
    plt.legend()
    out_png = os.path.join("output", f"pred_vs_actual_{cfg.ticker}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"saved plot to {out_png}")

    out_csv = os.path.join("output", f"predictions_{cfg.ticker}.csv")
    np.savetxt(
        out_csv,
        np.column_stack([denorm_trues, denorm_preds]),
        delimiter=",",
        header="actual,prediction",
        comments="",
        fmt="%.6f",
    )
    print(f"saved predictions to {out_csv}")

    k = min(10, len(denorm_trues))
    print("last values (actual, pred):")
    for a, p in zip(denorm_trues[-k:], denorm_preds[-k:]):
        print(f"{a:.2f}\t{p:.2f}")


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
