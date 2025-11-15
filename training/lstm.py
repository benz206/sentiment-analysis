"""
Stock-feature-only LSTM training using prices from stocks.sqlite.
Reads OHLCV for a ticker, builds sequences, trains, and evaluates.
"""

import math
import os
import random
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


@contextmanager
def db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def fetch_prices(ticker: str) -> np.ndarray:
    with db_connection() as conn:
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
        if not rows:
            raise RuntimeError(f"No price rows for ticker {ticker}")
        data = np.array(
            [[r["open"], r["high"], r["low"], r["close"], r["volume"]] for r in rows],
            dtype=np.float32,
        )
        return data


def fetch_all_tickers() -> List[str]:
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT ticker
            FROM prices
            ORDER BY ticker ASC
            """
        )
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError("No tickers found in prices table")
        return [r["ticker"] for r in rows]


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        return column in cols
    except Exception:
        return False


def fetch_features(
    ticker: str,
    use_sentiment: bool,
    sentiment_table: str,
    sentiment_col: str,
    sentiment_fillna: float,
) -> np.ndarray:
    """
    Returns feature matrix per day: [open, high, low, close, volume] (+ optional sentiment).
    If sentiment join fails or is missing, falls back to prices-only when use_sentiment is False.
    """
    if use_sentiment:
        try:
            with db_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT p.ts, p.open, p.high, p.low, p.close, p.volume, COALESCE(s.{sentiment_col}, ?) AS sentiment
                    FROM prices p
                    LEFT JOIN {sentiment_table} s
                      ON s.ticker = p.ticker AND s.ts = p.ts
                    WHERE p.ticker = ?
                    ORDER BY p.ts ASC
                    """,
                    (sentiment_fillna, ticker),
                )
                rows = cur.fetchall()
                if not rows:
                    raise RuntimeError(f"No rows for ticker {ticker}")
                data = np.array(
                    [
                        [
                            r["open"],
                            r["high"],
                            r["low"],
                            r["close"],
                            r["volume"],
                            r["sentiment"],
                        ]
                        for r in rows
                    ],
                    dtype=np.float32,
                )
                return data
        except Exception:
            if use_sentiment:
                return fetch_prices(ticker)

    return fetch_prices(ticker)


def train_val_split(arr: np.ndarray, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    # Split time series data into train and validation sets (chronological split)
    n = len(arr)
    split = max(1, int(n * (1 - val_ratio)))
    return arr[:split], arr[split:]


def standardize(
    train: np.ndarray, val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Normalize features using train set statistics (mean and std)
    # Apply same normalization to validation set to prevent data leakage
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (val - mean) / std, mean, std


def make_sequences(
    arr: np.ndarray, lookback: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    # Create sliding window sequences for time series prediction
    # X: sequences of lookback timesteps, Y: target value (close price) at horizon steps ahead
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class PriceLSTM(nn.Module):
    # LSTM model for stock price prediction
    # Architecture: LSTM layers -> Fully connected head -> Single output (close price)
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
        # Process sequence through LSTM, take last timestep output, predict price
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


@dataclass
class Config:
    ticker: str = "TSLA"
    target_ticker: str = "TSLA"
    lookback: int = 20
    horizon: int = 1
    val_ratio: float = 0.2
    batch_size: int = 32
    epochs: int = 250
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    multi_ticker: bool = False
    exclude_target_from_train: bool = False
    use_sentiment: bool = False
    sentiment_table: str = "daily_sentiment"
    sentiment_col: str = "sentiment"
    sentiment_fillna: float = 0.0
    early_stopping_patience: int = 20
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    gradient_clip: Optional[float] = 1.0
    num_workers: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> Dict[str, float]:
    # Calculate evaluation metrics: RMSE, MAE, MAPE, and directional accuracy
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae = float(np.mean(np.abs(preds - trues)))
    mape = float(np.mean(np.abs((preds - trues) / (trues + 1e-8))) * 100)

    # Directional accuracy: percentage of correct up/down predictions
    pred_direction = np.diff(preds) > 0
    true_direction = np.diff(trues) > 0
    directional_accuracy = float(np.mean(pred_direction == true_direction)) * 100

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }


def denormalize(
    arr: np.ndarray, mean: np.ndarray, std: np.ndarray, close_idx: int = 3
) -> np.ndarray:
    return arr * float(std[0, close_idx]) + float(mean[0, close_idx])


def prepare_multi_ticker_data(
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Prepare data for multi-ticker training: train on multiple tickers, validate on target ticker
    # Step 1: Load and prepare target ticker data (used for validation)
    all_tickers = fetch_all_tickers()
    target = cfg.target_ticker
    if target not in all_tickers:
        raise RuntimeError(f"Target ticker {target} not found in DB")

    raw_tgt = fetch_features(
        target,
        cfg.use_sentiment,
        cfg.sentiment_table,
        cfg.sentiment_col,
        cfg.sentiment_fillna,
    )
    train_raw_tgt, val_raw_tgt = train_val_split(raw_tgt, cfg.val_ratio)
    train_norm_tgt, val_norm_tgt, mean_tgt, std_tgt = standardize(
        train_raw_tgt, val_raw_tgt
    )
    x_train_tgt, y_train_tgt = make_sequences(train_norm_tgt, cfg.lookback, cfg.horizon)
    x_val_tgt, y_val_tgt = make_sequences(val_norm_tgt, cfg.lookback, cfg.horizon)

    # Step 2: Collect training data from all tickers (each standardized independently)
    xs_train: List[np.ndarray] = []
    ys_train: List[np.ndarray] = []

    if not cfg.exclude_target_from_train and len(x_train_tgt) > 0:
        xs_train.append(x_train_tgt)
        ys_train.append(y_train_tgt)

    for t in all_tickers:
        if t == target:
            continue
        try:
            raw = fetch_features(
                t,
                cfg.use_sentiment,
                cfg.sentiment_table,
                cfg.sentiment_col,
                cfg.sentiment_fillna,
            )
        except Exception:
            continue
        train_raw, _ = train_val_split(raw, cfg.val_ratio)
        if len(train_raw) < cfg.lookback + cfg.horizon:
            continue
        train_norm, _, _, _ = standardize(train_raw, train_raw)
        x_tr, y_tr = make_sequences(train_norm, cfg.lookback, cfg.horizon)
        if len(x_tr) == 0:
            continue
        xs_train.append(x_tr)
        ys_train.append(y_tr)

    if len(xs_train) == 0 or len(x_val_tgt) == 0:
        raise RuntimeError(
            "Insufficient data to form multi-ticker sequences; adjust lookback/horizon"
        )

    # Step 3: Concatenate all training data, return target ticker validation data
    x_train = np.concatenate(xs_train, axis=0)
    y_train = np.concatenate(ys_train, axis=0)
    return x_train, y_train, x_val_tgt, y_val_tgt, mean_tgt, std_tgt


def prepare_single_ticker_data(
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Prepare data for single-ticker training: train and validate on same ticker
    # Load features -> Split train/val -> Normalize -> Create sequences
    raw = fetch_features(
        cfg.ticker,
        cfg.use_sentiment,
        cfg.sentiment_table,
        cfg.sentiment_col,
        cfg.sentiment_fillna,
    )
    train_raw, val_raw = train_val_split(raw, cfg.val_ratio)
    train_norm, val_norm, mean, std = standardize(train_raw, val_raw)
    x_train, y_train = make_sequences(train_norm, cfg.lookback, cfg.horizon)
    x_val, y_val = make_sequences(val_norm, cfg.lookback, cfg.horizon)
    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError(
            "Insufficient data to form sequences; adjust lookback/horizon"
        )
    return x_train, y_train, x_val, y_val, mean, std


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: Config,
    ckpt_path: str,
) -> nn.Module:
    # Training loop with early stopping and learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )

    best_val = math.inf
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, cfg.epochs + 1):
        # Training phase: forward pass, compute loss, backpropagate, update weights
        model.train()
        train_losses: List[float] = []
        for xb, yb in train_dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        # Validation phase: evaluate on validation set without gradient computation
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
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f} lr={current_lr:.2e}"
        )

        # Track best model and implement early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (
            cfg.early_stopping_patience > 0
            and patience_counter >= cfg.early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(
    model: nn.Module,
    val_dl: DataLoader,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    # Generate predictions on validation set and denormalize to original price scale
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
    # Convert normalized predictions back to original price units
    denorm_preds = denormalize(preds, mean, std)
    denorm_trues = denormalize(trues, mean, std)
    return denorm_preds, denorm_trues


def save_results(
    denorm_preds: np.ndarray,
    denorm_trues: np.ndarray,
    ticker: str,
    metrics: Dict[str, float],
    output_dir: str = "output",
) -> None:
    # Save results to specified output directory
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"val_rmse={metrics['rmse']:.4f} val_mae={metrics['mae']:.4f} "
        f"val_mape={metrics['mape']:.2f}% dir_acc={metrics['directional_accuracy']:.2f}%"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(denorm_trues, label="actual")
    plt.plot(denorm_preds, label="pred")
    plt.title(f"{ticker} close: actual vs pred")
    plt.legend()
    out_png = os.path.join(output_dir, f"pred_vs_actual_{ticker}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"saved plot to {out_png}")

    out_csv = os.path.join(output_dir, f"predictions_{ticker}.csv")
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


def run(cfg: Config, output_dir: str = "output") -> None:
    # Main training pipeline: data preparation -> model training -> evaluation -> save results
    set_seed(cfg.seed)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and prepare data (single or multi-ticker mode)
    if cfg.multi_ticker:
        x_train, y_train, x_val, y_val, mean, std = prepare_multi_ticker_data(cfg)
        ticker = cfg.target_ticker
        ckpt_path = os.path.join(output_dir, f"lstm_multi_{ticker}.pt")
    else:
        x_train, y_train, x_val, y_val, mean, std = prepare_single_ticker_data(cfg)
        ticker = cfg.ticker
        ckpt_path = os.path.join(output_dir, f"lstm_{ticker}.pt")

    # Step 2: Create PyTorch datasets and data loaders
    train_ds = SeqDataset(x_train, y_train)
    val_ds = SeqDataset(x_val, y_val)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
    )

    # Step 3: Initialize model and move to device (GPU/CPU)
    model = PriceLSTM(
        input_size=x_train.shape[2],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    # Step 4: Train the model
    model = train_model(model, train_dl, val_dl, cfg, ckpt_path)

    # Step 5: Save best model checkpoint
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

    # Step 6: Evaluate model and save results (metrics, plots, predictions)
    denorm_preds, denorm_trues = evaluate_model(model, val_dl, mean, std, cfg)
    metrics = compute_metrics(denorm_preds, denorm_trues)
    save_results(denorm_preds, denorm_trues, ticker, metrics, output_dir)
    return metrics


def train_random_tickers(
    num_tickers: int = 100,
    output_dir: str = "generated",
    base_cfg: Optional[Config] = None,
) -> Dict[str, Dict[str, float]]:
    # Train models for random tickers and save all outputs to generated/ folder
    # Returns dictionary mapping ticker -> metrics
    all_tickers = fetch_all_tickers()

    if len(all_tickers) < num_tickers:
        print(f"Only {len(all_tickers)} tickers available, using all of them")
        selected_tickers = all_tickers
    else:
        selected_tickers = random.sample(all_tickers, num_tickers)

    print(f"Training models for {len(selected_tickers)} tickers...")
    os.makedirs(output_dir, exist_ok=True)

    if base_cfg is None:
        base_cfg = Config(multi_ticker=False)

    all_metrics: Dict[str, Dict[str, float]] = {}
    successful = 0
    failed = 0

    for i, ticker in enumerate(selected_tickers, 1):
        print(f"\n[{i}/{len(selected_tickers)}] Training model for {ticker}...")
        try:
            cfg = Config(
                ticker=ticker,
                multi_ticker=False,
                epochs=base_cfg.epochs,
                batch_size=base_cfg.batch_size,
                lr=base_cfg.lr,
                hidden_size=base_cfg.hidden_size,
                num_layers=base_cfg.num_layers,
                dropout=base_cfg.dropout,
                lookback=base_cfg.lookback,
                val_ratio=base_cfg.val_ratio,
                early_stopping_patience=base_cfg.early_stopping_patience,
                lr_scheduler_patience=base_cfg.lr_scheduler_patience,
                lr_scheduler_factor=base_cfg.lr_scheduler_factor,
                gradient_clip=base_cfg.gradient_clip,
                num_workers=base_cfg.num_workers,
                use_sentiment=base_cfg.use_sentiment,
                sentiment_table=base_cfg.sentiment_table,
                sentiment_col=base_cfg.sentiment_col,
                sentiment_fillna=base_cfg.sentiment_fillna,
            )
            metrics = run(cfg, output_dir=output_dir)
            all_metrics[ticker] = metrics
            successful += 1
        except Exception as e:
            print(f"Failed to train {ticker}: {e}")
            failed += 1
            continue

    # Save summary of all results
    summary_path = os.path.join(output_dir, "training_summary.csv")
    with open(summary_path, "w") as f:
        f.write("ticker,rmse,mae,mape,directional_accuracy\n")
        for ticker, metrics in sorted(all_metrics.items()):
            f.write(
                f"{ticker},{metrics['rmse']:.6f},{metrics['mae']:.6f},"
                f"{metrics['mape']:.2f},{metrics['directional_accuracy']:.2f}\n"
            )
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Successful: {successful}/{len(selected_tickers)}")
    print(f"Failed: {failed}/{len(selected_tickers)}")
    print(f"Summary saved to {summary_path}")
    print(f"{'='*60}")

    return all_metrics


if __name__ == "__main__":
    cfg = Config(
        multi_ticker=True, target_ticker="TSLA", exclude_target_from_train=False
    )
    run(cfg)
