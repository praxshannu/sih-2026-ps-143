"""LSTM autoencoder training for vessel behavior anomaly detection.

Trains an LSTM encoder-decoder on normal vessel behavior sequences
(speed, course, heading) and learns a reconstruction error threshold
for detecting anomalies (e.g., AIS gaps, speed drops near spill origin).

Usage:
    python ml/training/train_anomaly.py --data-dir ./data/ais_normal --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LSTMEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, (h_n, c_n) = self.lstm(x)
        output = self.norm(output)
        return output, h_n, c_n


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, encoder_output: torch.Tensor, h_n: torch.Tensor, c_n: torch.Tensor
    ) -> torch.Tensor:
        decoder_input = encoder_output[:, -1:, :].repeat(1, self.seq_len, 1)
        output, _ = self.lstm(decoder_input, (h_n, c_n))
        return self.output_proj(output)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        seq_len: int = 48,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, seq_len, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc_out, h_n, c_n = self.encoder(x)
        reconstructed = self.decoder(enc_out, h_n, c_n)
        return reconstructed, enc_out

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return reconstructed, error


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_ais_sequences(data_dir: str, seq_len: int = 48, stride: int = 6) -> np.ndarray:
    data_path = Path(data_dir)
    # One element type for the accumulator: every branch appends a 2-D
    # (seq_len, features) array. Mixing nested lists and arrays here made the
    # declared ``-> np.ndarray`` return a lie in one branch and a truth in the
    # other, which is how it stayed broken without failing.
    sequences: list[np.ndarray] = []

    csv_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.npy"))
    if not csv_files:
        print(f"[anomaly] No data files in {data_dir}, generating synthetic normal behavior...")
        synthetic = _generate_synthetic_normal(n_vessels=200, seq_len=seq_len)
        return np.asarray(synthetic, dtype=np.float32)

    for f in csv_files:
        if f.suffix == ".npy":
            data = np.load(f)
            for i in range(0, len(data) - seq_len, stride):
                sequences.append(data[i : i + seq_len])
        else:
            import csv

            with open(f) as fh:
                reader = csv.DictReader(fh)
                rows = []
                for row in reader:
                    try:
                        rows.append(
                            [
                                float(row.get("sog", 0)),
                                float(row.get("cog", 0)) / 360.0,
                                float(row.get("heading", 0)) / 360.0,
                            ]
                        )
                    except (ValueError, TypeError):
                        continue
                arr = np.array(rows, dtype=np.float32)
                for i in range(0, len(arr) - seq_len, stride):
                    seq = arr[i : i + seq_len]
                    if np.std(seq) > 0.01:
                        sequences.append(seq)

    if not sequences:
        print("[anomaly] No valid sequences found, generating synthetic data...")
        seqs = _generate_synthetic_normal(n_vessels=200, seq_len=seq_len)
        return np.array(seqs, dtype=np.float32)

    return np.array(sequences, dtype=np.float32)


def _generate_synthetic_normal(n_vessels: int = 200, seq_len: int = 48) -> list[list[list[float]]]:
    sequences: list[list[list[float]]] = []
    for _ in range(n_vessels):
        base_speed = np.random.uniform(5.0, 20.0) / 25.0
        base_course = np.random.uniform(0.0, 1.0)
        speed_noise = np.random.normal(0, 0.02, seq_len).cumsum() * 0.1
        course_noise = np.random.normal(0, 0.01, seq_len).cumsum() * 0.1

        speeds = np.clip(base_speed + speed_noise, 0, 1.0)
        courses = np.clip(base_course + course_noise, 0, 1.0)
        headings = courses + np.random.normal(0, 0.005, seq_len)

        seq = np.stack([speeds, courses, np.clip(headings, 0, 1.0)], axis=1).astype(np.float32)
        sequences.append(seq.tolist())
    return sequences


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch[0].to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstructed, _ = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def compute_threshold(
    model: LSTMAutoencoder, loader: DataLoader, device: torch.device, percentile: float = 95.0
) -> float:
    model.eval()
    all_errors = []
    for batch in loader:
        x = batch[0].to(device)
        _, errors = model.reconstruct(x)
        all_errors.extend(errors.cpu().tolist())
    threshold = float(np.percentile(all_errors, percentile))
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder for anomaly detection")
    parser.add_argument("--data-dir", type=str, default="./data/ais_normal")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/anomaly")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[anomaly] Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    sequences = load_ais_sequences(args.data_dir, seq_len=args.seq_len)
    print(f"[anomaly] Loaded {len(sequences)} sequences of length {args.seq_len}")

    split = int(0.9 * len(sequences))
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    train_tensor = torch.from_numpy(train_seqs)
    val_tensor = torch.from_numpy(val_seqs)

    train_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=args.batch_size, shuffle=False)

    model = LSTMAutoencoder(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[anomaly] Model params: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reconstructed, _ = model(x)
                val_loss += criterion(reconstructed, x).item() * x.size(0)
                n += x.size(0)
        val_loss /= n

        print(
            f"Epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "lstm_autoencoder_best.pth"),
            )

    threshold = compute_threshold(model, val_loader, device, percentile=args.threshold_percentile)
    print(f"[anomaly] Anomaly threshold ({args.threshold_percentile}th pct): {threshold:.6f}")

    threshold_info = {
        "threshold": threshold,
        "percentile": args.threshold_percentile,
        "val_loss": best_val_loss,
        "input_dim": 3,
        "hidden_dim": args.hidden_dim,
        "seq_len": args.seq_len,
        "num_layers": args.num_layers,
    }
    with open(os.path.join(args.output_dir, "anomaly_threshold.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "lstm_autoencoder_final.pth"))
    print(f"[anomaly] Saved model and threshold to {args.output_dir}")


if __name__ == "__main__":
    main()
