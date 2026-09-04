"""Train autoencoder on non-fraud transactions using statement-compatible features."""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, AUTOENCODER_PATH
from src.db.connection import get_session
from src.ml.autoencoder import FraudAutoencoder
from src.ml.dataset import load_transactions_df
from src.ml.statement import STATEMENT_FEATURE_DIM, FEATURE_NAMES, featurize_dataframe


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def scale(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def train_autoencoder(
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 16,
    latent_dim: int = 8,
    calib_fraction: float = 0.2,
):
    session = get_session()
    try:
        train_df, _ = load_transactions_df(session)
    finally:
        session.close()

    legit = train_df[train_df["actual_class"] == 0].reset_index(drop=True)
    # Fit scaler + train AE only on a train split of legitimate rows.
    # Hold out a validation split so error_calibration is not built on
    # reconstruction errors the model has already minimized (avoids optimistic CDF).
    fit_df, calib_df = train_test_split(
        legit, test_size=calib_fraction, random_state=42
    )
    X_fit = featurize_dataframe(fit_df)
    X_calib = featurize_dataframe(calib_df)
    input_dim = X_fit.shape[1]
    assert input_dim == STATEMENT_FEATURE_DIM

    mean, std = fit_scaler(X_fit)
    X_fit_scaled = scale(X_fit, mean, std)
    X_calib_scaled = scale(X_calib, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X_fit_scaled).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        avg_loss = epoch_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  loss={avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        calib_tensor = torch.from_numpy(X_calib_scaled).float().to(device)
        calib_errors = model.reconstruction_error(calib_tensor).cpu().numpy().tolist()

    artifact = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "feature_mode": "bank_statement_amount_time",
        "feature_names": FEATURE_NAMES,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "error_calibration": calib_errors,
        "calibration_source": "held_out_legit_validation",
        "calibration_n": len(calib_errors),
        "train_legit_n": len(fit_df),
    }
    AUTOENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, AUTOENCODER_PATH)
    (ARTIFACTS_DIR / "autoencoder_meta.json").write_text(
        json.dumps(
            {k: v for k, v in artifact.items() if k not in {"state_dict", "error_calibration"}},
            indent=2,
        )
    )
    print(
        f"Saved model to {AUTOENCODER_PATH} "
        f"(train_legit={len(fit_df):,}, held_out_calib={len(calib_errors):,})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    args = parser.parse_args()
    train_autoencoder(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        calib_fraction=args.calib_fraction,
    )


if __name__ == "__main__":
    main()
