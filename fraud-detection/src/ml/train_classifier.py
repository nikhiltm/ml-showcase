"""Train class-weighted classifier on statement-compatible features."""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ARTIFACTS_DIR, CLASSIFIER_PATH
from src.db.connection import get_session
from src.ml.autoencoder import FraudClassifier
from src.ml.dataset import load_transactions_df
from src.ml.statement import STATEMENT_FEATURE_DIM, FEATURE_NAMES, featurize_dataframe
from src.ml.train import fit_scaler, scale


def train_classifier(
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 64,
):
    session = get_session()
    try:
        train_df, _ = load_transactions_df(session)
    finally:
        session.close()

    X = featurize_dataframe(train_df)
    y = train_df["actual_class"].values.astype(np.float32)
    input_dim = X.shape[1]
    assert input_dim == STATEMENT_FEATURE_DIM

    mean, std = fit_scaler(X)
    X_scaled = scale(X, mean, std)

    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudClassifier(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    dataset = TensorDataset(
        torch.from_numpy(X_scaled).float(),
        torch.from_numpy(y).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  loss={avg_loss:.6f}  pos_weight={pos_weight.item():.1f}")

    artifact = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "feature_mode": "bank_statement_amount_time",
        "feature_names": FEATURE_NAMES,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "pos_weight": float(pos_weight.item()),
    }
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, CLASSIFIER_PATH)
    (ARTIFACTS_DIR / "classifier_meta.json").write_text(
        json.dumps({k: v for k, v in artifact.items() if k != "state_dict"}, indent=2)
    )
    print(f"Saved classifier to {CLASSIFIER_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
