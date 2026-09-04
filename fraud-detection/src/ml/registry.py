"""Model registry: load artifacts and score feature matrices or statements."""

from dataclasses import dataclass, field

import numpy as np
import torch

from src.config import (
    ARTIFACTS_DIR,
    AUTOENCODER_PATH,
    AUTOENCODER_VERSION,
    CLASSIFIER_PATH,
    CLASSIFIER_VERSION,
)
from src.ml.autoencoder import FraudAutoencoder, FraudClassifier
from src.ml.metrics import errors_to_calibrated_scores
from src.ml.statement import featurize_bank_statement
from src.ml.train import scale


@dataclass
class ModelSpec:
    key: str
    version: str
    path: str
    model_type: str


MODELS: dict[str, ModelSpec] = {
    "autoencoder": ModelSpec(
        key="autoencoder",
        version=AUTOENCODER_VERSION,
        path=str(AUTOENCODER_PATH),
        model_type="anomaly_detection",
    ),
    "classifier": ModelSpec(
        key="classifier",
        version=CLASSIFIER_VERSION,
        path=str(CLASSIFIER_PATH),
        model_type="supervised",
    ),
}


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    device: torch.device
    error_calibration: np.ndarray | None = None


def load_model(model_key: str, device: torch.device | None = None) -> LoadedModel:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(MODELS)}")

    spec = MODELS[model_key]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(spec.path, map_location=device, weights_only=False)

    if model_key == "autoencoder":
        model = FraudAutoencoder(
            artifact["input_dim"],
            artifact["hidden_dim"],
            artifact["latent_dim"],
        ).to(device)
    else:
        model = FraudClassifier(
            artifact["input_dim"],
            artifact.get("hidden_dim", 64),
        ).to(device)

    model.load_state_dict(artifact["state_dict"])
    model.eval()

    calib = artifact.get("error_calibration")
    return LoadedModel(
        spec=spec,
        model=model,
        mean=np.array(artifact["scaler_mean"], dtype=np.float32),
        std=np.array(artifact["scaler_std"], dtype=np.float32),
        device=device,
        error_calibration=np.array(calib, dtype=np.float64) if calib is not None else None,
    )


def score_batch(loaded: LoadedModel, X: np.ndarray) -> np.ndarray:
    X_scaled = scale(X, loaded.mean, loaded.std)
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(X_scaled, dtype=np.float32)).float().to(loaded.device)
        if loaded.spec.key == "autoencoder":
            errors = loaded.model.reconstruction_error(tensor).cpu().numpy()
            if loaded.error_calibration is None or len(loaded.error_calibration) == 0:
                raise RuntimeError(
                    "Autoencoder artifact missing error_calibration — retrain with "
                    "python -m src.ml.train"
                )
            return errors_to_calibrated_scores(errors, loaded.error_calibration)
        logits = loaded.model(tensor).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))


def score_statements(loaded: LoadedModel, statements: list[dict]) -> np.ndarray:
    X = np.vstack(
        [
            featurize_bank_statement(
                amount=s["amount"],
                date=s["date"],
                description=s.get("description", ""),
                merchant=s.get("merchant", ""),
                mcc=s.get("mcc"),
            )
            for s in statements
        ]
    )
    return score_batch(loaded, X)


def evaluation_path(model_key: str):
    from pathlib import Path

    return Path(ARTIFACTS_DIR) / f"evaluation_{model_key}.json"
