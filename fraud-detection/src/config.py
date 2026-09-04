import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "")

AUTOENCODER_PATH = ARTIFACTS_DIR / "autoencoder.pt"
CLASSIFIER_PATH = ARTIFACTS_DIR / "classifier.pt"

AUTOENCODER_VERSION = os.getenv("AUTOENCODER_VERSION", "v1.0.0-autoencoder")
CLASSIFIER_VERSION = os.getenv("CLASSIFIER_VERSION", "v1.1.0-classifier")

# Backward-compatible aliases (default API model)
MODEL_PATH = Path(os.getenv("MODEL_PATH", AUTOENCODER_PATH))
MODEL_VERSION = os.getenv("MODEL_VERSION", AUTOENCODER_VERSION)

FEATURE_COLUMNS = [f"v{i}" for i in range(1, 29)]
NUM_FEATURES = len(FEATURE_COLUMNS)
