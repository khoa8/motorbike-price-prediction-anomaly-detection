from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"

required_files = [
    "price_model.joblib",
    "isolation_preprocessor.joblib",
    "isolation_forest.joblib",
    "deployment_config.json",
    "segment_rules.json",
    "segment_statistics.csv",
    "input_options.json",
]

missing = [
    filename
    for filename in required_files
    if not (ARTIFACT_DIR / filename).exists()
]

if missing:
    raise FileNotFoundError(
        "Thiếu artifact: " + ", ".join(missing)
    )

with open(
    ARTIFACT_DIR / "deployment_config.json",
    encoding="utf-8",
) as file:
    config = json.load(file)

price_model = joblib.load(ARTIFACT_DIR / "price_model.joblib")
isolation_preprocessor = joblib.load(
    ARTIFACT_DIR / "isolation_preprocessor.joblib"
)
isolation_forest = joblib.load(
    ARTIFACT_DIR / "isolation_forest.joblib"
)
segment_statistics = pd.read_csv(
    ARTIFACT_DIR / "segment_statistics.csv"
)

print("Artifacts loaded successfully.")
print("Model:", type(price_model).__name__)
print("Isolation preprocessor:", type(isolation_preprocessor).__name__)
print("Isolation model:", type(isolation_forest).__name__)
print("Segments:", len(segment_statistics))
print("Configured Python:", config["versions"]["python"])
print("Configured scikit-learn:", config["versions"]["scikit_learn"])
