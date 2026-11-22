import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN = Path("data/processed/v3/train.parquet")
SCHEMA = Path("src/features/features_v1.json")
OUTPUT = Path("artifacts/models/v3/feature_drift_reference.json")

if not TRAIN.exists():
    raise SystemExit(f"Missing V3 training data: {TRAIN}")
if not SCHEMA.exists():
    raise SystemExit(f"Missing feature schema: {SCHEMA}")

schema = json.loads(SCHEMA.read_text())
features = schema.get("features", schema) if isinstance(schema, dict) else schema

if not isinstance(features, list) or len(features) != 52:
    raise SystemExit(f"Expected 52 schema features; found {len(features) if isinstance(features, list) else 'invalid schema'}")

df = pd.read_parquet(TRAIN)

missing = [f for f in features if f not in df.columns]
if missing:
    raise SystemExit(f"Training data is missing schema features: {missing}")

reference = {}

for feature in features:
    values = pd.to_numeric(df[feature], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna().to_numpy(dtype=np.float64)

    if len(values) == 0:
        raise SystemExit(f"No valid training values for feature: {feature}")

    # Quantile-based boundaries, calculated only from V3 training data.
    quantiles = np.quantile(values, np.linspace(0, 1, 11))
    edges = np.unique(quantiles)

    # Constant or near-constant features cannot support meaningful
    # multi-bin PSI. Keep them explicitly marked as non-informative.
    if len(edges) < 3:
        reference[feature] = {
            "edges": [],
            "expected": [],
            "informative": False,
            "training_samples": int(len(values)),
        }
        continue

    counts, _ = np.histogram(values, bins=edges)
    proportions = counts.astype(float) / counts.sum()

    reference[feature] = {
        "edges": [float(x) for x in edges],
        "expected": [float(x) for x in proportions],
        "informative": True,
        "training_samples": int(len(values)),
    }

artifact = {
    "version": "v3.0.0",
    "method": "PSI",
    "reference_source": str(TRAIN),
    "feature_count": len(features),
    "binning": "10 quantile bins; duplicate boundaries removed",
    "features": reference,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(artifact, indent=2))

print(f"Reference created: {OUTPUT}")
print(f"Features: {len(reference)}")
print(f"Informative features: {sum(x['informative'] for x in reference.values())}")
print(f"Training rows: {len(df):,}")
