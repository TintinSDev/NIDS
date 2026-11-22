import json
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

API = "http://localhost:8000/analyze/flows?persist=false"

df = pd.read_parquet("data/processed/test.parquet")
schema = json.load(open("src/features/features_v1.json"))
features = schema["features"] if isinstance(schema, dict) else schema

benign = df[df["is_malicious"] == 0].sample(n=10, random_state=42)
attacks = df[df["is_malicious"] == 1].sample(n=10, random_state=42)
sample = pd.concat([benign, attacks]).sample(frac=1, random_state=42)

flows = []
for _, row in sample.iterrows():
    values = {f: float(row[f]) for f in features}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite feature value found")

    flows.append({
        "src_ip": "192.0.2.10",
        "dst_ip": "198.51.100.20",
        "src_port": 12345,
        "dst_port": 80,
        "proto": "TCP",
        "features": values,
    })

payload = {"source": "v2_api_smoke_test", "flows": flows}

print("Submitting 20 held-out test flows...")
response = requests.post(API, json=payload, timeout=120)
print("HTTP:", response.status_code)
response.raise_for_status()

result = response.json()
predictions = result["results"]

if len(predictions) != len(sample):
    raise RuntimeError(f"Expected 20 predictions; received {len(predictions)}")

y_true = sample["is_malicious"].astype(int).to_numpy()
y_pred = np.array([p["level"] != "benign" for p in predictions], dtype=int)

print("\nAPI:", result["source"])
print("Flows analyzed:", result["flows_analyzed"])
print("Tier counts:", result["tier_counts"])
print("Persistence:", result["persistence"])

print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_true, y_pred, zero_division=0), 4))
print("F1:", round(f1_score(y_true, y_pred, zero_division=0), 4))

print("\nIndividual predictions:")
for i, p in enumerate(predictions):
    scores = p["detector_scores"]
    actual = "ATTACK" if y_true[i] else "BENIGN"
    print(
        f"{i+1:02d} Actual={actual:6s} "
        f"Predicted={p['level']:20s} "
        f"Class={p['attack_class']:12s} "
        f"XGB={scores['xgb_binary_prob']:.5f} "
        f"RF={scores['rf_binary_prob']:.5f} "
        f"Version={p['model_version']}"
    )

assert all(p["model_version"] == "v2.0.0" for p in predictions)
assert result["persistence"]["flows_persisted"] == 0
assert result["persistence"]["alerts_created"] == 0

print("\nPASS: v2 API predictions verified.")
print("PASS: No test records persisted.")
