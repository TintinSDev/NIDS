import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from src.decision.engine import NIDSDetectionEngine

DATA = "data/processed/test.parquet"
CHUNK_SIZE = 5000

df = pd.read_parquet(DATA)
engine = NIDSDetectionEngine()

totals = Counter()
family_stats = defaultdict(Counter)

print(f"Evaluating {len(df):,} held-out flows...")
print(f"Model: {engine.model_version}")
print(f"XGBoost threshold: {engine.malicious_threshold}")
print(f"Random Forest suspicious threshold: {engine.suspicious_threshold}")
print(f"Isolation Forest threshold: {engine.anomaly_threshold}")

for start in range(0, len(df), CHUNK_SIZE):
    chunk = df.iloc[start:start + CHUNK_SIZE]
    predictions = engine.predict_dataframe(chunk)

    y_true = chunk["is_malicious"].astype(int).to_numpy()
    y_pred = np.array(
        [p["level"] != "benign" for p in predictions],
        dtype=int,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    totals.update({
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "rows": len(chunk),
    })

    for family, indexes in chunk.groupby("attack_family").groups.items():
        positions = chunk.index.get_indexer(indexes)
        actual = y_true[positions]
        predicted = y_pred[positions]

        family_stats[str(family)].update({
            "rows": len(positions),
            "actual_attacks": int(actual.sum()),
            "detected": int(((actual == 1) & (predicted == 1)).sum()),
            "missed": int(((actual == 1) & (predicted == 0)).sum()),
            "false_positives": int(((actual == 0) & (predicted == 1)).sum()),
        })

    if (start // CHUNK_SIZE) % 10 == 0:
        print(f"Processed {min(start + CHUNK_SIZE, len(df)):,}/{len(df):,}")

tn, fp, fn, tp = (
    totals["tn"], totals["fp"], totals["fn"], totals["tp"]
)

precision = precision_score(
    [0] * (tn + fp) + [1] * (fn + tp),
    [0] * tn + [1] * fp + [0] * fn + [1] * tp,
    zero_division=0,
)

recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
fpr = fp / (fp + tn) if fp + tn else 0

print("\n========== FULL V2 ENGINE EVALUATION ==========")
print(f"Rows evaluated: {totals['rows']:,}")
print(f"Confusion matrix [[TN, FP], [FN, TP]]:")
print(np.array([[tn, fp], [fn, tp]]))
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"False-positive rate: {fpr:.4%}")

print("\n========== PER-ATTACK-FAMILY RESULTS ==========")
for family, stats in sorted(family_stats.items()):
    attacks = stats["actual_attacks"]
    detected = stats["detected"]
    family_recall = detected / attacks if attacks else None

    print(f"\n{family}")
    print(f"  Rows: {stats['rows']:,}")
    print(f"  Actual attacks: {attacks:,}")
    print(f"  Detected: {detected:,}")
    print(f"  Missed: {stats['missed']:,}")
    print(f"  False positives: {stats['false_positives']:,}")
    if family_recall is not None:
        print(f"  Attack recall: {family_recall:.4%}")

print("\nEvaluation complete. No database writes were performed.")
