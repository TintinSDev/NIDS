import argparse
import json
from urllib import request

import pandas as pd

from src.decision.engine import NIDSDetectionEngine


FAMILIES = ["DoS/DDoS", "Brute Force", "PortScan"]


def post_flow(api_url: str, payload: dict) -> dict:
    req = request.Request(
        f"{api_url.rstrip('/')}/analyze/flows",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Validate SENTINEL V3 using labelled test samples."
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--dataset",
        default="data/processed/v3/test.parquet",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    engine = NIDSDetectionEngine()

    print(f"[sim] Loaded {len(df)} held-out V3 samples.")

    for number, family in enumerate(FAMILIES):
        candidates = df.index[df["attack_family"] == family]
        selected = None

        for idx in candidates:
            row = df.loc[[idx]]
            prediction = engine.predict_dataframe(row)[0]

            if prediction["level"] == "confirmed_malicious":
                selected = (row, prediction)
                break

        if selected is None:
            print(f"[sim] {family}: no confirmed-malicious sample found.")
            continue

        row, expected = selected
        source_ip = f"198.51.100.{10 + number}"

        payload = {
            "source": "simulation",
            "flows": [{
                "src_ip": source_ip,
                "dst_ip": "192.0.2.10",
                "src_port": 49152 + number,
                "dst_port": 443,
                "proto": "TCP",
                "features": {
                    feature: float(row.iloc[0][feature])
                    for feature in engine.features
                },
            }],
        }

        response = post_flow(args.api_url, payload)
        result = response["results"][0]
        persistence = response["persistence"]

        print(f"\n[sim] {family}")
        print(f"  Expected: {expected['level']} / {expected['attack_class']}")
        print(f"  Detected: {result['level']} / {result['attack_class']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Model: {result['model_version']}")
        print(f"  Alerts persisted: {persistence['alerts_created']}")
        print(f"  Evidence persisted: {persistence['evidence_records_created']}")

        if result["level"] != "confirmed_malicious":
            raise RuntimeError(f"V3 failed to confirm {family}")

    print("\n[sim] V3 API validation complete.")


if __name__ == "__main__":
    main()
