import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


SCHEMA_PATH = Path("src/features/features_v1.json")


def load_schema_features(schema_path: Path = SCHEMA_PATH) -> List[str]:
    payload = json.loads(schema_path.read_text())
    return list(payload["features"])


def vectorize_flow_dict(
    raw_features: Dict[str, float],
    feature_order: Optional[List[str]] = None,
) -> np.ndarray:
    """Return a 2D float32 array aligned to features_v1.json."""
    cols = feature_order or load_schema_features()
    row = []

    for col in cols:
        val = raw_features.get(col, 0.0)

        try:
            fval = float(val)
        except (TypeError, ValueError):
            fval = 0.0

        if not np.isfinite(fval):
            fval = 0.0

        row.append(fval)

    return np.asarray([row], dtype=np.float32)


def vectorize_dataframe(
    df: pd.DataFrame,
    feature_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Align a DataFrame to features_v1.json and sanitize non-finite values."""
    cols = feature_order or load_schema_features()
    out = pd.DataFrame(index=df.index)

    for col in cols:
        if col in df.columns:
            series = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
        else:
            series = pd.Series(0.0, index=df.index)

        out[col] = series.astype(np.float32)

    return out


class FlowAccumulator:
    """
    Aggregate both directions of a connection into one bidirectional flow.

    The first observed packet defines the forward direction. Packets travelling
    in the opposite direction contribute to backward statistics.
    """

    FLAG_NAMES = (
        "fin",
        "syn",
        "rst",
        "psh",
        "ack",
        "urg",
        "cwe",
        "ece",
    )

    def __init__(
        self,
        idle_timeout_s: float = 15.0,
        active_timeout_s: float = 120.0,
    ):
        if idle_timeout_s <= 0:
            raise ValueError("idle_timeout_s must be greater than zero")

        if active_timeout_s <= 0:
            raise ValueError("active_timeout_s must be greater than zero")

        self.idle_timeout_s = float(idle_timeout_s)
        self.active_timeout_s = float(active_timeout_s)
        self.flows: Dict[tuple, dict] = {}
        self.feature_order = load_schema_features()

    @staticmethod
    def _endpoint(ip: str, port: int) -> tuple:
        return str(ip), int(port)

    @classmethod
    def _flow_key(
        cls,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        proto: str,
    ) -> tuple:
        """
        Produce the same key regardless of packet direction.

        The endpoint pairs are sorted so A:port1 -> B:port2 and
        B:port2 -> A:port1 resolve to the same connection.
        """
        src = cls._endpoint(src_ip, src_port)
        dst = cls._endpoint(dst_ip, dst_port)

        endpoint_a, endpoint_b = sorted((src, dst))

        return endpoint_a, endpoint_b, str(proto).upper()

    @staticmethod
    def _intervals(timestamps: List[float]) -> np.ndarray:
        """Return non-negative inter-arrival intervals in microseconds."""
        if len(timestamps) < 2:
            return np.asarray([], dtype=np.float64)

        ordered = np.sort(np.asarray(timestamps, dtype=np.float64))
        intervals = np.diff(ordered) * 1e6

        return np.maximum(intervals, 0.0)

    @staticmethod
    def _stats(values: List[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)

        if array.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "variance": 0.0,
            }

        return {
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "variance": float(array.var()),
        }

    def add_packet(
        self,
        ts: float,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        proto: str,
        pkt_len: int,
        header_len: int = 20,
        flags: Optional[Dict[str, int]] = None,
    ) -> List[dict]:
        """Ingest one packet and return flows that have timed out."""
        ts = float(ts)
        src = self._endpoint(src_ip, src_port)
        dst = self._endpoint(dst_ip, dst_port)
        proto = str(proto).upper()

        completed = self.flush_expired(now_ts=ts)
        key = self._flow_key(src_ip, dst_ip, src_port, dst_port, proto)
        flags = flags or {}

        if key not in self.flows:
            self.flows[key] = {
                "start_ts": ts,
                "last_ts": ts,

                # Preserve the direction of the first packet.
                "src_ip": src[0],
                "dst_ip": dst[0],
                "src_port": src[1],
                "dst_port": dst[1],
                "proto": proto,
                "forward_endpoint": src,
                "backward_endpoint": dst,

                "fwd_lengths": [],
                "bwd_lengths": [],
                "fwd_timestamps": [],
                "bwd_timestamps": [],
                "all_timestamps": [],

                "fwd_header_len": 0.0,
                "bwd_header_len": 0.0,

                "fwd_psh_flags": 0,
                "bwd_psh_flags": 0,
                "fwd_urg_flags": 0,
                "bwd_urg_flags": 0,

                "flag_counts": {
                    name: 0 for name in self.FLAG_NAMES
                },
            }

        flow = self.flows[key]

        if src == flow["forward_endpoint"] and dst == flow["backward_endpoint"]:
            direction = "fwd"
        elif src == flow["backward_endpoint"] and dst == flow["forward_endpoint"]:
            direction = "bwd"
        else:
            raise RuntimeError("Packet endpoints do not match the flow key")

        length = float(pkt_len)
        header = float(header_len)

        flow[f"{direction}_lengths"].append(length)
        flow[f"{direction}_timestamps"].append(ts)
        flow[f"{direction}_header_len"] += header
        flow["all_timestamps"].append(ts)

        flow["start_ts"] = min(flow["start_ts"], ts)
        flow["last_ts"] = max(flow["last_ts"], ts)

        if direction == "fwd":
            flow["fwd_psh_flags"] += int(bool(flags.get("psh", 0)))
            flow["fwd_urg_flags"] += int(bool(flags.get("urg", 0)))
        else:
            flow["bwd_psh_flags"] += int(bool(flags.get("psh", 0)))
            flow["bwd_urg_flags"] += int(bool(flags.get("urg", 0)))

        for name in self.FLAG_NAMES:
            if flags.get(name, 0):
                flow["flag_counts"][name] += 1

        return completed

    def _finalize_flow(self, flow: dict) -> dict:
        fwd_lengths = flow["fwd_lengths"]
        bwd_lengths = flow["bwd_lengths"]
        all_lengths = fwd_lengths + bwd_lengths

        fwd_count = len(fwd_lengths)
        bwd_count = len(bwd_lengths)
        total_count = fwd_count + bwd_count

        fwd_bytes = float(sum(fwd_lengths))
        bwd_bytes = float(sum(bwd_lengths))
        total_bytes = fwd_bytes + bwd_bytes

        duration_us = max(
            (flow["last_ts"] - flow["start_ts"]) * 1e6,
            1.0,
        )
        duration_s = duration_us / 1e6

        flow_iats = self._intervals(flow["all_timestamps"])
        fwd_iats = self._intervals(flow["fwd_timestamps"])
        bwd_iats = self._intervals(flow["bwd_timestamps"])

        fwd_stats = self._stats(fwd_lengths)
        bwd_stats = self._stats(bwd_lengths)
        packet_stats = self._stats(all_lengths)

        flags = flow["flag_counts"]

        vec = {
            "flow_duration": duration_us,

            "total_fwd_packets": float(fwd_count),
            "total_bwd_packets": float(bwd_count),

            "total_length_fwd_packets": fwd_bytes,
            "total_length_bwd_packets": bwd_bytes,

            "fwd_packet_length_max": fwd_stats["max"],
            "fwd_packet_length_min": fwd_stats["min"],
            "fwd_packet_length_mean": fwd_stats["mean"],

            "bwd_packet_length_max": bwd_stats["max"],
            "bwd_packet_length_min": bwd_stats["min"],
            "bwd_packet_length_mean": bwd_stats["mean"],

            "flow_bytes_per_sec": total_bytes / duration_s,
            "flow_packets_per_sec": total_count / duration_s,

            "flow_iat_mean": float(flow_iats.mean()) if flow_iats.size else 0.0,
            "flow_iat_std": float(flow_iats.std()) if flow_iats.size else 0.0,
            "flow_iat_max": float(flow_iats.max()) if flow_iats.size else 0.0,
            "flow_iat_min": float(flow_iats.min()) if flow_iats.size else 0.0,

            "fwd_iat_total": float(fwd_iats.sum()) if fwd_iats.size else 0.0,
            "fwd_iat_mean": float(fwd_iats.mean()) if fwd_iats.size else 0.0,
            "fwd_iat_std": float(fwd_iats.std()) if fwd_iats.size else 0.0,
            "fwd_iat_max": float(fwd_iats.max()) if fwd_iats.size else 0.0,
            "fwd_iat_min": float(fwd_iats.min()) if fwd_iats.size else 0.0,

            "bwd_iat_total": float(bwd_iats.sum()) if bwd_iats.size else 0.0,
            "bwd_iat_mean": float(bwd_iats.mean()) if bwd_iats.size else 0.0,
            "bwd_iat_std": float(bwd_iats.std()) if bwd_iats.size else 0.0,
            "bwd_iat_max": float(bwd_iats.max()) if bwd_iats.size else 0.0,
            "bwd_iat_min": float(bwd_iats.min()) if bwd_iats.size else 0.0,

            "fwd_psh_flags": float(flow["fwd_psh_flags"]),
            "bwd_psh_flags": float(flow["bwd_psh_flags"]),
            "fwd_urg_flags": float(flow["fwd_urg_flags"]),
            "bwd_urg_flags": float(flow["bwd_urg_flags"]),

            "fwd_header_length": float(flow["fwd_header_len"]),
            "bwd_header_length": float(flow["bwd_header_len"]),

            "fwd_packets_per_sec": fwd_count / duration_s,
            "bwd_packets_per_sec": bwd_count / duration_s,

            "packet_length_min": packet_stats["min"],
            "packet_length_max": packet_stats["max"],
            "packet_length_mean": packet_stats["mean"],
            "packet_length_std": packet_stats["std"],
            "packet_length_variance": packet_stats["variance"],

            "fin_flag_count": float(flags["fin"]),
            "syn_flag_count": float(flags["syn"]),
            "rst_flag_count": float(flags["rst"]),
            "psh_flag_count": float(flags["psh"]),
            "ack_flag_count": float(flags["ack"]),
            "urg_flag_count": float(flags["urg"]),
            "cwe_flag_count": float(flags["cwe"]),
            "ece_flag_count": float(flags["ece"]),

            "down_up_ratio": (
                float(bwd_count / fwd_count) if fwd_count else 0.0
            ),

            "avg_packet_size": packet_stats["mean"],
            "avg_fwd_segment_size": fwd_stats["mean"],
            "avg_bwd_segment_size": bwd_stats["mean"],
        }

        return {
            "ts": flow["last_ts"],
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "proto": flow["proto"],
            "features": {
                name: float(vec.get(name, 0.0))
                for name in self.feature_order
            },
        }

    def flush_expired(
        self,
        now_ts: Optional[float] = None,
        flush_all: bool = False,
    ) -> List[dict]:
        """Finalize flows whose idle or active timeout has elapsed."""
        done_keys = []

        for key, flow in self.flows.items():
            if flush_all:
                done_keys.append(key)
                continue

            ref_ts = (
                float(now_ts)
                if now_ts is not None
                else flow["last_ts"]
            )

            idle_elapsed = ref_ts - flow["last_ts"]
            active_elapsed = ref_ts - flow["start_ts"]

            if (
                idle_elapsed >= self.idle_timeout_s
                or active_elapsed >= self.active_timeout_s
            ):
                done_keys.append(key)

        completed = []

        for key in done_keys:
            flow = self.flows.pop(key)
            completed.append(self._finalize_flow(flow))

        return completed
