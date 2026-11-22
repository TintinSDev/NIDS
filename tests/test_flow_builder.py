import unittest

from src.features.flow_builder import (
    FlowAccumulator,
    load_schema_features,
)


class TestBidirectionalFlowAccumulator(unittest.TestCase):

    def setUp(self):
        self.acc = FlowAccumulator(
            idle_timeout_s=10,
            active_timeout_s=60,
        )

    def add_forward(self, ts, pkt_len=100, flags=None):
        return self.acc.add_packet(
            ts=ts,
            src_ip="192.168.1.10",
            dst_ip="192.168.1.20",
            src_port=50000,
            dst_port=443,
            proto="TCP",
            pkt_len=pkt_len,
            header_len=40,
            flags=flags or {},
        )

    def add_backward(self, ts, pkt_len=200, flags=None):
        return self.acc.add_packet(
            ts=ts,
            src_ip="192.168.1.20",
            dst_ip="192.168.1.10",
            src_port=443,
            dst_port=50000,
            proto="TCP",
            pkt_len=pkt_len,
            header_len=40,
            flags=flags or {},
        )

    def test_bidirectional_packets_form_one_flow(self):
        self.add_forward(1000.0, flags={"syn": 1})
        self.add_backward(1000.1, flags={"ack": 1})

        flows = self.acc.flush_expired(flush_all=True)

        self.assertEqual(len(flows), 1)

        features = flows[0]["features"]

        self.assertEqual(features["total_fwd_packets"], 1)
        self.assertEqual(features["total_bwd_packets"], 1)

        self.assertEqual(features["total_length_fwd_packets"], 100)
        self.assertEqual(features["total_length_bwd_packets"], 200)

        self.assertEqual(features["fwd_header_length"], 40)
        self.assertEqual(features["bwd_header_length"], 40)

        self.assertEqual(features["syn_flag_count"], 1)
        self.assertEqual(features["ack_flag_count"], 1)

    def test_multiple_packets_preserve_direction(self):
        self.add_forward(1000.0, 100)
        self.add_backward(1000.1, 200)
        self.add_forward(1000.2, 150)
        self.add_backward(1000.3, 250)

        flows = self.acc.flush_expired(flush_all=True)

        self.assertEqual(len(flows), 1)

        features = flows[0]["features"]

        self.assertEqual(features["total_fwd_packets"], 2)
        self.assertEqual(features["total_bwd_packets"], 2)

        self.assertEqual(features["total_length_fwd_packets"], 250)
        self.assertEqual(features["total_length_bwd_packets"], 450)

    def test_different_ports_produce_separate_flows(self):
        self.add_forward(1000.0)

        self.acc.add_packet(
            ts=1000.1,
            src_ip="192.168.1.10",
            dst_ip="192.168.1.20",
            src_port=50001,
            dst_port=443,
            proto="TCP",
            pkt_len=100,
            header_len=40,
        )

        flows = self.acc.flush_expired(flush_all=True)

        self.assertEqual(len(flows), 2)

    def test_single_packet_has_minimum_duration(self):
        self.add_forward(1000.0)

        flows = self.acc.flush_expired(flush_all=True)

        self.assertEqual(len(flows), 1)
        self.assertEqual(
            flows[0]["features"]["flow_duration"],
            1.0,
        )

    def test_schema_has_52_features(self):
        features = load_schema_features()

        self.assertEqual(len(features), 52)

        flow = self.acc
        self.add_forward(1000.0)

        result = flow.flush_expired(flush_all=True)[0]

        self.assertEqual(
            list(result["features"].keys()),
            features,
        )

    def test_tcp_directional_flags(self):
        self.add_forward(
            1000.0,
            flags={"syn": 1, "psh": 1},
        )

        self.add_backward(
            1000.1,
            flags={"ack": 1, "psh": 1},
        )

        flow = self.acc.flush_expired(flush_all=True)[0]
        features = flow["features"]

        self.assertEqual(features["fwd_psh_flags"], 1)
        self.assertEqual(features["bwd_psh_flags"], 1)
        self.assertEqual(features["syn_flag_count"], 1)
        self.assertEqual(features["ack_flag_count"], 1)


if __name__ == "__main__":
    unittest.main()
