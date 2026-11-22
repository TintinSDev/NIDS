import unittest
from unittest.mock import patch

from src.capture.live_capture import LiveCaptureWorker


class TestPcapReplayTiming(unittest.TestCase):

    def test_pcap_mode_does_not_expire_flows_using_wall_clock(self):
        worker = LiveCaptureWorker(
            api_url="http://127.0.0.1:8000",
            flush_interval_s=0,
            batch_size=100,
        )

        worker._pcap_mode = True

        # Simulate a flow from an older PCAP timestamp.
        worker.accumulator.add_packet(
            ts=1000.0,
            src_ip="192.168.1.10",
            dst_ip="192.168.1.20",
            src_port=50000,
            dst_port=443,
            proto="TCP",
            pkt_len=100,
            header_len=40,
        )

        # A second packet arrives 100 ms later in the same replay.
        parsed_packet = {
            "ts": 1000.1,
            "src_ip": "192.168.1.10",
            "dst_ip": "192.168.1.20",
            "src_port": 50000,
            "dst_port": 443,
            "proto": "TCP",
            "pkt_len": 120,
            "header_len": 40,
            "flags": {"ack": 1},
        }

        with patch(
            "src.capture.live_capture._extract_packet_fields",
            return_value=parsed_packet,
        ):
            worker.handle_packet(object())

        # The system clock must not prematurely expire this replay flow.
        self.assertEqual(len(worker.accumulator.flows), 1)

        active_flow = next(iter(worker.accumulator.flows.values()))
        self.assertEqual(len(active_flow["fwd_lengths"]), 2)

        # The flow should still be emitted normally at replay completion.
        completed = worker.accumulator.flush_expired(flush_all=True)

        self.assertEqual(len(completed), 1)
        self.assertEqual(
            completed[0]["features"]["total_fwd_packets"],
            2.0,
        )


class TestCaptureBatchDelivery(unittest.TestCase):

    def make_worker(self):
        worker = LiveCaptureWorker(
            api_url="http://127.0.0.1:8000",
            batch_size=1,
        )
        worker._pending_flows.append({
            "src_ip": "192.0.2.10",
            "dst_ip": "198.51.100.20",
            "src_port": 51000,
            "dst_port": 443,
            "proto": "TCP",
            "features": {},
        })
        return worker

    @patch("src.capture.live_capture._post_flows_to_api")
    def test_failed_batch_is_retained_and_successful_retry_removes_it(self, post):
        post.side_effect = [False, True]
        worker = self.make_worker()

        self.assertFalse(worker._emit_pending_batch(force=True))
        self.assertEqual(len(worker._pending_flows), 1)

        self.assertTrue(worker._emit_pending_batch(force=True))
        self.assertEqual(len(worker._pending_flows), 0)
        self.assertEqual(post.call_count, 2)

    @patch("src.capture.live_capture._post_flows_to_api", return_value=False)
    def test_shutdown_does_not_discard_failed_batch_or_loop_forever(self, post):
        worker = self.make_worker()

        worker.flush_remaining()

        self.assertEqual(len(worker._pending_flows), 1)
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
