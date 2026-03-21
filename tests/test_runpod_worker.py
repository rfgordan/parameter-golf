from __future__ import annotations

import unittest

from ops.runpod.worker import parse_metrics, slugify


class RunpodWorkerTests(unittest.TestCase):
    def test_slugify(self) -> None:
        self.assertEqual(slugify("My Run / seq=2048"), "My_Run_seq_2048")

    def test_parse_metrics(self) -> None:
        log_text = """
step:20/200 train_loss:1.2345 train_time:100ms step_avg:5.00ms
step:100/200 val_loss:1.1111 val_bpb:1.2222 train_time:500ms step_avg:5.00ms
Serialized model int8+zlib: 12345 bytes (payload:1 raw_torch:2 payload_ratio:3.00x)
Total submission size int8+zlib: 12456 bytes
final_int8_zlib_roundtrip_exact val_loss:1.01010101 val_bpb:1.02020202
"""
        metrics = parse_metrics(log_text)
        self.assertEqual(metrics["last_train_step"], 20)
        self.assertAlmostEqual(metrics["last_train_loss"], 1.2345)
        self.assertEqual(metrics["last_val_step"], 100)
        self.assertAlmostEqual(metrics["last_val_bpb"], 1.2222)
        self.assertAlmostEqual(metrics["final_roundtrip_val_bpb"], 1.02020202)
        self.assertEqual(metrics["artifact_total_submission_bytes"], 12456)
        self.assertEqual(metrics["artifact_compressed_model_bytes"], 12345)


if __name__ == "__main__":
    unittest.main()
