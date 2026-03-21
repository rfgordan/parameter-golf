from __future__ import annotations

import unittest

from ops.runpod.controller import parse_pod_id, parse_ssh_info, scp_base_args, ssh_base_args


class RunpodControllerTests(unittest.TestCase):
    def test_parse_pod_id(self) -> None:
        output = 'Created pod "abc123xyz456" successfully'
        self.assertEqual(parse_pod_id(output), "abc123xyz456")

    def test_parse_ssh_info(self) -> None:
        output = """
Some other text
ssh -i /tmp/key -p 22001 root@203.0.113.10
"""
        spec = parse_ssh_info(output)
        self.assertEqual(spec.host, "203.0.113.10")
        self.assertEqual(spec.user, "root")
        self.assertEqual(spec.port, 22001)
        self.assertEqual(spec.identity_file, "/tmp/key")
        self.assertEqual(
            ssh_base_args(spec),
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-i",
                "/tmp/key",
                "-p",
                "22001",
                "root@203.0.113.10",
            ],
        )
        self.assertEqual(
            scp_base_args(spec),
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-i",
                "/tmp/key",
                "-P",
                "22001",
            ],
        )


if __name__ == "__main__":
    unittest.main()
