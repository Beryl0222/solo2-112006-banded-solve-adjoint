from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from read_capture import read_capture


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"


class CaptureTests(unittest.TestCase):
    def test_event_order_and_dependency(self):
        self.assertEqual(np.__version__, "2.2.6")
        events = read_capture(CAPTURE)["events"]
        self.assertEqual([event[0] for event in events], ['solve', 'solve', 'publish', 'solve', 'back', 'back'])

    def test_invalid_event_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"events":[[]]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "事件列表"):
                read_capture(path)

    def test_command_reads_the_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "read_capture", str(CAPTURE)], cwd=ROOT,
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(len(result.stdout.splitlines()), 6)

    def test_capture_geometry_and_values(self):
        capture = read_capture(CAPTURE)
        ridge = capture["profiles"]["ridge"]
        self.assertEqual(len(capture["rhs"]["easy"]), len(ridge["active"]))
        self.assertEqual({len(row) for row in ridge["bands"]}, {len(capture["offsets"])})
        self.assertEqual(capture["slots"]["z"], [0, True])
