import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from mstar_hrrp.core import read_phoenix_chip, extract_profile
from mstar_hrrp.cli import main


def chip(path, magnitude, phase):
    header = (f"NumberOfRows = {magnitude.shape[0]}\n"
              f"NumberOfColumns = {magnitude.shape[1]}\n"
              "TargetAz = 42.5\nEndofPhoenixHeader\n").encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(np.asarray(magnitude, dtype=">f4").tobytes())
        stream.write(np.asarray(phase, dtype=">f4").tobytes())


class ExtractionTests(unittest.TestCase):
    def test_chip_reader_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "HB00001.000"
            magnitude = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
            phase = np.full_like(magnitude, 0.3)
            chip(path, magnitude, phase)
            image, header = read_phoenix_chip(path)
            np.testing.assert_allclose(np.abs(image), magnitude, rtol=1e-6)
            self.assertEqual(header["TargetAz"], "42.5")
            result = extract_profile(image, target_len=128)
            self.assertEqual(result.shape, (128,))
            self.assertEqual(result.dtype, np.float32)
            self.assertAlmostEqual(float(result.max()), 1.0)
            self.assertTrue(np.all(result >= 0))

    def test_cli_outputs_and_bad_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "2S1"
            source.mkdir(parents=True)
            chip(source / "A.000", np.ones((8, 4)), np.zeros((8, 4)))
            (source / "B.001").write_bytes(b"bad")
            out = root / "output"
            main([str(root / "input"), str(out), "--skip-errors"])
            self.assertEqual(np.load(out / "hrrp.npy", allow_pickle=False).shape, (1, 128))
            with (out / "metadata.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["class_name"], "2S1")
            self.assertEqual(rows[0]["azimuth_deg"], "42.5")
            self.assertEqual(len(json.loads((out / "run.json").read_text())["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
