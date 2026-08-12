from pathlib import Path
import threading
import tempfile
import unittest

import numpy as np
import tifffile

from neutron_imaging_gui.processing import (
    ProcessingCancelled,
    ReductionConfig,
    group_repeated_files,
    run_reduction,
)


def _write(path: Path, value):
    tifffile.imwrite(path, np.asarray(value, dtype=np.float32))
    return str(path)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    def test_group_repeated_files_keeps_uid_and_removes_exposure_index(self):
        paths = [
            self.temp_path / "sample_ab12cd34_0001.tif",
            self.temp_path / "sample_ab12cd34_0000.tif",
            self.temp_path / "other_0000.tif",
        ]
        groups = group_repeated_files(paths)
        self.assertEqual(list(groups), ["other", "sample_ab12cd34"])
        self.assertEqual(
            [Path(item).name for item in groups["sample_ab12cd34"]],
            ["sample_ab12cd34_0000.tif", "sample_ab12cd34_0001.tif"],
        )

    def test_combine_scans_merges_matching_sample_uids(self):
        paths = [
            self.temp_path / "sample_ab12cd34_0000.tif",
            self.temp_path / "sample_ab12cd34_0001.tif",
            self.temp_path / "sample_12ef5678_0000.tif",
        ]
        separate = group_repeated_files(paths)
        combined = group_repeated_files(paths, combine_scans=True)
        self.assertEqual(list(separate), ["sample_12ef5678", "sample_ab12cd34"])
        self.assertEqual(list(combined), ["sample"])
        self.assertEqual(len(combined["sample"]), 3)

    def test_basic_transmission_and_attenuation(self):
        shape = (8, 9)
        white = [
            _write(self.temp_path / f"white_{i}.tif", np.full(shape, 110.0))
            for i in range(2)
        ]
        dark = [
            _write(self.temp_path / f"dark_{i}.tif", np.full(shape, 10.0))
            for i in range(2)
        ]
        samples = [
            _write(self.temp_path / f"sample_uid12345_{i:04d}.tif", np.full(shape, 60.0))
            for i in range(2)
        ]
        config = ReductionConfig(
            white_files=tuple(white),
            dark_files=tuple(dark),
            sample_files=tuple(samples),
            merge_method="median",
            gamma_filter=False,
        )
        result = run_reduction(config)
        product = result.products["sample_uid12345"]
        np.testing.assert_allclose(product.transmission, 0.5)
        np.testing.assert_allclose(product.attenuation, -np.log(0.5), rtol=1e-6)
        self.assertIsNotNone(result.processing_started_utc)
        self.assertIsNotNone(result.processing_finished_utc)
        self.assertGreaterEqual(result.processing_elapsed_seconds, 0.0)

    def test_dose_normalization_matches_open_beam_roi(self):
        shape = (6, 6)
        white_image = np.full(shape, 110.0)
        dark_image = np.full(shape, 10.0)
        sample_image = np.full(shape, 60.0)
        sample_image[:2, :2] = 90.0
        config = ReductionConfig(
            white_files=(_write(self.temp_path / "white.tif", white_image),),
            dark_files=(_write(self.temp_path / "dark.tif", dark_image),),
            sample_files=(_write(self.temp_path / "sample_0000.tif", sample_image),),
            merge_method="median",
            gamma_filter=False,
            dose_normalization=True,
            dose_roi=(0, 0, 2, 2),
        )
        product = run_reduction(config).products["sample"]
        self.assertAlmostEqual(product.dose_scale, 0.8)
        np.testing.assert_allclose(product.transmission[:2, :2], 1.0)

    def test_per_group_dose_rois_are_applied(self):
        shape = (6, 6)
        white = np.full(shape, 110.0)
        dark = np.full(shape, 10.0)
        sample_a = np.full(shape, 60.0)
        sample_b = np.full(shape, 60.0)
        sample_a[:2, :2] = 90.0
        sample_b[-2:, -2:] = 80.0
        config = ReductionConfig(
            white_files=(_write(self.temp_path / "white.tif", white),),
            dark_files=(_write(self.temp_path / "dark.tif", dark),),
            sample_files=(
                _write(self.temp_path / "sample_a_0000.tif", sample_a),
                _write(self.temp_path / "sample_b_0000.tif", sample_b),
            ),
            merge_method="median",
            gamma_filter=False,
            dose_normalization=True,
            dose_roi=(0, 0, 2, 2),
            dose_rois={"sample_a": (0, 0, 2, 2), "sample_b": (4, 4, 2, 2)},
        )
        result = run_reduction(config)
        self.assertAlmostEqual(result.products["sample_a"].dose_scale, 0.8)
        self.assertAlmostEqual(result.products["sample_b"].dose_scale, 0.7)

    def test_cancelled_before_start(self):
        event = threading.Event()
        event.set()
        config = ReductionConfig(
            white_files=("white.tif",),
            dark_files=("dark.tif",),
            sample_files=("sample.tif",),
        )
        with self.assertRaises(ProcessingCancelled):
            run_reduction(config, cancel_event=event)

    def test_progress_counts_loading_and_processing_stages(self):
        shape = (5, 5)
        config = ReductionConfig(
            white_files=(_write(self.temp_path / "white.tif", np.full(shape, 100.0)),),
            dark_files=(_write(self.temp_path / "dark.tif", np.zeros(shape)),),
            sample_files=(_write(self.temp_path / "sample_0000.tif", np.full(shape, 50.0)),),
            merge_method="median",
            gamma_filter=False,
        )
        updates = []
        run_reduction(config, progress=lambda *args: updates.append(args))
        self.assertEqual(updates[-1][0], "done")
        self.assertEqual(updates[-1][1], updates[-1][2])
        messages = [update[3] for update in updates]
        self.assertIn("Loaded white.tif", messages)
        self.assertIn("Merged white field", messages)
        self.assertIn("Transmission ready for sample", messages)

    def test_multiprocessing_matches_serial_and_reports_image_progress(self):
        shape = (12, 13)
        white = tuple(
            _write(self.temp_path / f"white_{i}.tif", np.full(shape, 110.0 + i))
            for i in range(2)
        )
        dark = tuple(
            _write(self.temp_path / f"dark_{i}.tif", np.full(shape, 10.0 + i))
            for i in range(2)
        )
        samples = tuple(
            _write(self.temp_path / f"sample_{name}_{i:04d}.tif", np.full(shape, value + i))
            for name, value in (("a", 60.0), ("b", 35.0))
            for i in range(2)
        )
        base = dict(
            white_files=white,
            dark_files=dark,
            sample_files=samples,
            merge_method="median",
            gamma_filter=False,
        )
        serial = run_reduction(ReductionConfig(**base))
        updates = []
        parallel = run_reduction(
            ReductionConfig(**base, use_multiprocessing=True, process_count=2),
            progress=lambda *args: updates.append(args),
        )
        self.assertEqual(list(parallel.products), ["sample_a", "sample_b"])
        for name in serial.products:
            np.testing.assert_allclose(
                parallel.products[name].attenuation,
                serial.products[name].attenuation,
            )
        self.assertTrue(any("Starting 2 background process" in item[3] for item in updates))
        loaded = [item for item in updates if item[3].startswith("Loaded ")]
        self.assertEqual(len(loaded), 8)
        self.assertEqual(updates[-1][1], updates[-1][2])


if __name__ == "__main__":
    unittest.main()
