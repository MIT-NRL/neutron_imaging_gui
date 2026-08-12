import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile

from neutron_imaging_gui.tomography import (
    PreparedTomography,
    TOMOGRAPHY_RECIPE_SCHEMA,
    TomographyInput,
    backend_capabilities,
    default_recipe,
    inspect_dataset,
    load_preview_stack,
    load_recipe,
    manual_angles,
    prepare_tomography,
    reconstruction_preflight,
    run_full_reconstruction,
    run_tomopy_trial,
    save_recipe,
    scale_native_roi,
)


class TomographyCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name, value=1, angle=None, shape=(8, 10)):
        description = {} if angle is None else {"RotationAngle": angle}
        path = self.root / name
        tifffile.imwrite(
            path,
            np.full(shape, value, dtype=np.uint16),
            description=json.dumps(description),
            metadata=None,
        )
        return str(path)

    def _dataset(self):
        projections = tuple(
            self._write(f"StackTomo_{index:04d}.tif", index + 1, angle=index * 90)
            for index in range(4)
        )
        whites = (self._write("WhiteField_0000.tif", 100),)
        darks = (self._write("DarkField_0000.tif", 0),)
        return projections, whites, darks

    def test_leap_requires_a_detected_gpu_to_be_usable(self):
        leap = mock.Mock()
        leap.tomographicModels.return_value.number_of_gpus.return_value = 0
        original_find_spec = __import__("importlib.util").util.find_spec

        def find_spec(name):
            if name == "leapctype":
                return object()
            return original_find_spec(name)

        with mock.patch("neutron_imaging_gui.tomography.importlib.util.find_spec", side_effect=find_spec), mock.patch(
            "neutron_imaging_gui.tomography.importlib.import_module", return_value=leap
        ):
            capabilities = backend_capabilities()

        self.assertTrue(capabilities["leap_installed"])
        self.assertEqual(capabilities["leap_gpu_count"], 0)
        self.assertFalse(capabilities["leap"])

    def test_pattern_and_explicit_inputs_produce_the_same_manifest(self):
        projections, whites, darks = self._dataset()
        pattern = inspect_dataset(
            TomographyInput(
                mode="patterns",
                data_dir=str(self.root),
                projection_pattern="StackTomo*.tif",
                white_pattern="WhiteField*.tif",
                dark_pattern="DarkField*.tif",
            )
        )
        explicit = inspect_dataset(
            TomographyInput(
                mode="files",
                projection_files=projections,
                white_files=whites,
                dark_files=darks,
            )
        )
        self.assertEqual(pattern.files, explicit.files)
        self.assertEqual(pattern.shape, (8, 10))
        np.testing.assert_allclose(pattern.angles, [0, 90, 180, 270])
        self.assertEqual(pattern.angle_summary["median_step"], 90)

    def test_manual_angles_crop_binning_and_shape_validation(self):
        self._dataset()
        inputs = TomographyInput(mode="patterns", data_dir=str(self.root))
        manifest = inspect_dataset(
            inputs,
            angle_mode="manual",
            manual_start=0,
            manual_stop=180,
        )
        np.testing.assert_allclose(manifest.angles, [0, 45, 90, 135])
        preview = load_preview_stack(
            manifest,
            crop=[0, 2, 8, 10],
            binning=(2, 2),
            max_images=2,
        )
        self.assertEqual(preview.shape, (2, 4, 4))
        self.assertEqual(scale_native_roi([2, 4, 8, 10], (2, 2)), [1, 2, 4, 5])

        bad = self._write("StackTomo_bad.tif", shape=(7, 10), angle=300)
        self.assertTrue(Path(bad).exists())
        with self.assertRaisesRegex(ValueError, "Inconsistent TIFF shape"):
            inspect_dataset(inputs)

    def test_missing_metadata_requires_manual_fallback(self):
        projections, whites, darks = self._dataset()
        no_angle = self._write("no_angle.tif")
        inputs = TomographyInput(
            mode="files",
            projection_files=(projections[0], no_angle),
            white_files=whites,
            dark_files=darks,
        )
        with self.assertRaisesRegex(ValueError, "manual angle range"):
            inspect_dataset(inputs)
        self.assertEqual(len(manual_angles(2, 0, 180)), 2)

    def test_recipe_round_trip_and_schema_validation(self):
        recipe = default_recipe()
        recipe["paths"]["local"]["data_dir"] = str(self.root)
        recipe["paths"]["cluster"]["data_dir"] = "/cluster/data"
        recipe["loading"]["crop"] = [1, 2, 7, 9]
        path = save_recipe(recipe, self.root / "recipe.json")
        loaded = load_recipe(path)
        self.assertEqual(loaded["schema"], TOMOGRAPHY_RECIPE_SCHEMA)
        self.assertEqual(loaded["paths"]["cluster"]["data_dir"], "/cluster/data")
        self.assertEqual(loaded["loading"]["crop"], [1, 2, 7, 9])
        invalid = self.root / "invalid.json"
        invalid.write_text('{"schema": "wrong"}')
        with self.assertRaisesRegex(ValueError, "Unsupported tomography recipe"):
            load_recipe(invalid)

    def test_real_tomopy_cpu_trial(self):
        try:
            import tomopy
            from neutron_imaging_tools.reconstruction import TomographyData
        except ImportError:
            self.skipTest("TomoPy is not installed")
        theta = np.linspace(0, np.pi, 24, endpoint=False, dtype=np.float32)
        phantom = np.zeros((1, 24, 24), dtype=np.float32)
        phantom[:, 7:17, 8:16] = 1
        projections = tomopy.project(phantom, theta, pad=False)
        data = TomographyData(projections=projections, angles=np.rad2deg(theta))
        prepared = PreparedTomography(data=data, previews={}, diagnostics={}, elapsed_seconds=0)
        trial = run_tomopy_trial(prepared, method="gridrec", slice_index=0, ncore=1)
        self.assertEqual(trial.backend, "tomopy")
        self.assertEqual(trial.image.ndim, 2)
        self.assertTrue(np.all(np.isfinite(trial.image)))
        self.assertIn("gradient_energy", trial.metrics)

    def test_actual_nit_preparation_pipeline(self):
        projections = tuple(
            self._write(f"StackTomo_{index:04d}.tif", 40 + index, angle=index * 15, shape=(16, 20))
            for index in range(12)
        )
        whites = tuple(self._write(f"WhiteField_{index:04d}.tif", 100 + index, shape=(16, 20)) for index in range(3))
        darks = tuple(self._write(f"DarkField_{index:04d}.tif", 5 + index, shape=(16, 20)) for index in range(3))
        manifest = inspect_dataset(
            TomographyInput(mode="files", projection_files=projections, white_files=whites, dark_files=darks)
        )
        result = prepare_tomography(
            manifest,
            {
                "loading": {"crop": None, "binning": [1, 1], "skip_files": 1, "max_files": None, "dtype": "float32"},
                "reference_method": "median",
                "outlier": {"size": 3, "dif": "auto", "sigma_multiplier": 8, "backend": "auto", "threshold_mode": "shared", "calibration_frames": 4},
                "dose": {"enabled": True, "roi": [0, 0, 4, 4]},
                "stripe": {"enabled": True, "snr": 2, "la_size": 9, "sm_size": 5, "sizes_are_binned": False},
                "ncore": 1,
            },
        )
        self.assertEqual(result.data.projections.shape, (12, 16, 20))
        self.assertTrue(np.all(np.isfinite(result.data.projections)))
        self.assertIn("Merged white", result.previews)
        self.assertIn("Stripe corrected sinogram", result.previews)
        self.assertIn("Attenuation projection", result.previews)

    def test_leap_preflight_and_full_reconstruction_parameter_transfer(self):
        class Model:
            @staticmethod
            def z_samples():
                return np.arange(100)

            @staticmethod
            def get_numY():
                return 80

            @staticmethod
            def get_numX():
                return 60

            @staticmethod
            def number_of_gpus():
                return 2

        class Geometry:
            model = Model()

        class Data:
            projections = np.zeros((12, 20, 30), dtype=np.float32)

        prepared = PreparedTomography(
            data=Data(), previews={}, diagnostics={}, elapsed_seconds=0
        )
        settings = {
            "output_dir": str(self.root / "recon"),
            "base_filename": "sample",
            "z_range": [10, 50],
            "y_range": [5, 45],
            "x_range": [4, 34],
            "chunk_size": 12,
            "pad_each": 3,
            "method": "RWLS",
            "num_iter": 7,
            "preconditioner": "SQS",
            "regularization": {"tv_enabled": False},
            "diagnostics": False,
            "resume": True,
            "overwrite": False,
        }
        report = reconstruction_preflight(prepared, Geometry(), settings)
        self.assertEqual(report["output_shape"], [40, 40, 30])
        self.assertEqual(report["chunk_count"], 4)
        self.assertEqual(report["gpu_count"], 2)

        sentinel = object()
        with mock.patch(
            "neutron_imaging_tools.reconstruction.reconstruct_leap_volume",
            return_value=sentinel,
        ) as reconstruct:
            result = run_full_reconstruction(
                prepared, Geometry(), settings, manifest_metadata={"recipe": 1}
            )
        self.assertIs(result, sentinel)
        kwargs = reconstruct.call_args.kwargs
        self.assertEqual(kwargs["z_range"], [10, 50])
        self.assertEqual(kwargs["xy_crop"], (slice(5, 45), slice(4, 34)))
        self.assertEqual(kwargs["iterative_method"], "RWLS")
        self.assertEqual(kwargs["num_iter"], 7)
        self.assertTrue(kwargs["resume"])


if __name__ == "__main__":
    unittest.main()
