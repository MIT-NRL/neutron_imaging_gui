import gc
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore, QtWidgets

from neutron_imaging_gui.main_window import MainWindow
from neutron_imaging_gui.tomography import ReconstructionTrial, default_recipe
from neutron_imaging_gui.tomography_workspace import TomographyWorkspace
from neutron_imaging_gui.tomography_workers import TomographyJobRunner


class TomographyGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def tearDown(self):
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()
        gc.collect()

    def test_aaa_actual_preparation_runs_in_background_worker(self):
        from neutron_imaging_gui.tomography import TomographyInput, inspect_dataset, prepare_tomography

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projections = []
            for index in range(8):
                path = root / f"StackTomo_{index:04d}.tif"
                tifffile.imwrite(path, np.full((12, 16), 40 + index, dtype=np.uint16))
                projections.append(str(path))
            white = root / "WhiteField_0000.tif"
            dark = root / "DarkField_0000.tif"
            tifffile.imwrite(white, np.full((12, 16), 100, dtype=np.uint16))
            tifffile.imwrite(dark, np.full((12, 16), 5, dtype=np.uint16))
            manifest = inspect_dataset(
                TomographyInput(
                    mode="files",
                    projection_files=tuple(projections),
                    white_files=(str(white),),
                    dark_files=(str(dark),),
                ),
                angle_mode="manual",
                manual_start=0,
                manual_stop=180,
            )
            settings = {
                "loading": {"crop": None, "binning": [1, 1], "dtype": "float32"},
                "reference_method": "median",
                "outlier": {"fields": ()},
                "dose": {"enabled": False, "roi": None},
                "stripe": {"enabled": False},
                "ncore": 1,
            }
            runner = TomographyJobRunner()
            loop = QtCore.QEventLoop()
            results, failures = [], []
            runner.succeeded.connect(
                lambda _kind, result: (results.append(result), loop.quit())
            )
            runner.failed.connect(
                lambda _kind, message, trace: (
                    failures.append((message, trace)),
                    loop.quit(),
                )
            )
            timeout = QtCore.QTimer()
            timeout.setSingleShot(True)
            timeout.timeout.connect(loop.quit)
            timeout.start(15_000)
            runner.start(
                "prepare",
                lambda **kwargs: prepare_tomography(manifest, settings, **kwargs),
            )
            loop.exec_()
            runner.shutdown()
            self.assertFalse(failures, failures[0][1] if failures else "")
            self.assertTrue(results, "Background preparation timed out")
            self.assertEqual(results[0].data.projections.shape, (8, 12, 16))

    def test_main_window_has_separate_tomography_workspace(self):
        window = MainWindow()
        self.assertEqual(window.workspace_tabs.count(), 2)
        self.assertEqual(window.workspace_tabs.tabText(0), "Radiography")
        self.assertEqual(window.workspace_tabs.tabText(1), "Tomography")
        self.assertEqual(window.tomography.pages.count(), 5)
        self.assertEqual(window.tomography.input_mode_combo.currentData(), "patterns")
        self.assertEqual(window.tomography.reference_combo.currentData(), "mad_adaptive")
        self.assertEqual(window.tomography.stripe_large_spin.value(), 163)
        self.assertEqual(window.tomography.stripe_small_spin.value(), 31)
        window.close()
        self.app.processEvents()

    def test_tomography_background_runner_progress_and_shared_busy_state(self):
        window = MainWindow()
        runner = TomographyJobRunner()
        loop = QtCore.QEventLoop()
        progress = []
        results = []
        failures = []
        runner.progress.connect(lambda *values: progress.append(values))
        runner.succeeded.connect(
            lambda kind, result: (results.append((kind, result)), loop.quit())
        )
        runner.failed.connect(
            lambda kind, message, trace: (failures.append((kind, message, trace)), loop.quit())
        )

        def operation(*, cancel_event, progress):
            progress("test", 1, 2, "halfway")
            progress("test", 2, 2, "done")
            return 42

        runner.start("test", operation)
        window._tomography_busy_changed(True)
        self.assertFalse(window.run_button.isEnabled())
        timeout = QtCore.QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        timeout.start(5000)
        loop.exec_()
        runner.shutdown()
        self.assertFalse(failures, failures[0][2] if failures else "")
        self.assertEqual(results, [("test", 42)])
        self.assertEqual(progress[-1], ("test", 2, 2, "done"))
        window.close()
        self.app.processEvents()

    def test_recipe_application_restores_controls_without_loading(self):
        workspace = TomographyWorkspace()
        recipe = default_recipe()
        recipe["paths"]["local"]["data_dir"] = "/local/data"
        recipe["paths"]["cluster"]["data_dir"] = "/cluster/data"
        recipe["paths"]["cluster"]["output_dir"] = "/cluster/output"
        recipe["input"]["angle_source"] = "manual"
        recipe["input"]["manual_angles"] = {"start": -90, "stop": 90, "endpoint": True}
        recipe["loading"]["binning"] = [2, 4]
        recipe["loading"]["crop"] = [20, 40, 220, 440]
        recipe["preparation"]["dose"] = {"enabled": True, "roi": [40, 80, 100, 160]}
        recipe["geometry"]["pixel_size_mm"] = 0.125
        recipe["volume"]["z_range"] = [10, 30]
        workspace.apply_recipe(recipe)
        self.assertEqual(workspace.data_directory_edit.text(), "/local/data")
        self.assertEqual(workspace.cluster_data_edit.text(), "/cluster/data")
        self.assertEqual(workspace.cluster_output_edit.text(), "/cluster/output")
        self.assertEqual(workspace.angle_mode_combo.currentData(), "manual")
        self.assertEqual((workspace.ybin_spin.value(), workspace.xbin_spin.value()), (2, 4))
        self.assertTrue(workspace.crop_check.isChecked())
        self.assertTrue(workspace.dose_check.isChecked())
        self.assertEqual(workspace.z_range_edit.text(), "10, 30")
        self.assertIsNone(workspace._manifest)
        workspace.close()
        workspace.prepare_close()

    def test_native_roi_scaling_and_trial_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (
                ("StackTomo_0000.tif", 10),
                ("WhiteField_0000.tif", 100),
                ("DarkField_0000.tif", 0),
            ):
                tifffile.imwrite(root / name, np.full((100, 200), value, dtype=np.uint16))
            workspace = TomographyWorkspace(initial_directory=root)
            workspace.data_directory_edit.setText(str(root))
            workspace.angle_mode_combo.setCurrentIndex(
                workspace.angle_mode_combo.findData("manual")
            )
            from neutron_imaging_gui.tomography import inspect_dataset

            workspace._manifest = inspect_dataset(
                workspace._input_spec(), angle_mode="manual", manual_start=0, manual_stop=180
            )
            workspace.ybin_spin.setValue(2)
            workspace.xbin_spin.setValue(4)
            workspace._initialize_rois((50, 50))
            workspace.crop_check.setChecked(True)
            workspace.crop_roi.setPos((5, 10), update=False)
            workspace.crop_roi.setSize((20, 15), update=True)
            self.assertEqual(workspace._current_crop(), [20, 20, 50, 100])

            trial = ReconstructionTrial(
                name="gridrec test",
                backend="tomopy",
                method="gridrec",
                image=np.ones((16, 16), dtype=np.float32),
                parameters={"slice_index": 10},
                elapsed_seconds=0.25,
            )
            workspace._trials.append(trial)
            workspace.trial_list.addItem("")
            workspace._refresh_trial_labels()
            workspace.trial_list.setCurrentRow(0)
            workspace._promote_trial()
            workspace.local_output_edit.setText(str(root / "local_recon"))
            workspace.cluster_output_edit.setText("/cluster/recon")
            recipe = workspace.build_recipe()
            self.assertEqual(recipe["selected_trial"]["name"], "gridrec test")
            self.assertEqual(recipe["loading"]["crop"], [20, 20, 50, 100])
            self.assertEqual(recipe["paths"]["local"]["output_dir"], str(root / "local_recon"))
            self.assertEqual(recipe["volume"]["output_dir"], "/cluster/recon")
            workspace.close()
            workspace.prepare_close()


if __name__ == "__main__":
    unittest.main()
