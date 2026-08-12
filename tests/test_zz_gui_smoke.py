import gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore, QtWidgets

from neutron_imaging_gui.main_window import MainWindow
from neutron_imaging_gui.export_dialogs import BatchExportDialog, CurrentImageExportDialog
from neutron_imaging_gui.processing import ReductionConfig
from neutron_imaging_gui.workers import ReductionQueue
from neutron_imaging_gui.widgets import integrated_profile


class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def tearDown(self):
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()
        gc.collect()

    def test_main_window_builds(self):
        window = MainWindow()
        self.assertEqual(window.pages.count(), 4)
        self.assertEqual(window.windowTitle(), "Neutron Imaging Reduction")
        self.assertEqual((window.width(), window.height()), (1500, 850))
        self.assertTrue(window.multiprocessing_check.isChecked())
        self.assertEqual(
            window.process_count_spin.value(), min(4, window.process_count_spin.maximum())
        )
        window.multiprocessing_check.setChecked(False)
        self.assertFalse(window.process_count_spin.isEnabled())
        window.close()
        self.app.processEvents()

    def test_input_dialogs_start_in_home_and_remember_selection(self):
        window = MainWindow()
        self.assertEqual(window.white_card._last_directory, str(Path.home()))
        self.assertEqual(window.dark_card._last_directory, str(Path.home()))
        self.assertEqual(window.sample_card._last_directory, str(Path.home()))
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "white.tif"
            tifffile.imwrite(image_path, np.ones((4, 4), dtype=np.float32))
            window.white_card.add_paths([image_path])
            self.assertEqual(Path(window.white_card._last_directory), Path(directory).resolve())
            self.assertEqual(Path(window.dark_card._last_directory), Path(directory).resolve())
            self.assertEqual(Path(window.sample_card._last_directory), Path(directory).resolve())
            self.assertEqual(window._last_directory, Path(directory).resolve())

            dark_directory = Path(directory) / "dark"
            dark_directory.mkdir()
            dark_path = dark_directory / "dark.tif"
            tifffile.imwrite(dark_path, np.zeros((4, 4), dtype=np.float32))
            with mock.patch.object(
                QtWidgets.QFileDialog,
                "getOpenFileNames",
                return_value=([str(dark_path)], "TIFF images (*.tif *.tiff)"),
            ) as chooser:
                window.dark_card._choose_files()
            self.assertEqual(Path(chooser.call_args.args[2]), Path(directory).resolve())
            self.assertEqual(window._last_directory, dark_directory.resolve())
            self.assertTrue(
                all(
                    Path(card._last_directory) == dark_directory.resolve()
                    for card in (window.white_card, window.dark_card, window.sample_card)
                )
            )
        window.close()
        self.app.processEvents()

    def test_integrated_profiles_use_expected_axes_and_statistics(self):
        image = np.arange(20, dtype=float).reshape(4, 5)
        x, vertical_mean, label = integrated_profile(
            image, (1, 1, 3, 2), integration="vertical", statistic="mean"
        )
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_allclose(vertical_mean, np.mean(image[1:3, 1:4], axis=0))
        self.assertEqual(label, "X pixel")

        y, horizontal_sum, label = integrated_profile(
            image, (1, 1, 3, 2), integration="horizontal", statistic="sum"
        )
        np.testing.assert_array_equal(y, [1, 2])
        np.testing.assert_allclose(horizontal_sum, np.sum(image[1:3, 1:4], axis=1))
        self.assertEqual(label, "Y pixel")

    def test_export_option_dialog_defaults_and_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            current = CurrentImageExportDialog(
                image=np.arange(20_000, dtype=np.float32).reshape(100, 200),
                image_name="sample_transmission",
                initial_directory=directory,
                colormap="magma",
                viewer_levels=(100.0, 10_000.0),
                profile_bounds=(10, 20, 30, 40),
            )
            self.assertEqual(current.format_combo.currentData(), "tiff")
            self.assertTrue(current.embed_metadata_check.isChecked())
            self.assertTrue(current.companion_json_check.isChecked())
            current.use_profile_roi_button.click()
            self.assertEqual(current.options().crop_bounds, (10, 20, 30, 40))
            current._refresh_preview()
            current.preview_canvas.draw()
            self.assertEqual(
                current.preview_figure.axes[0].images[0].get_array().shape,
                (40, 30),
            )
            self.assertEqual(current.preview_figure.axes[0].images[0].get_cmap().name, "gray")
            self.assertFalse(current.preview_figure.axes[0].axison)
            self.assertEqual(len(current.preview_figure.axes), 1)

            current.format_combo.setCurrentIndex(1)
            current._refresh_preview()
            self.assertEqual(
                current.preview_figure.axes[0].images[0].get_cmap().name,
                "magma",
            )
            self.assertFalse(current.preview_figure.axes[0].axison)
            self.assertEqual(len(current.preview_figure.axes), 1)

            current.format_combo.setCurrentIndex(2)
            current._refresh_preview()
            current.preview_canvas.draw()
            self.assertEqual(len(current.preview_figure.axes), 2)
            self.assertTrue(current.preview_figure.axes[0].axison)
            renderer = current.preview_canvas.get_renderer()
            image_height = current.preview_figure.axes[0].get_window_extent(renderer).height
            colorbar_height = current.preview_figure.axes[1].get_window_extent(renderer).height
            self.assertAlmostEqual(image_height, colorbar_height, places=6)
            self.assertEqual(current.options().format, "png_styled")
            self.assertEqual(current.options().path.suffix, ".png")
            current.close()

            batch = BatchExportDialog(initial_directory=directory)
            options = batch.options()
            self.assertEqual(
                options.categories,
                ("background", "combined", "transmission", "attenuation"),
            )
            self.assertTrue(options.use_subfolders)
            self.assertTrue(options.embed_tiff_metadata)
            self.assertTrue(options.companion_json)
            batch.close()

    def test_measurement_calibration_and_profile_roi(self):
        class MeasureEvent:
            def __init__(self, event_type, scene_position):
                self._event_type = event_type
                self._scene_position = scene_position
                self.accepted = False

            def type(self):
                return self._event_type

            @staticmethod
            def button():
                return QtCore.Qt.LeftButton

            def scenePos(self):
                return self._scene_position

            def accept(self):
                self.accepted = True

        window = MainWindow()
        window.preview.set_image(np.arange(200, dtype=np.float32).reshape(10, 20))
        window.preview.measure_check.setChecked(True)
        self.assertFalse(window.preview.measure_line.isVisible())
        self.assertIsNone(window.preview.measurement_length_px())
        view_box = window.preview.view.getView().getViewBox()
        start = view_box.mapViewToScene(QtCore.QPointF(5.0, 5.0))
        end = view_box.mapViewToScene(QtCore.QPointF(15.0, 5.0))
        for event_type, position in (
            (QtCore.QEvent.GraphicsSceneMousePress, start),
            (QtCore.QEvent.GraphicsSceneMouseMove, end),
            (QtCore.QEvent.GraphicsSceneMouseRelease, end),
        ):
            event = MeasureEvent(event_type, position)
            self.assertTrue(window.preview.eventFilter(window.preview._measure_scene, event))
            self.assertTrue(event.accepted)
        self.assertTrue(window.preview.measure_line.isVisible())
        self.assertAlmostEqual(window.preview.measurement_length_px(), 10.0)
        self.assertAlmostEqual(window.preview.set_calibration_from_line(1000.0), 100.0)
        self.assertIn("1 mm", window.preview.measure_readout.text())

        bounds = window.preview.profile_bounds()
        coordinates, values, label = integrated_profile(
            window.preview._current_image,
            bounds,
            integration=window.preview.profile_orientation_combo.currentData(),
            statistic=window.preview.profile_stat_combo.currentData(),
        )
        self.assertEqual(values.size, bounds[2])
        self.assertEqual(coordinates.size, values.size)
        self.assertEqual(label, "X pixel")
        window.preview.profile_orientation_combo.setCurrentIndex(1)
        window.preview.profile_stat_combo.setCurrentIndex(1)
        coordinates, values, label = integrated_profile(
            window.preview._current_image,
            bounds,
            integration=window.preview.profile_orientation_combo.currentData(),
            statistic=window.preview.profile_stat_combo.currentData(),
        )
        self.assertEqual(values.size, bounds[3])
        self.assertEqual(coordinates.size, values.size)
        self.assertEqual(label, "Y pixel")
        window.close()
        self.app.processEvents()

    def test_aaa_queue_runs_multiprocessing_from_background_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, value in (("white", 100.0), ("dark", 0.0), ("sample", 50.0)):
                path = root / f"{name}_0000.tif"
                tifffile.imwrite(path, np.full((8, 8), value, dtype=np.float32))
                paths[name] = str(path)
            config = ReductionConfig(
                white_files=(paths["white"],),
                dark_files=(paths["dark"],),
                sample_files=(paths["sample"],),
                merge_method="median",
                gamma_filter=False,
                use_multiprocessing=True,
                process_count=2,
            )
            queue = ReductionQueue()
            event_loop = QtCore.QEventLoop()
            results = []
            failures = []
            queue.succeeded.connect(
                lambda _job, result: (results.append(result), event_loop.quit())
            )
            queue.failed.connect(
                lambda _job, message, trace: (failures.append((message, trace)), event_loop.quit())
            )
            timeout = QtCore.QTimer()
            timeout.setSingleShot(True)
            timeout.timeout.connect(event_loop.quit)
            timeout.start(20_000)
            queue.submit(config)
            event_loop.exec_()
            queue.shutdown()
            self.assertFalse(failures, failures[0][1] if failures else "")
            self.assertTrue(results, "Multiprocessing reduction timed out")
            np.testing.assert_allclose(results[0].products["sample"].transmission, 0.5)

    def test_viewer_controls_without_side_histogram(self):
        window = MainWindow()
        window.preview.set_image(np.arange(100, dtype=np.float32).reshape(10, 10))
        self.assertFalse(window.preview.view.ui.histogram.isVisible())
        window.preview.colormap_combo.setCurrentIndex(1)
        self.assertTrue(window.preview.histogram_plot.isVisibleTo(window.preview))
        window.resize(2000, 900)
        window.show()
        self.app.processEvents()
        self.assertLessEqual(window.preview.histogram_plot.width(), 280)
        self.assertGreaterEqual(window.preview.histogram_plot.width(), 160)
        self.assertTrue(window.preview.auto_levels_check.isChecked())
        self.assertIs(
            window.preview.level_controls_stack.currentWidget(),
            window.preview.auto_levels_panel,
        )
        window.preview.low_percent_slider.setValue(20)
        window.preview.high_percent_slider.setValue(980)
        self.assertAlmostEqual(window.preview.low_percent_spin.value(), 2.0)
        self.assertAlmostEqual(window.preview.high_percent_spin.value(), 98.0)
        window.preview.auto_levels_check.setChecked(False)
        self.assertIs(
            window.preview.level_controls_stack.currentWidget(),
            window.preview.manual_levels_panel,
        )
        window.preview.minimum_spin.setValue(10.0)
        window.preview.maximum_spin.setValue(80.0)
        window.preview.minimum_slider.setValue(200)
        self.assertAlmostEqual(window.preview.minimum_spin.value(), 51.0, places=1)
        window.preview.gamma_slider.setValue(150)
        self.assertAlmostEqual(window.preview.gamma_spin.value(), 1.5)
        window.preview.reset_levels_button.click()
        self.assertTrue(window.preview.auto_levels_check.isChecked())
        self.assertAlmostEqual(window.preview.gamma_spin.value(), 1.0)
        window.close()
        self.app.processEvents()

    def test_auto_levels_toggle_restores_robust_bounds(self):
        window = MainWindow()
        image = np.linspace(100.0, 200.0, 10_000, dtype=np.float32).reshape(100, 100)
        image[0, 0] = -1e8
        image[-1, -1] = 1e8
        window.preview.set_image(image)
        window.preview.auto_levels_check.setChecked(False)
        window.preview.minimum_spin.setValue(-1e6)
        window.preview.maximum_spin.setValue(1e6)
        window.preview.auto_levels_check.setChecked(True)
        expected_low, expected_high = np.percentile(image, (1.0, 99.7))
        actual_low, actual_high = window.preview.levels()
        item_low, item_high = window.preview.view.getImageItem().getLevels()
        self.assertAlmostEqual(actual_low, expected_low, places=2)
        self.assertAlmostEqual(actual_high, expected_high, places=2)
        self.assertAlmostEqual(item_low, expected_low, places=2)
        self.assertAlmostEqual(item_high, expected_high, places=2)
        self.assertGreater(actual_low, 100.0)
        self.assertLess(actual_high, 201.0)
        window.close()
        self.app.processEvents()

    def test_auto_levels_handle_constant_and_normalized_images(self):
        window = MainWindow()
        window.preview.set_image(np.full((20, 20), 5.0, dtype=np.float32))
        self.assertEqual(window.preview.levels(), (5.0, 6.0))

        normalized = np.linspace(-0.2, 1.2, 2500, dtype=np.float32).reshape(50, 50)
        normalized[0, 0] = np.nan
        window.preview.set_image(normalized)
        finite = normalized[np.isfinite(normalized)]
        expected_low, expected_high = np.percentile(finite, (1.0, 99.7))
        actual_low, actual_high = window.preview.levels()
        self.assertAlmostEqual(actual_low, expected_low, places=3)
        self.assertAlmostEqual(actual_high, expected_high, places=3)
        window.close()
        self.app.processEvents()

    def test_manual_level_bounds_ignore_isolated_hot_pixel(self):
        window = MainWindow()
        image = np.linspace(1000.0, 2000.0, 10_000, dtype=np.float32).reshape(100, 100)
        image[0, 0] = 1e8
        window.preview.set_image(image)
        robust_levels = window.preview.levels()
        window.preview.auto_levels_check.setChecked(False)
        self.assertEqual(window.preview._level_low_bound, 0.0)
        self.assertEqual(window.preview._level_high_bound, 65535.0)
        self.assertAlmostEqual(window.preview.levels()[0], robust_levels[0], places=2)
        self.assertAlmostEqual(window.preview.levels()[1], robust_levels[1], places=2)
        window.preview.maximum_slider.setValue(1000)
        self.assertAlmostEqual(window.preview.maximum_spin.value(), 65535.0, places=1)
        window.close()
        self.app.processEvents()

    def test_double_click_fits_and_centers_image(self):
        class DoubleClickEvent:
            accepted = False

            @staticmethod
            def double():
                return True

            def accept(self):
                self.accepted = True

        window = MainWindow()
        window.preview.set_image(np.ones((50, 100), dtype=np.float32))
        view_box = window.preview.view.getView()
        view_box.setRange(xRange=(20, 30), yRange=(10, 20), padding=0)
        event = DoubleClickEvent()
        window.preview._scene_mouse_clicked(event)
        x_range, y_range = view_box.viewRange()
        self.assertTrue(event.accepted)
        self.assertLessEqual(x_range[0], 0)
        self.assertGreaterEqual(x_range[1], 100)
        self.assertLessEqual(y_range[0], 0)
        self.assertGreaterEqual(y_range[1], 50)
        window.close()
        self.app.processEvents()

    def test_histogram_ticks_range_and_double_click_reset(self):
        class HistogramDoubleClick:
            accepted = False

            @staticmethod
            def type():
                return QtCore.QEvent.MouseButtonDblClick

            def accept(self):
                self.accepted = True

        window = MainWindow()
        image = np.arange(65536, dtype=np.float32).reshape(256, 256)
        window.preview.set_image(image)
        self.assertEqual(window.preview._histogram_default_range, (0.0, 65535.0))
        self.assertEqual(window.preview._format_histogram_tick(20000), "20.0k")
        window.preview._histogram_user_view_active = True
        window.preview.histogram_plot.setXRange(1000.0, 2000.0, padding=0)
        event = HistogramDoubleClick()
        handled = window.preview.eventFilter(window.preview._histogram_viewport, event)
        visible = window.preview.histogram_plot.getPlotItem().viewRange()[0]
        self.assertTrue(handled)
        self.assertTrue(event.accepted)
        self.assertAlmostEqual(visible[0], 0.0, places=2)
        self.assertAlmostEqual(visible[1], 65535.0, places=2)
        self.assertFalse(window.preview._histogram_user_view_active)
        window.close()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
