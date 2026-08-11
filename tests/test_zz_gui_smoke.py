import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore, QtWidgets

from neutron_imaging_gui.main_window import MainWindow


class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_main_window_builds(self):
        window = MainWindow()
        self.assertEqual(window.pages.count(), 4)
        self.assertEqual(window.windowTitle(), "Neutron Imaging Reduction")
        self.assertEqual((window.width(), window.height()), (1500, 850))
        window.close()
        self.app.processEvents()

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
