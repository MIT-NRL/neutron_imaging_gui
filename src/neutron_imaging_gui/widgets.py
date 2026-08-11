"""Reusable GUI widgets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from .processing import discover_tiffs


class FileSelectionCard(QtWidgets.QGroupBox):
    filesChanged = QtCore.Signal()
    previewRequested = QtCore.Signal(str)

    def __init__(self, title: str, description: str, parent=None):
        super().__init__(title, parent)
        self._paths: list[str] = []
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(description)
        label.setWordWrap(True)
        label.setProperty("muted", True)
        layout.addWidget(label)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setMinimumHeight(90)
        self.list_widget.currentItemChanged.connect(
            lambda item, _previous: (
                self.previewRequested.emit(item.data(QtCore.Qt.UserRole)) if item is not None else None
            )
        )
        layout.addWidget(self.list_widget)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.add_files_button = QtWidgets.QPushButton("Add files…")
        self.add_folder_button = QtWidgets.QPushButton("Add folder…")
        self.clear_button = QtWidgets.QPushButton("Clear")
        for button in (self.add_files_button, self.add_folder_button, self.clear_button):
            button.setStyleSheet("padding: 6px 8px;")
        self.count_label = QtWidgets.QLabel("0 files")
        self.count_label.setMinimumWidth(42)
        row.addWidget(self.add_files_button)
        row.addWidget(self.add_folder_button)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        row.addWidget(self.count_label)
        layout.addLayout(row)

        self.add_files_button.clicked.connect(self._choose_files)
        self.add_folder_button.clicked.connect(self._choose_folder)
        self.clear_button.clicked.connect(self.clear)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(self._paths)

    def add_paths(self, values) -> None:
        merged = discover_tiffs([*self._paths, *values])
        self._paths = merged
        self._refresh(select_first=True)

    def clear(self) -> None:
        self._paths = []
        self._refresh()

    def _choose_files(self) -> None:
        values, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select TIFF images", "", "TIFF images (*.tif *.tiff)"
        )
        if values:
            self.add_paths(values)

    def _choose_folder(self) -> None:
        value = QtWidgets.QFileDialog.getExistingDirectory(self, "Select image folder")
        if value:
            self.add_paths([value])

    def _refresh(self, *, select_first=False) -> None:
        self.list_widget.clear()
        for path in self._paths:
            item = QtWidgets.QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            item.setData(QtCore.Qt.UserRole, path)
            self.list_widget.addItem(item)
        self.count_label.setText(f"{len(self._paths)} file{'s' if len(self._paths) != 1 else ''}")
        self.filesChanged.emit()
        if select_first and self._paths:
            self.list_widget.setCurrentRow(0)


class ImagePreview(QtWidgets.QWidget):
    roiChanged = QtCore.Signal(tuple)
    previousRequested = QtCore.Signal()
    nextRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None
        self._level_low_bound = 0.0
        self._level_high_bound = 1.0
        self._default_low_percentile = 1.0
        self._default_high_percentile = 99.7
        self._default_gamma = 1.0
        self._base_lut = None
        self._histogram_user_view_active = False
        self._histogram_default_range = (0.0, 1.0)
        self._histogram_normalized = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = pg.ImageView(view=pg.PlotItem())
        self.view.getView().setAspectLocked(True)
        self.view.setToolTip("Double-click the image to fit and center it in the viewer.")
        if hasattr(self.view, "ui"):
            self.view.ui.histogram.hide()
            self.view.ui.roiBtn.hide()
            self.view.ui.menuBtn.hide()
        scene = self.view.getView().scene()
        if scene is not None and hasattr(scene, "sigMouseClicked"):
            scene.sigMouseClicked.connect(self._scene_mouse_clicked)
        layout.addWidget(self.view, 1)
        self.roi = pg.RectROI((10, 10), (100, 100), pen=pg.mkPen("#f0ad4e", width=2))
        self.roi.addScaleHandle((1, 1), (0, 0))
        self.roi.addScaleHandle((0, 0), (1, 1))
        self.view.getView().addItem(self.roi)
        self.roi.hide()
        self.roi.sigRegionChangeFinished.connect(lambda: self.roiChanged.emit(self.roi_bounds()))
        self.caption = QtWidgets.QLabel("No image selected")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        navigation = QtWidgets.QWidget()
        navigation_layout = QtWidgets.QHBoxLayout(navigation)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(4)
        self.previous_button = QtWidgets.QToolButton()
        self.previous_button.setText("◀")
        self.previous_button.setToolTip("Previous image")
        self.position_label = QtWidgets.QLabel("0 / 0")
        self.position_label.setMinimumWidth(54)
        self.position_label.setAlignment(QtCore.Qt.AlignCenter)
        self.next_button = QtWidgets.QToolButton()
        self.next_button.setText("▶")
        self.next_button.setToolTip("Next image")
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addWidget(self.position_label)
        navigation_layout.addWidget(self.next_button)
        navigation.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.auto_levels_check = QtWidgets.QCheckBox("Auto levels")
        self.auto_levels_check.setChecked(True)
        self.auto_levels_check.setToolTip(
            "Use robust image percentiles for the displayed minimum and maximum."
        )
        self.reset_levels_button = QtWidgets.QPushButton("Reset Levels")
        auto_block = QtWidgets.QWidget()
        auto_layout = QtWidgets.QVBoxLayout(auto_block)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.setSpacing(2)
        auto_layout.addWidget(self.auto_levels_check)
        auto_layout.addWidget(self.reset_levels_button)
        auto_block.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.minimum_spin = self._level_spinbox()
        self.maximum_spin = self._level_spinbox()
        self.minimum_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.maximum_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        for slider in (self.minimum_slider, self.maximum_slider):
            slider.setRange(0, 1000)
        minimum_block = self._slider_block("Min", self.minimum_spin, self.minimum_slider)
        maximum_block = self._slider_block("Max", self.maximum_spin, self.maximum_slider)

        self.low_percent_spin = QtWidgets.QDoubleSpinBox()
        self.low_percent_spin.setDecimals(2)
        self.low_percent_spin.setRange(0.0, 5.0)
        self.low_percent_spin.setSingleStep(0.1)
        self.low_percent_spin.setValue(self._default_low_percentile)
        self.low_percent_spin.setFixedWidth(60)
        self.low_percent_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.low_percent_slider.setRange(0, 50)
        self.low_percent_slider.setValue(round(self._default_low_percentile * 10))
        low_percent_block = self._slider_block(
            "Low %", self.low_percent_spin, self.low_percent_slider
        )

        self.high_percent_spin = QtWidgets.QDoubleSpinBox()
        self.high_percent_spin.setDecimals(2)
        self.high_percent_spin.setRange(95.0, 100.0)
        self.high_percent_spin.setSingleStep(0.1)
        self.high_percent_spin.setValue(self._default_high_percentile)
        self.high_percent_spin.setFixedWidth(60)
        self.high_percent_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.high_percent_slider.setRange(950, 1000)
        self.high_percent_slider.setValue(round(self._default_high_percentile * 10))
        high_percent_block = self._slider_block(
            "High %", self.high_percent_spin, self.high_percent_slider
        )

        self.manual_levels_panel = QtWidgets.QWidget()
        manual_levels_layout = QtWidgets.QHBoxLayout(self.manual_levels_panel)
        manual_levels_layout.setContentsMargins(0, 0, 0, 0)
        manual_levels_layout.setSpacing(6)
        manual_levels_layout.addWidget(minimum_block)
        manual_levels_layout.addWidget(maximum_block)

        self.auto_levels_panel = QtWidgets.QWidget()
        auto_levels_layout = QtWidgets.QHBoxLayout(self.auto_levels_panel)
        auto_levels_layout.setContentsMargins(0, 0, 0, 0)
        auto_levels_layout.setSpacing(6)
        auto_levels_layout.addWidget(low_percent_block)
        auto_levels_layout.addWidget(high_percent_block)

        self.level_controls_stack = QtWidgets.QStackedWidget()
        self.level_controls_stack.addWidget(self.manual_levels_panel)
        self.level_controls_stack.addWidget(self.auto_levels_panel)
        self.level_controls_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )

        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(self._default_gamma)
        self.gamma_spin.setFixedWidth(60)
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(10, 500)
        self.gamma_slider.setValue(round(self._default_gamma * 100))
        gamma_block = self._slider_block("Gamma", self.gamma_spin, self.gamma_slider)
        gamma_block.setToolTip("Display-only gamma; this does not change processed data.")

        self.colormap_combo = QtWidgets.QComboBox()
        for label, name in (
            ("Gray", "gray"),
            ("Viridis", "viridis"),
            ("Magma", "magma"),
            ("Inferno", "inferno"),
            ("Plasma", "plasma"),
            ("Cividis", "cividis"),
        ):
            self.colormap_combo.addItem(label, name)
        self.colormap_combo.setFixedWidth(90)
        colormap_block = QtWidgets.QWidget()
        colormap_layout = QtWidgets.QVBoxLayout(colormap_block)
        colormap_layout.setContentsMargins(0, 0, 0, 0)
        colormap_layout.setSpacing(2)
        colormap_layout.addWidget(QtWidgets.QLabel("Color map"))
        colormap_layout.addWidget(self.colormap_combo)
        colormap_block.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.histogram_plot = pg.PlotWidget()
        self.histogram_plot.setMinimumWidth(160)
        self.histogram_plot.setMaximumWidth(280)
        self.histogram_plot.setMinimumHeight(72)
        self.histogram_plot.setMaximumHeight(72)
        self.histogram_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.histogram_plot.setMouseEnabled(x=True, y=False)
        self.histogram_plot.hideAxis("left")
        histogram_item = self.histogram_plot.getPlotItem()
        histogram_item.setMenuEnabled(False)
        histogram_item.hideButtons()
        histogram_axis = histogram_item.getAxis("bottom")
        tick_font = histogram_axis.style.get("tickFont") or self.font()
        tick_font.setPointSize(8)
        histogram_axis.setTickFont(tick_font)
        histogram_axis.setStyle(tickTextOffset=2)
        self._histogram_axis = histogram_axis
        histogram_view = histogram_item.getViewBox()
        pan_mode = getattr(histogram_view, "PanMode", None)
        if pan_mode is not None:
            histogram_view.setMouseMode(pan_mode)
        histogram_view.setMouseEnabled(x=True, y=False)
        histogram_view.sigXRangeChanged.connect(self._histogram_x_range_changed)
        self._histogram_viewport = self.histogram_plot.viewport()
        self._histogram_viewport.installEventFilter(self)
        self.histogram_plot.setToolTip(
            "Scroll or drag horizontally to inspect the intensity range. Double-click to reset."
        )
        self.histogram_curve = self.histogram_plot.plot(
            pen=pg.mkPen("#2f6f9f", width=1.5),
            fillLevel=0,
            brush=pg.mkBrush(47, 111, 159, 60),
        )
        self.histogram_low_line = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen("#c83737", width=1)
        )
        self.histogram_high_line = pg.InfiniteLine(
            pos=1, angle=90, movable=False, pen=pg.mkPen("#159447", width=1)
        )
        histogram_item.addItem(self.histogram_low_line)
        histogram_item.addItem(self.histogram_high_line)

        controls_layout.addWidget(navigation)
        controls_layout.addWidget(auto_block)
        controls_layout.addWidget(self.level_controls_stack)
        controls_layout.addWidget(gamma_block)
        controls_layout.addWidget(colormap_block)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.histogram_plot)
        controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        controls.setMaximumHeight(88)
        layout.addWidget(controls)

        self.previous_button.clicked.connect(self.previousRequested.emit)
        self.next_button.clicked.connect(self.nextRequested.emit)
        self.auto_levels_check.toggled.connect(self._auto_levels_toggled)
        self.reset_levels_button.clicked.connect(self.reset_levels)
        self.minimum_spin.valueChanged.connect(self._manual_levels_changed)
        self.maximum_spin.valueChanged.connect(self._manual_levels_changed)
        self.minimum_spin.valueChanged.connect(self._minimum_spin_changed)
        self.maximum_spin.valueChanged.connect(self._maximum_spin_changed)
        self.minimum_slider.valueChanged.connect(self._minimum_slider_changed)
        self.maximum_slider.valueChanged.connect(self._maximum_slider_changed)
        self.low_percent_spin.valueChanged.connect(self._auto_percentiles_changed)
        self.high_percent_spin.valueChanged.connect(self._auto_percentiles_changed)
        self.low_percent_spin.valueChanged.connect(self._low_percent_spin_changed)
        self.high_percent_spin.valueChanged.connect(self._high_percent_spin_changed)
        self.low_percent_slider.valueChanged.connect(self._low_percent_slider_changed)
        self.high_percent_slider.valueChanged.connect(self._high_percent_slider_changed)
        self.gamma_spin.valueChanged.connect(self._gamma_spin_changed)
        self.gamma_slider.valueChanged.connect(self._gamma_slider_changed)
        self.colormap_combo.currentIndexChanged.connect(self._apply_colormap)
        self._auto_levels_toggled(True)

    def set_image(self, image: np.ndarray, caption: str = "") -> None:
        array = np.asarray(image, dtype=np.float32)
        self._current_image = array
        self._configure_level_controls(array)
        levels = self._auto_level_values(array) if self.auto_levels_check.isChecked() else self.levels()
        self.view.setImage(array.T, autoLevels=False, levels=levels)
        self._apply_colormap()
        self._update_histogram(array)
        self.caption.setText(caption or f"{array.shape[1]} × {array.shape[0]}")
        if self.auto_levels_check.isChecked():
            QtCore.QTimer.singleShot(0, self.apply_auto_levels)

    def set_roi_visible(self, visible: bool) -> None:
        self.roi.setVisible(bool(visible))

    def roi_bounds(self) -> tuple[int, int, int, int]:
        pos = self.roi.pos()
        size = self.roi.size()
        return (round(pos.x()), round(pos.y()), max(1, round(size.x())), max(1, round(size.y())))

    def set_roi_bounds(self, bounds) -> None:
        x, y, width, height = bounds
        self.roi.setPos((float(x), float(y)), update=False)
        self.roi.setSize((float(width), float(height)), update=True)

    def set_position(self, current: int, total: int) -> None:
        self.position_label.setText(f"{current} / {total}")
        self.previous_button.setEnabled(total > 1)
        self.next_button.setEnabled(total > 1)

    def fit_image(self) -> None:
        """Fit and center the complete image while preserving its aspect ratio."""
        if self._current_image is None:
            return
        self.view.getView().autoRange(padding=0.02)

    def _scene_mouse_clicked(self, event) -> None:
        try:
            is_double_click = bool(event.double())
        except Exception:
            is_double_click = False
        if not is_double_click:
            return
        self.fit_image()
        try:
            event.accept()
        except Exception:
            pass

    def levels(self):
        low = float(self.minimum_spin.value())
        high = float(self.maximum_spin.value())
        return (low, high if high > low else low + 1.0)

    @staticmethod
    def _level_spinbox():
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e15, 1e15)
        spin.setDecimals(3)
        spin.setFixedWidth(70)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _slider_block(label, spinbox, slider):
        block = QtWidgets.QWidget()
        block.setMinimumWidth(104)
        block.setMaximumWidth(120)
        block_layout = QtWidgets.QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        top.addWidget(QtWidgets.QLabel(label))
        top.addWidget(spinbox)
        block_layout.addLayout(top)
        block_layout.addWidget(slider)
        return block

    def _auto_level_values(self, array):
        finite = np.asarray(array)[np.isfinite(array)]
        if not finite.size:
            return (0.0, 1.0)
        low, high = np.percentile(
            finite,
            (self.low_percent_spin.value(), self.high_percent_spin.value()),
        )
        if high <= low:
            high = low + max(1.0, abs(float(low)) * 0.01)
        return float(low), float(high)

    def _configure_level_controls(self, array):
        finite = array[np.isfinite(array)]
        if not finite.size:
            return
        bound_low, bound_high, _normalized = self._histogram_data_range(finite)
        auto_low, auto_high = self._auto_level_values(array)
        self._level_low_bound = min(float(bound_low), float(auto_low))
        self._level_high_bound = max(float(bound_high), float(auto_high))
        span = self._level_high_bound - self._level_low_bound
        if span <= 0:
            span = max(1.0, abs(self._level_low_bound) * 0.01)
            self._level_high_bound = self._level_low_bound + span
        decimals = 6 if span < 1 else 3
        for spin in (self.minimum_spin, self.maximum_spin):
            spin.setDecimals(decimals)
            spin.setSingleStep(span / 100.0)
            spin.setRange(self._level_low_bound, self._level_high_bound)
        if self.auto_levels_check.isChecked():
            self._set_level_values(*self._auto_level_values(array))
        self._sync_manual_sliders()

    def _set_level_values(self, low, high):
        self.minimum_spin.blockSignals(True)
        self.maximum_spin.blockSignals(True)
        self.minimum_spin.setValue(float(low))
        self.maximum_spin.setValue(float(high))
        self.minimum_spin.blockSignals(False)
        self.maximum_spin.blockSignals(False)
        self._sync_manual_sliders()
        self._update_histogram_markers()

    def _auto_levels_toggled(self, enabled):
        self.level_controls_stack.setCurrentWidget(
            self.auto_levels_panel if enabled else self.manual_levels_panel
        )
        self.minimum_spin.setEnabled(not enabled)
        self.maximum_spin.setEnabled(not enabled)
        self.minimum_slider.setEnabled(not enabled)
        self.maximum_slider.setEnabled(not enabled)
        self.low_percent_spin.setEnabled(enabled)
        self.high_percent_spin.setEnabled(enabled)
        self.low_percent_slider.setEnabled(enabled)
        self.high_percent_slider.setEnabled(enabled)
        if enabled:
            self.apply_auto_levels()
        else:
            if self._current_image is not None:
                current_levels = self.levels()
                self._configure_level_controls(self._current_image)
                self._set_level_values(*current_levels)
            self._manual_levels_changed()

    def _manual_levels_changed(self, *_args):
        if self.auto_levels_check.isChecked() or self._current_image is None:
            return
        self.view.getImageItem().setLevels(self.levels())
        self._update_histogram_markers()

    def reset_levels(self):
        self.low_percent_spin.setValue(self._default_low_percentile)
        self.high_percent_spin.setValue(self._default_high_percentile)
        self.gamma_spin.setValue(self._default_gamma)
        self.auto_levels_check.setChecked(True)
        self.apply_auto_levels()

    def _auto_percentiles_changed(self, *_args):
        if self.low_percent_spin.value() >= self.high_percent_spin.value():
            return
        if self.auto_levels_check.isChecked():
            self.apply_auto_levels()

    def apply_auto_levels(self):
        """Apply robust percentile display bounds to the current image."""
        if self._current_image is None or not self.auto_levels_check.isChecked():
            return
        low, high = self._auto_level_values(self._current_image)
        self._set_level_values(low, high)
        image_item = self.view.getImageItem()
        if image_item is not None:
            image_item.setLevels((low, high))
        self._update_histogram_markers()

    def _value_to_slider(self, value):
        span = self._level_high_bound - self._level_low_bound
        if span <= 0:
            return 0
        return round(1000 * (float(value) - self._level_low_bound) / span)

    def _slider_to_value(self, value):
        return self._level_low_bound + (float(value) / 1000.0) * (
            self._level_high_bound - self._level_low_bound
        )

    def _sync_manual_sliders(self):
        self.minimum_slider.blockSignals(True)
        self.maximum_slider.blockSignals(True)
        self.minimum_slider.setValue(self._value_to_slider(self.minimum_spin.value()))
        self.maximum_slider.setValue(self._value_to_slider(self.maximum_spin.value()))
        self.minimum_slider.blockSignals(False)
        self.maximum_slider.blockSignals(False)

    def _minimum_slider_changed(self, value):
        self.minimum_spin.setValue(self._slider_to_value(value))

    def _maximum_slider_changed(self, value):
        self.maximum_spin.setValue(self._slider_to_value(value))

    def _minimum_spin_changed(self, value):
        self.minimum_slider.blockSignals(True)
        self.minimum_slider.setValue(self._value_to_slider(value))
        self.minimum_slider.blockSignals(False)

    def _maximum_spin_changed(self, value):
        self.maximum_slider.blockSignals(True)
        self.maximum_slider.setValue(self._value_to_slider(value))
        self.maximum_slider.blockSignals(False)

    def _low_percent_slider_changed(self, value):
        self.low_percent_spin.setValue(float(value) / 10.0)

    def _high_percent_slider_changed(self, value):
        self.high_percent_spin.setValue(float(value) / 10.0)

    def _low_percent_spin_changed(self, value):
        self.low_percent_slider.blockSignals(True)
        self.low_percent_slider.setValue(round(float(value) * 10.0))
        self.low_percent_slider.blockSignals(False)

    def _high_percent_spin_changed(self, value):
        self.high_percent_slider.blockSignals(True)
        self.high_percent_slider.setValue(round(float(value) * 10.0))
        self.high_percent_slider.blockSignals(False)

    def _gamma_slider_changed(self, value):
        self.gamma_spin.setValue(float(value) / 100.0)

    def _gamma_spin_changed(self, value):
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(round(float(value) * 100.0))
        self.gamma_slider.blockSignals(False)
        self._apply_colormap()

    def _apply_colormap(self, *_args):
        name = self.colormap_combo.currentData()
        if name == "gray":
            lut = np.column_stack([np.arange(256, dtype=np.uint8)] * 3)
        else:
            try:
                lut = pg.colormap.getFromMatplotlib(str(name)).getLookupTable(nPts=256)
            except Exception:
                lut = np.column_stack([np.arange(256, dtype=np.uint8)] * 3)
        self._base_lut = np.asarray(lut, dtype=np.uint8)
        gamma = float(self.gamma_spin.value())
        positions = np.linspace(0.0, 1.0, len(self._base_lut)) ** (1.0 / gamma)
        indices = np.clip(np.rint(positions * (len(self._base_lut) - 1)), 0, len(self._base_lut) - 1)
        self.view.getImageItem().setLookupTable(self._base_lut[indices.astype(int)])

    def _update_histogram(self, array):
        finite = array[np.isfinite(array)].ravel()
        if not finite.size:
            self.histogram_curve.clear()
            return
        if finite.size > 200_000:
            finite = finite[:: max(1, finite.size // 200_000)]
        low, high, normalized = self._histogram_data_range(finite)
        self._histogram_default_range = (low, high)
        self._histogram_normalized = normalized
        histogram, edges = np.histogram(finite, bins=160, range=(low, high))
        centers = (edges[:-1] + edges[1:]) * 0.5
        self.histogram_curve.setData(centers, histogram)
        maximum = float(np.max(histogram)) if histogram.size else 1.0
        self.histogram_plot.setYRange(0.0, max(1.0, maximum) * 1.05, padding=0)
        if not self._histogram_user_view_active:
            self.histogram_plot.setXRange(float(low), float(high), padding=0)
            self._set_histogram_axis_ticks(low, high)
        else:
            visible = self.histogram_plot.getPlotItem().viewRange()[0]
            self._set_histogram_axis_ticks(float(visible[0]), float(visible[1]))
        self._update_histogram_markers()

    def _update_histogram_markers(self):
        low, high = self.levels()
        self.histogram_low_line.setPos(low)
        self.histogram_high_line.setPos(high)

    @staticmethod
    def _histogram_data_range(finite):
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        raw_like = data_min >= 0.0 and data_max > 10.0
        if raw_like:
            if data_max <= 255.0:
                upper = 255.0
            else:
                robust_high = float(np.percentile(finite, 99.9))
                upper = 65535.0 if robust_high <= 65535.0 else robust_high * 1.02
            return 0.0, upper, False
        low, high = np.percentile(finite, (0.1, 99.9))
        low = min(0.0, float(low))
        high = float(high)
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        return low, high + 0.02 * (high - low), True

    def _format_histogram_tick(self, value):
        value = float(value)
        absolute = abs(value)
        if self._histogram_normalized:
            if absolute >= 100:
                return f"{value:.0f}"
            if absolute >= 10:
                return f"{value:.1f}"
            if absolute >= 1:
                return f"{value:.2f}"
            return f"{value:.3f}"
        if absolute >= 1000:
            thousands = value / 1000.0
            if abs(thousands) >= 100:
                return f"{thousands:.0f}k"
            if abs(thousands) >= 10:
                return f"{thousands:.1f}k"
            return f"{thousands:.2f}k"
        return f"{value:.0f}"

    def _set_histogram_axis_ticks(self, x_min, x_max):
        span = float(x_max) - float(x_min)
        if not np.isfinite(span) or span <= 0:
            return
        width = max(120.0, float(self.histogram_plot.width()))
        target_count = int(max(3, min(7, round(width / 58.0))))
        raw_step = span / max(1, target_count)
        exponent = np.floor(np.log10(raw_step))
        base = 10.0**exponent
        fraction = raw_step / base
        nice_fraction = 1.0 if fraction <= 1 else 2.0 if fraction <= 2 else 5.0 if fraction <= 5 else 10.0
        step = nice_fraction * base
        start = np.ceil(float(x_min) / step) * step
        values = []
        value = float(start)
        for _ in range(128):
            if value > float(x_max) + step * 0.25:
                break
            values.append(value)
            value += step
        if not values:
            values = [float(x_min), float(x_max)]
        self._histogram_axis.setTicks(
            [[(value, self._format_histogram_tick(value)) for value in values], []]
        )

    def _histogram_x_range_changed(self, _view_box, x_range):
        try:
            low, high = float(x_range[0]), float(x_range[1])
        except Exception:
            return
        self._set_histogram_axis_ticks(low, high)

    def _refresh_histogram_ticks(self):
        try:
            low, high = self.histogram_plot.getPlotItem().viewRange()[0]
        except Exception:
            return
        self._set_histogram_axis_ticks(float(low), float(high))

    def reset_histogram_range(self):
        self._histogram_user_view_active = False
        low, high = self._histogram_default_range
        self.histogram_plot.setXRange(low, high, padding=0)
        self._set_histogram_axis_ticks(low, high)

    def eventFilter(self, watched, event):
        if watched is self._histogram_viewport:
            event_type = event.type()
            if event_type == QtCore.QEvent.Resize:
                QtCore.QTimer.singleShot(0, self._refresh_histogram_ticks)
                return False
            if event_type == QtCore.QEvent.MouseButtonDblClick:
                self.reset_histogram_range()
                event.accept()
                return True
            if event_type in (QtCore.QEvent.Wheel, QtCore.QEvent.MouseButtonPress):
                self._histogram_user_view_active = True
            elif event_type == QtCore.QEvent.MouseMove:
                try:
                    if event.buttons() != QtCore.Qt.NoButton:
                        self._histogram_user_view_active = True
                except Exception:
                    pass
        return super().eventFilter(watched, event)


class StepList(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stepList")
        self.setFixedWidth(194)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSpacing(4)
        for number, title in enumerate(("Select inputs", "Prepare images", "Normalize", "Review results"), 1):
            item = QtWidgets.QListWidgetItem(f"{number}   {title}")
            item.setSizeHint(QtCore.QSize(160, 45))
            self.addItem(item)
        self.setCurrentRow(0)
