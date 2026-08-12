"""Reusable GUI widgets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from .processing import discover_tiffs


def integrated_profile(image, bounds, *, integration="vertical", statistic="mean"):
    """Return an X or Y profile from a rectangular image region.

    Vertical integration collapses rows and returns an X profile. Horizontal
    integration collapses columns and returns a Y profile.
    """
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Profile input must be a 2D image.")
    x, y, width, height = (int(round(value)) for value in bounds)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(array.shape[1], x + max(1, width))
    y1 = min(array.shape[0], y + max(1, height))
    if x1 <= x0 or y1 <= y0:
        return np.array([], dtype=float), np.array([], dtype=float), "X pixel"
    region = array[y0:y1, x0:x1]
    direction = str(integration).strip().lower()
    reduction = str(statistic).strip().lower()
    if reduction not in {"mean", "sum"}:
        raise ValueError("Profile statistic must be 'mean' or 'sum'.")
    reducer = np.nanmean if reduction == "mean" else np.nansum
    if direction.startswith("h"):
        return (
            np.arange(y0, y1, dtype=float),
            np.asarray(reducer(region, axis=1), dtype=float),
            "Y pixel",
        )
    return (
        np.arange(x0, x1, dtype=float),
        np.asarray(reducer(region, axis=0), dtype=float),
        "X pixel",
    )


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate distance")
        layout = QtWidgets.QFormLayout(self)
        self.distance_spin = QtWidgets.QDoubleSpinBox()
        self.distance_spin.setDecimals(6)
        self.distance_spin.setRange(1e-9, 1e12)
        self.distance_spin.setValue(1.0)
        self.distance_spin.selectAll()
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItem("µm", 1.0)
        self.unit_combo.addItem("mm", 1000.0)
        self.unit_combo.addItem("cm", 10_000.0)
        unit_row = QtWidgets.QWidget()
        unit_layout = QtWidgets.QHBoxLayout(unit_row)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.addWidget(self.distance_spin, 1)
        unit_layout.addWidget(self.unit_combo)
        layout.addRow("Known line length", unit_row)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def distance_um(self):
        return float(self.distance_spin.value()) * float(self.unit_combo.currentData())


class FileSelectionCard(QtWidgets.QGroupBox):
    filesChanged = QtCore.Signal()
    previewRequested = QtCore.Signal(str)
    directoryChanged = QtCore.Signal(str)

    def __init__(self, title: str, description: str, parent=None):
        super().__init__(title, parent)
        self._paths: list[str] = []
        self._last_directory = str(Path.home())
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
        if self._paths:
            self._remember_directory(Path(self._paths[-1]).parent)
        self._refresh(select_first=True)

    def set_last_directory(self, directory) -> None:
        self._last_directory = str(Path(directory).expanduser().resolve())

    def _remember_directory(self, directory) -> None:
        self.set_last_directory(directory)
        self.directoryChanged.emit(self._last_directory)

    def clear(self) -> None:
        self._paths = []
        self._refresh()

    def _choose_files(self) -> None:
        values, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select TIFF images",
            self._last_directory,
            "TIFF images (*.tif *.tiff)",
        )
        if values:
            self._remember_directory(Path(values[0]).parent)
            self.add_paths(values)

    def _choose_folder(self) -> None:
        value = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select image folder", self._last_directory
        )
        if value:
            self._remember_directory(value)
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
        self._measure_drag_active = False
        self._measure_start_point = None
        self._measure_points = None
        self._profile_roi_initialized = False
        self._profile_popup = None
        self._profile_plot = None
        self._profile_curve = None
        self._profile_coordinates = np.array([], dtype=float)
        self._profile_values = np.array([], dtype=float)
        self._profile_update_timer = QtCore.QTimer(self)
        self._profile_update_timer.setSingleShot(True)
        self._profile_update_timer.setInterval(40)
        self._profile_update_timer.timeout.connect(self._update_profile_plot)
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
        self._measure_scene = scene
        if scene is not None and hasattr(scene, "sigMouseClicked"):
            scene.sigMouseClicked.connect(self._scene_mouse_clicked)
        layout.addWidget(self.view, 1)
        self.roi = pg.RectROI((10, 10), (100, 100), pen=pg.mkPen("#f0ad4e", width=2))
        self.roi.addScaleHandle((1, 1), (0, 0))
        self.roi.addScaleHandle((0, 0), (1, 1))
        self.view.getView().addItem(self.roi)
        self.roi.hide()
        self.roi.sigRegionChangeFinished.connect(lambda: self.roiChanged.emit(self.roi_bounds()))

        self.measure_line = pg.PlotCurveItem(pen=pg.mkPen("#ff9f1c", width=2))
        self.view.getView().addItem(self.measure_line, ignoreBounds=True)
        self.measure_line.hide()

        self.profile_roi = pg.RectROI(
            (10, 10),
            (100, 100),
            pen=pg.mkPen("#00b4d8", width=2),
            movable=True,
            rotatable=False,
            removable=False,
            resizable=True,
        )
        self.profile_roi.addScaleHandle((1, 1), (0, 0))
        self.profile_roi.addScaleHandle((0, 0), (1, 1))
        self.view.getView().addItem(self.profile_roi, ignoreBounds=True)
        self.profile_roi.hide()
        self.profile_roi.sigRegionChanged.connect(self._schedule_profile_update)
        self.profile_roi.sigRegionChangeFinished.connect(self._update_profile_plot)

        analysis_panel = self._build_analysis_panel()
        layout.insertWidget(0, analysis_panel)
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

    def _build_analysis_panel(self):
        panel = QtWidgets.QGroupBox("Analysis")
        row = QtWidgets.QHBoxLayout(panel)
        row.setContentsMargins(8, 3, 8, 3)
        row.setSpacing(7)

        self.measure_check = QtWidgets.QCheckBox("Line")
        self.measure_check.setToolTip("Drag across the image to draw a measurement line.")
        self.measure_readout = QtWidgets.QLabel("-- px")
        self.measure_readout.setFixedWidth(130)
        self.pixel_size_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_size_spin.setDecimals(6)
        self.pixel_size_spin.setRange(0.0, 1e9)
        self.pixel_size_spin.setSingleStep(1.0)
        self.pixel_size_spin.setSpecialValueText("Not set")
        self.pixel_size_spin.setSuffix(" µm/px")
        self.pixel_size_spin.setFixedWidth(108)
        self.pixel_size_spin.setToolTip(
            "Physical pixel size. Enter it directly or calibrate it from the measurement line."
        )
        self.calibrate_button = QtWidgets.QPushButton("Calibrate")
        self.calibrate_button.setFixedWidth(94)
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.setToolTip(
            "Set the physical pixel size from the line and a known physical distance."
        )

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.profile_check = QtWidgets.QCheckBox("Profile")
        self.profile_check.setToolTip("Show a blue integration ROI and its live profile plot.")
        self.profile_orientation_combo = QtWidgets.QComboBox()
        self.profile_orientation_combo.addItem("Vertical → X", "vertical")
        self.profile_orientation_combo.addItem("Horizontal → Y", "horizontal")
        self.profile_orientation_combo.setFixedWidth(145)
        self.profile_orientation_combo.setToolTip(
            "Vertical integration collapses rows; horizontal integration collapses columns."
        )
        self.profile_stat_combo = QtWidgets.QComboBox()
        self.profile_stat_combo.addItem("Mean", "mean")
        self.profile_stat_combo.addItem("Sum", "sum")
        self.profile_stat_combo.setFixedWidth(68)
        self.profile_stat_combo.setToolTip("Choose averaging or summed integration.")

        row.addWidget(self.measure_check)
        row.addWidget(self.measure_readout)
        row.addWidget(QtWidgets.QLabel("Scale"))
        row.addWidget(self.pixel_size_spin)
        row.addWidget(self.calibrate_button)
        row.addWidget(separator)
        row.addWidget(self.profile_check)
        row.addWidget(self.profile_orientation_combo)
        row.addWidget(self.profile_stat_combo)
        row.addStretch(1)
        panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.measure_check.toggled.connect(self._measurement_toggled)
        self.pixel_size_spin.valueChanged.connect(self._calibration_changed)
        self.calibrate_button.clicked.connect(self._show_calibration_dialog)
        self.profile_check.toggled.connect(self._profile_toggled)
        self.profile_orientation_combo.currentIndexChanged.connect(self._update_profile_plot)
        self.profile_stat_combo.currentIndexChanged.connect(self._update_profile_plot)
        return panel

    def _initialize_analysis_overlays(self, *, force=False):
        if self._current_image is None:
            return
        height, width = self._current_image.shape
        if force or not self._profile_roi_initialized:
            roi_width = max(2.0, width * 0.5)
            roi_height = max(2.0, height * 0.25)
            previous = self.profile_roi.blockSignals(True)
            self.profile_roi.setPos(
                ((width - roi_width) * 0.5, (height - roi_height) * 0.5),
                update=False,
            )
            self.profile_roi.setSize((roi_width, roi_height), update=True)
            self.profile_roi.blockSignals(previous)
            self._profile_roi_initialized = True
        self._update_measurement()
        if self.profile_check.isChecked():
            self._update_profile_plot()

    def _measurement_points(self):
        return [] if self._measure_points is None else list(self._measure_points)

    def measurement_length_px(self):
        points = self._measurement_points()
        if len(points) != 2:
            return None
        length = float(
            np.hypot(points[1].x() - points[0].x(), points[1].y() - points[0].y())
        )
        return length if np.isfinite(length) and length > 0 else None

    def _measurement_toggled(self, enabled):
        self._set_measure_interaction_enabled(enabled)
        if not enabled:
            self._clear_measurement()
        self.calibrate_button.setEnabled(bool(enabled and self.measurement_length_px()))
        self._update_measurement()

    def _set_measure_interaction_enabled(self, enabled):
        view_box = self.view.getView()
        view_box.setMouseEnabled(x=not enabled, y=not enabled)
        if self._measure_scene is None:
            return
        self._measure_scene.removeEventFilter(self)
        if enabled:
            self._measure_scene.installEventFilter(self)

    def prepare_close(self):
        """Detach scene hooks before Qt tears down the graphics scene."""
        self._measure_drag_active = False
        self._measure_start_point = None
        if self._measure_scene is not None:
            self._measure_scene.removeEventFilter(self)

    def _clear_measurement(self):
        self._measure_drag_active = False
        self._measure_start_point = None
        self._measure_points = None
        self.measure_line.setData([], [])
        self.measure_line.hide()

    def _map_scene_to_image_point(self, scene_position):
        try:
            view_box = self.view.getView().getViewBox()
            return QtCore.QPointF(view_box.mapSceneToView(scene_position))
        except Exception:
            return None

    def _update_measure_line(self, start_point, end_point):
        if start_point is None or end_point is None:
            return
        start = QtCore.QPointF(start_point)
        end = QtCore.QPointF(end_point)
        self._measure_points = (start, end)
        self.measure_line.setData([start.x(), end.x()], [start.y(), end.y()])
        self.measure_line.show()
        self._update_measurement()

    def _update_measurement(self, *_args):
        length_px = self.measurement_length_px()
        if length_px is None or self._current_image is None:
            self.measure_readout.setText("-- px")
            self.measure_readout.setToolTip("Enable Line, then drag across the image to measure.")
            self.calibrate_button.setEnabled(False)
            return
        pixel_size_um = float(self.pixel_size_spin.value())
        text = f"{length_px:.2f} px"
        tooltip = text
        if pixel_size_um > 0:
            length_um = length_px * pixel_size_um
            physical = f"{length_um:.3g} µm" if length_um < 1000 else f"{length_um / 1000:.4g} mm"
            text = f"{length_px:.2f} px / {physical}"
            tooltip = (
                f"{length_px:.6g} pixels × {pixel_size_um:.6g} µm/pixel = "
                f"{length_um:.6g} µm"
            )
        self.measure_readout.setText(text)
        self.measure_readout.setToolTip(tooltip)
        self.calibrate_button.setEnabled(self.measure_check.isChecked())

    def set_calibration_from_line(self, distance_um):
        length_px = self.measurement_length_px()
        distance_um = float(distance_um)
        if length_px is None or not np.isfinite(distance_um) or distance_um <= 0:
            raise ValueError("Calibration requires a valid line and positive distance.")
        self.pixel_size_spin.setValue(distance_um / length_px)
        return float(self.pixel_size_spin.value())

    def _show_calibration_dialog(self):
        if self.measurement_length_px() is None:
            return
        dialog = CalibrationDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.set_calibration_from_line(dialog.distance_um())

    def _calibration_changed(self, *_args):
        self._update_measurement()
        if self.profile_check.isChecked():
            self._update_profile_plot()

    def profile_bounds(self):
        if self._current_image is None:
            return (0, 0, 0, 0)
        height, width = self._current_image.shape
        pos = self.profile_roi.pos()
        size = self.profile_roi.size()
        x0 = max(0, min(width, int(np.floor(min(pos.x(), pos.x() + size.x())))))
        x1 = max(0, min(width, int(np.ceil(max(pos.x(), pos.x() + size.x())))))
        y0 = max(0, min(height, int(np.floor(min(pos.y(), pos.y() + size.y())))))
        y1 = max(0, min(height, int(np.ceil(max(pos.y(), pos.y() + size.y())))))
        return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def _ensure_profile_popup(self):
        if self._profile_popup is not None:
            return
        popup = QtWidgets.QWidget(self, QtCore.Qt.Tool)
        popup.setWindowTitle("Image Profile")
        popup.resize(600, 360)
        popup.installEventFilter(self)
        popup_layout = QtWidgets.QVBoxLayout(popup)
        self._profile_plot = pg.PlotWidget()
        self._profile_plot.showGrid(x=True, y=True, alpha=0.2)
        self._profile_curve = self._profile_plot.plot(pen=pg.mkPen("#0077b6", width=2))
        popup_layout.addWidget(self._profile_plot, 1)
        self.profile_roi_label = QtWidgets.QLabel("ROI: --")
        popup_layout.addWidget(self.profile_roi_label)
        self._profile_popup = popup

    def _profile_toggled(self, enabled):
        self._initialize_analysis_overlays()
        self.profile_roi.setVisible(bool(enabled and self._current_image is not None))
        if enabled:
            self._ensure_profile_popup()
            self._profile_popup.show()
            self._profile_popup.raise_()
            self._update_profile_plot()
        elif self._profile_popup is not None:
            self._profile_popup.hide()

    def _schedule_profile_update(self, *_args):
        if self.profile_check.isChecked():
            self._profile_update_timer.start()

    def _update_profile_plot(self, *_args):
        if not self.profile_check.isChecked() or self._current_image is None:
            return
        self._ensure_profile_popup()
        bounds = self.profile_bounds()
        coordinates, values, axis_label = integrated_profile(
            self._current_image,
            bounds,
            integration=self.profile_orientation_combo.currentData(),
            statistic=self.profile_stat_combo.currentData(),
        )
        pixel_size_um = float(self.pixel_size_spin.value())
        if pixel_size_um > 0 and coordinates.size:
            coordinates = coordinates * pixel_size_um
            axis_label = f"{axis_label[0]} distance (µm)"
        self._profile_coordinates = coordinates
        self._profile_values = values
        self._profile_curve.setData(coordinates, values)
        plot_item = self._profile_plot.getPlotItem()
        statistic = str(self.profile_stat_combo.currentData())
        plot_item.setLabel("bottom", axis_label)
        plot_item.setLabel("left", "Mean intensity" if statistic == "mean" else "Integrated intensity")
        plot_item.enableAutoRange(axis="xy", enable=True)
        x, y, width, height = bounds
        self.profile_roi_label.setText(
            f"ROI: x={x}:{x + width}, y={y}:{y + height} · {width} × {height} px"
        )

    def set_image(self, image: np.ndarray, caption: str = "") -> None:
        array = np.asarray(image, dtype=np.float32)
        previous_shape = None if self._current_image is None else self._current_image.shape
        self._current_image = array
        self._configure_level_controls(array)
        levels = self._auto_level_values(array) if self.auto_levels_check.isChecked() else self.levels()
        self.view.setImage(array.T, autoLevels=False, levels=levels)
        self._apply_colormap()
        self._update_histogram(array)
        self.caption.setText(caption or f"{array.shape[1]} × {array.shape[0]}")
        self._initialize_analysis_overlays(force=previous_shape != array.shape)
        self.measure_line.setVisible(
            self.measure_check.isChecked() and self.measurement_length_px() is not None
        )
        self.profile_roi.setVisible(self.profile_check.isChecked())
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
        if watched is self._measure_scene and self.measure_check.isChecked():
            event_type = event.type()
            if event_type == QtCore.QEvent.GraphicsSceneMousePress:
                try:
                    if event.button() != QtCore.Qt.LeftButton or self._current_image is None:
                        return False
                    point = self._map_scene_to_image_point(event.scenePos())
                    if point is None:
                        return False
                    self._measure_drag_active = True
                    self._measure_start_point = point
                    self._update_measure_line(point, point)
                    event.accept()
                    return True
                except Exception:
                    return False
            if event_type == QtCore.QEvent.GraphicsSceneMouseMove and self._measure_drag_active:
                point = self._map_scene_to_image_point(event.scenePos())
                if point is not None and self._measure_start_point is not None:
                    self._update_measure_line(self._measure_start_point, point)
                event.accept()
                return True
            if event_type == QtCore.QEvent.GraphicsSceneMouseRelease and self._measure_drag_active:
                try:
                    if event.button() != QtCore.Qt.LeftButton:
                        return False
                    point = self._map_scene_to_image_point(event.scenePos())
                    if point is not None and self._measure_start_point is not None:
                        self._update_measure_line(self._measure_start_point, point)
                    self._measure_drag_active = False
                    self._measure_start_point = None
                    event.accept()
                    return True
                except Exception:
                    self._measure_drag_active = False
                    self._measure_start_point = None
                    return True
        if watched is self._profile_popup and event.type() == QtCore.QEvent.Close:
            if self.profile_check.isChecked():
                self.profile_check.setChecked(False)
            return False
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
    def __init__(self, parent=None, titles=None):
        super().__init__(parent)
        self.setObjectName("stepList")
        self.setFixedWidth(194)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSpacing(4)
        titles = titles or ("Select inputs", "Prepare images", "Normalize", "Review results")
        for number, title in enumerate(titles, 1):
            item = QtWidgets.QListWidgetItem(f"{number}   {title}")
            item.setToolTip(str(title))
            item.setSizeHint(QtCore.QSize(160, 45))
            self.addItem(item)
        self.setCurrentRow(0)
