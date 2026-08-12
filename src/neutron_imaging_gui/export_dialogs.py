"""Qt option dialogs for current-image and batch exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qtpy import QtCore, QtWidgets

from .exporting import crop_image, populate_raster_preview, populate_styled_figure


@dataclass(frozen=True)
class CurrentExportOptions:
    path: Path
    format: str
    crop_bounds: tuple[int, int, int, int] | None
    cmap: str
    use_viewer_levels: bool
    manual_levels: tuple[float, float]
    colorbar: bool
    title: str
    dpi: int
    embed_tiff_metadata: bool
    companion_json: bool


@dataclass(frozen=True)
class BatchExportOptions:
    directory: Path
    categories: tuple[str, ...]
    use_subfolders: bool
    embed_tiff_metadata: bool
    companion_json: bool
    overwrite: bool


class CurrentImageExportDialog(QtWidgets.QDialog):
    directorySelected = QtCore.Signal(str)

    def __init__(
        self,
        *,
        image_shape=None,
        image=None,
        image_name,
        initial_directory,
        colormap="gray",
        viewer_levels=None,
        profile_bounds=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export current image")
        self.resize(1120, 690)
        self.setMinimumSize(980, 590)
        if image is None:
            if image_shape is None:
                raise ValueError("Current-image export requires an image or image shape.")
            image = np.zeros(tuple(int(value) for value in image_shape), dtype=np.float32)
        self._image = np.asarray(image, dtype=np.float32)
        if self._image.ndim != 2:
            raise ValueError("Current-image export preview requires a 2D image.")
        self._image_shape = self._image.shape
        self._viewer_levels = viewer_levels
        self._profile_bounds = profile_bounds
        root = QtWidgets.QVBoxLayout(self)
        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, 1)
        controls_widget = QtWidgets.QWidget()
        controls_widget.setMaximumWidth(540)
        controls = QtWidgets.QVBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        body.addWidget(controls_widget)

        destination_group = QtWidgets.QGroupBox("Destination and format")
        destination_form = QtWidgets.QFormLayout(destination_group)
        destination_row = QtWidgets.QWidget()
        destination_layout = QtWidgets.QHBoxLayout(destination_row)
        destination_layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QtWidgets.QLineEdit(
            str(Path(initial_directory) / f"{image_name}.tif")
        )
        self.browse_button = QtWidgets.QPushButton("Browse…")
        destination_layout.addWidget(self.path_edit, 1)
        destination_layout.addWidget(self.browse_button)
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItem("TIFF · full-resolution float data", "tiff")
        self.format_combo.addItem("PNG · image only", "png_image")
        self.format_combo.addItem("PNG · styled figure", "png_styled")
        destination_form.addRow("Output", destination_row)
        destination_form.addRow("File type", self.format_combo)
        controls.addWidget(destination_group)

        self.crop_group = QtWidgets.QGroupBox("Crop image")
        crop_form = QtWidgets.QFormLayout(self.crop_group)
        self.crop_check = QtWidgets.QCheckBox("Enable crop")
        self.crop_check.setChecked(False)
        crop_form.addRow(self.crop_check)
        crop_row = QtWidgets.QWidget()
        crop_layout = QtWidgets.QHBoxLayout(crop_row)
        crop_layout.setContentsMargins(0, 0, 0, 0)
        height, width = self._image_shape
        self.crop_x_spin = self._integer_spin(0, max(0, width - 1), 0)
        self.crop_y_spin = self._integer_spin(0, max(0, height - 1), 0)
        self.crop_width_spin = self._integer_spin(1, max(1, width), max(1, width))
        self.crop_height_spin = self._integer_spin(1, max(1, height), max(1, height))
        for label, widget in (
            ("X", self.crop_x_spin),
            ("Y", self.crop_y_spin),
            ("Width", self.crop_width_spin),
            ("Height", self.crop_height_spin),
        ):
            crop_layout.addWidget(QtWidgets.QLabel(label))
            crop_layout.addWidget(widget)
        self.use_profile_roi_button = QtWidgets.QPushButton("Use profile ROI")
        self.use_profile_roi_button.setEnabled(bool(profile_bounds and profile_bounds[2] > 0))
        crop_form.addRow(crop_row)
        crop_form.addRow(self.use_profile_roi_button)
        controls.addWidget(self.crop_group)

        appearance_group = QtWidgets.QGroupBox("Figure appearance and preview")
        appearance_form = QtWidgets.QFormLayout(appearance_group)
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
        cmap_index = self.colormap_combo.findData(colormap)
        if cmap_index >= 0:
            self.colormap_combo.setCurrentIndex(cmap_index)
        self.viewer_levels_check = QtWidgets.QCheckBox("Use current viewer min/max")
        self.viewer_levels_check.setChecked(True)
        finite = self._image[np.isfinite(self._image)]
        if viewer_levels is not None:
            initial_low, initial_high = (float(value) for value in viewer_levels)
        elif finite.size:
            initial_low, initial_high = (
                float(value) for value in np.percentile(finite, (1.0, 99.7))
            )
        else:
            initial_low, initial_high = 0.0, 1.0
        if not np.isfinite(initial_low):
            initial_low = 0.0
        if not np.isfinite(initial_high) or initial_high <= initial_low:
            initial_high = initial_low + 1.0
        manual_levels_row = QtWidgets.QWidget()
        manual_levels_layout = QtWidgets.QHBoxLayout(manual_levels_row)
        manual_levels_layout.setContentsMargins(0, 0, 0, 0)
        self.minimum_spin = self._level_spin(initial_low)
        self.maximum_spin = self._level_spin(initial_high)
        manual_levels_layout.addWidget(QtWidgets.QLabel("Min"))
        manual_levels_layout.addWidget(self.minimum_spin)
        manual_levels_layout.addWidget(QtWidgets.QLabel("Max"))
        manual_levels_layout.addWidget(self.maximum_spin)
        self.colorbar_check = QtWidgets.QCheckBox("Include colorbar")
        self.colorbar_check.setChecked(True)
        self.title_edit = QtWidgets.QLineEdit(str(image_name))
        self.dpi_spin = self._integer_spin(72, 600, 150)
        appearance_form.addRow("Color map", self.colormap_combo)
        appearance_form.addRow(self.viewer_levels_check)
        appearance_form.addRow("Manual range", manual_levels_row)
        appearance_form.addRow(self.colorbar_check)
        appearance_form.addRow("Figure title", self.title_edit)
        appearance_form.addRow("DPI", self.dpi_spin)
        controls.addWidget(appearance_group)

        metadata_group = QtWidgets.QGroupBox("Metadata")
        metadata_layout = QtWidgets.QVBoxLayout(metadata_group)
        self.embed_metadata_check = QtWidgets.QCheckBox("Embed provenance in TIFF tags")
        self.embed_metadata_check.setChecked(True)
        self.companion_json_check = QtWidgets.QCheckBox("Write companion JSON")
        self.companion_json_check.setChecked(True)
        metadata_layout.addWidget(self.embed_metadata_check)
        metadata_layout.addWidget(self.companion_json_check)
        controls.addWidget(metadata_group)
        controls.addStretch(1)

        preview_group = QtWidgets.QGroupBox("Output preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.preview_figure = Figure(figsize=(6.0, 5.2), dpi=100)
        self.preview_canvas = FigureCanvasQTAgg(self.preview_figure)
        self.preview_canvas.setMinimumSize(430, 420)
        preview_layout.addWidget(self.preview_canvas, 1)
        self.preview_note = QtWidgets.QLabel()
        self.preview_note.setWordWrap(True)
        self.preview_note.setProperty("muted", True)
        preview_layout.addWidget(self.preview_note)
        body.addWidget(preview_group, 1)

        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(75)
        self._preview_timer.timeout.connect(self._refresh_preview)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.browse_button.clicked.connect(self._browse)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        self.crop_x_spin.valueChanged.connect(self._update_crop_limits)
        self.crop_y_spin.valueChanged.connect(self._update_crop_limits)
        self.use_profile_roi_button.clicked.connect(self._use_profile_roi)
        self.crop_check.toggled.connect(self._set_crop_controls_enabled)
        self.crop_check.toggled.connect(self._schedule_preview)
        for spin in (
            self.crop_x_spin,
            self.crop_y_spin,
            self.crop_width_spin,
            self.crop_height_spin,
        ):
            spin.valueChanged.connect(self._schedule_preview)
        self.colormap_combo.currentIndexChanged.connect(self._schedule_preview)
        self.viewer_levels_check.toggled.connect(self._schedule_preview)
        self.viewer_levels_check.toggled.connect(self._set_level_controls_enabled)
        self.minimum_spin.valueChanged.connect(self._schedule_preview)
        self.maximum_spin.valueChanged.connect(self._schedule_preview)
        self.colorbar_check.toggled.connect(self._schedule_preview)
        self.title_edit.textChanged.connect(self._schedule_preview)
        self._set_crop_controls_enabled(False)
        self._set_level_controls_enabled(True)
        self._format_changed()
        self._refresh_preview()

    @staticmethod
    def _integer_spin(minimum, maximum, value):
        spin = QtWidgets.QSpinBox()
        spin.setRange(int(minimum), int(maximum))
        spin.setValue(int(value))
        return spin

    @staticmethod
    def _level_spin(value):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1.0e15, 1.0e15)
        spin.setDecimals(6)
        spin.setValue(float(value))
        spin.setKeyboardTracking(False)
        return spin

    def _format_changed(self, *_args):
        kind = str(self.format_combo.currentData())
        suffix = ".tif" if kind == "tiff" else ".png"
        path = Path(self.path_edit.text().strip())
        if path.suffix.lower() in {".tif", ".tiff", ".png"}:
            self.path_edit.setText(str(path.with_suffix(suffix)))
        styled = kind == "png_styled"
        image_png = kind == "png_image"
        self.colormap_combo.setEnabled(image_png or styled)
        self.viewer_levels_check.setEnabled(True)
        self.colorbar_check.setEnabled(styled)
        self.title_edit.setEnabled(styled)
        self.dpi_spin.setEnabled(styled)
        self.embed_metadata_check.setEnabled(kind == "tiff")
        if styled:
            note = "Styled PNG preview includes the exported axes, title, and optional colorbar."
        elif image_png:
            note = "Image-only PNG preview matches the selected color map and levels, without figure decoration."
        else:
            note = (
                "TIFF preview is grayscale without decoration and uses the selected min/max "
                "for visibility. The exported TIFF retains the original full-resolution "
                "numeric data."
            )
        self.preview_note.setText(note)
        self._schedule_preview()

    def _browse(self):
        kind = str(self.format_combo.currentData())
        if kind == "tiff":
            file_filter = "TIFF image (*.tif *.tiff)"
        else:
            file_filter = "PNG image (*.png)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export current image",
            self.path_edit.text(),
            file_filter,
        )
        if path:
            self.path_edit.setText(path)
            self.directorySelected.emit(str(Path(path).expanduser().resolve().parent))

    def _update_crop_limits(self, *_args):
        height, width = self._image_shape
        self.crop_width_spin.setMaximum(max(1, width - self.crop_x_spin.value()))
        self.crop_height_spin.setMaximum(max(1, height - self.crop_y_spin.value()))

    def _set_crop_controls_enabled(self, enabled):
        for widget in (
            self.crop_x_spin,
            self.crop_y_spin,
            self.crop_width_spin,
            self.crop_height_spin,
        ):
            widget.setEnabled(bool(enabled))

    def _set_level_controls_enabled(self, use_viewer_levels):
        manual = not bool(use_viewer_levels)
        self.minimum_spin.setEnabled(manual)
        self.maximum_spin.setEnabled(manual)

    def _selected_levels(self):
        if self.viewer_levels_check.isChecked() and self._viewer_levels is not None:
            return tuple(float(value) for value in self._viewer_levels)
        return self._manual_levels()

    def _manual_levels(self):
        low = float(self.minimum_spin.value())
        high = float(self.maximum_spin.value())
        if high <= low:
            high = low + max(1.0e-6, abs(low) * 1.0e-9)
        return low, high

    def _schedule_preview(self, *_args):
        self._preview_timer.start()

    def _preview_crop_bounds(self):
        if not self.crop_check.isChecked():
            return None
        return (
            self.crop_x_spin.value(),
            self.crop_y_spin.value(),
            self.crop_width_spin.value(),
            self.crop_height_spin.value(),
        )

    def _refresh_preview(self):
        cropped = crop_image(self._image, self._preview_crop_bounds())
        height, width = cropped.shape
        stride = max(1, int(np.ceil(max(height, width) / 1200.0)))
        displayed = cropped[::stride, ::stride]
        extent = (0, width, height, 0) if stride > 1 else None
        kind = str(self.format_combo.currentData())
        levels = self._selected_levels()
        if kind == "tiff":
            populate_raster_preview(
                self.preview_figure,
                displayed,
                cmap="gray",
                levels=levels,
                extent=extent,
            )
        elif kind == "png_image":
            populate_raster_preview(
                self.preview_figure,
                displayed,
                cmap=str(self.colormap_combo.currentData()),
                levels=levels,
                extent=extent,
            )
        else:
            populate_styled_figure(
                self.preview_figure,
                displayed,
                cmap=str(self.colormap_combo.currentData()),
                levels=levels,
                colorbar=self.colorbar_check.isChecked(),
                title=self.title_edit.text().strip(),
                extent=extent,
            )
        self.preview_canvas.draw_idle()

    def _use_profile_roi(self):
        if not self._profile_bounds:
            return
        x, y, width, height = self._profile_bounds
        self.crop_x_spin.setValue(int(x))
        self.crop_y_spin.setValue(int(y))
        self.crop_width_spin.setValue(max(1, int(width)))
        self.crop_height_spin.setValue(max(1, int(height)))
        self.crop_check.setChecked(True)

    def options(self):
        kind = str(self.format_combo.currentData())
        path = Path(self.path_edit.text().strip()).expanduser()
        expected_suffix = ".tif" if kind == "tiff" else ".png"
        if path.suffix.lower() not in {".tif", ".tiff", ".png"}:
            path = path.with_suffix(expected_suffix)
        crop = None
        if self.crop_check.isChecked():
            crop = (
                self.crop_x_spin.value(),
                self.crop_y_spin.value(),
                self.crop_width_spin.value(),
                self.crop_height_spin.value(),
            )
        return CurrentExportOptions(
            path=path.resolve(),
            format=kind,
            crop_bounds=crop,
            cmap=str(self.colormap_combo.currentData()),
            use_viewer_levels=self.viewer_levels_check.isChecked(),
            manual_levels=self._manual_levels(),
            colorbar=self.colorbar_check.isChecked(),
            title=self.title_edit.text().strip(),
            dpi=self.dpi_spin.value(),
            embed_tiff_metadata=self.embed_metadata_check.isChecked(),
            companion_json=self.companion_json_check.isChecked(),
        )


class BatchExportDialog(QtWidgets.QDialog):
    directorySelected = QtCore.Signal(str)

    def __init__(self, *, initial_directory, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export all reduction results")
        self.resize(560, 420)
        root = QtWidgets.QVBoxLayout(self)

        destination_group = QtWidgets.QGroupBox("Destination")
        destination_layout = QtWidgets.QHBoxLayout(destination_group)
        self.directory_edit = QtWidgets.QLineEdit(
            str(Path(initial_directory) / "neutron_imaging_export")
        )
        self.browse_button = QtWidgets.QPushButton("Browse…")
        destination_layout.addWidget(self.directory_edit, 1)
        destination_layout.addWidget(self.browse_button)
        root.addWidget(destination_group)

        products_group = QtWidgets.QGroupBox("Products")
        products_layout = QtWidgets.QVBoxLayout(products_group)
        self.category_checks = {}
        for key, label in (
            ("background", "Background references · white and dark"),
            ("combined", "Combined sample images"),
            ("transmission", "Normalized transmission"),
            ("attenuation", "Attenuation · −log(transmission)"),
        ):
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(True)
            self.category_checks[key] = checkbox
            products_layout.addWidget(checkbox)
        root.addWidget(products_group)

        organization_group = QtWidgets.QGroupBox("Organization and metadata")
        organization_layout = QtWidgets.QVBoxLayout(organization_group)
        self.subfolders_check = QtWidgets.QCheckBox(
            "Create background, combined, transmission, and attenuation subfolders"
        )
        self.subfolders_check.setChecked(True)
        self.embed_metadata_check = QtWidgets.QCheckBox("Embed provenance in every TIFF")
        self.embed_metadata_check.setChecked(True)
        self.companion_json_check = QtWidgets.QCheckBox("Write metadata.json manifest")
        self.companion_json_check.setChecked(True)
        self.overwrite_check = QtWidgets.QCheckBox("Overwrite existing files")
        self.overwrite_check.setChecked(False)
        organization_layout.addWidget(self.subfolders_check)
        organization_layout.addWidget(self.embed_metadata_check)
        organization_layout.addWidget(self.companion_json_check)
        organization_layout.addWidget(self.overwrite_check)
        root.addWidget(organization_group)

        note = QtWidgets.QLabel(
            "Exports are float32 TIFFs. Metadata includes source filenames and tags, "
            "known exposures, total known exposure time, processing settings and time, "
            "and software versions."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.browse_button.clicked.connect(self._browse)

    def _browse(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose export parent directory",
            str(Path(self.directory_edit.text()).expanduser().parent),
        )
        if directory:
            self.directory_edit.setText(str(Path(directory) / "neutron_imaging_export"))
            self.directorySelected.emit(str(Path(directory).expanduser().resolve()))

    def options(self):
        categories = tuple(
            key for key, checkbox in self.category_checks.items() if checkbox.isChecked()
        )
        return BatchExportOptions(
            directory=Path(self.directory_edit.text().strip()).expanduser().resolve(),
            categories=categories,
            use_subfolders=self.subfolders_check.isChecked(),
            embed_tiff_metadata=self.embed_metadata_check.isChecked(),
            companion_json=self.companion_json_check.isChecked(),
            overwrite=self.overwrite_check.isChecked(),
        )
