"""Interactive tomography preparation and reconstruction workspace."""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from .tomography import (
    ReconstructionTrial,
    TomographyInput,
    backend_capabilities,
    create_leap_geometry,
    default_recipe,
    inspect_dataset,
    leap_tilt_comparison,
    load_preview_stack,
    load_recipe,
    prepare_tomography,
    reconstruction_preflight,
    run_full_reconstruction,
    run_leap_trial,
    run_tomopy_trial,
    save_recipe,
    trial_summary,
)
from .tomography_workers import TomographyJobRunner
from .widgets import FileSelectionCard, ImagePreview, StepList


log = logging.getLogger(__name__)


class TomographyWorkspace(QtWidgets.QWidget):
    busyChanged = QtCore.Signal(bool)
    directoryChanged = QtCore.Signal(str)

    def __init__(self, *, initial_directory=None, parent=None):
        super().__init__(parent)
        self._last_directory = Path(initial_directory or Path.home()).expanduser().resolve()
        self._manifest = None
        self._preview_stack = None
        self._prepared = None
        self._geometry = None
        self._tilt_previews = None
        self._trials: list[ReconstructionTrial] = []
        self._selected_trial_index = None
        self._recipe = default_recipe()
        self._pending_crop = None
        self._pending_dose_roi = None
        self._external_busy = False
        self._runner = TomographyJobRunner(self)
        self._full_progress_timer = QtCore.QTimer(self)
        self._full_progress_timer.setInterval(750)
        self._full_progress_timer.timeout.connect(self._poll_full_progress)
        self._capabilities = backend_capabilities()
        self._build_ui()
        self._connect_signals()
        self._update_capabilities()
        self._update_actions()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QFrame(objectName="header")
        header_layout = QtWidgets.QHBoxLayout(header)
        title_column = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Tomography preparation and reconstruction", objectName="title")
        subtitle = QtWidgets.QLabel(
            "Prepare projections with NIT, compare reconstruction trials, and export a cluster recipe.",
            objectName="subtitle",
        )
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header_layout.addLayout(title_column, 1)
        self.import_recipe_button = QtWidgets.QPushButton("Import recipe…")
        self.export_recipe_button = QtWidgets.QPushButton("Export recipe…")
        header_layout.addWidget(self.import_recipe_button)
        header_layout.addWidget(self.export_recipe_button)
        root.addWidget(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        workflow = QtWidgets.QWidget()
        workflow_layout = QtWidgets.QHBoxLayout(workflow)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        self.steps = StepList(
            titles=("Dataset", "Load and crop", "Prepare", "Geometry and trials", "Export and full recon")
        )
        self.steps.setFixedWidth(175)
        self.pages = QtWidgets.QStackedWidget()
        workflow_layout.addWidget(self.steps)
        workflow_layout.addWidget(self.pages, 1)
        splitter.addWidget(workflow)
        workflow.setMinimumWidth(590)
        workflow.setMaximumWidth(600)

        preview_panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.addWidget(QtWidgets.QLabel("Tomography preview"))
        self.preview_combo = QtWidgets.QComboBox()
        self.preview_combo.setMinimumWidth(280)
        preview_row.addWidget(self.preview_combo, 1)
        preview_layout.addLayout(preview_row)
        self.preview = ImagePreview()
        preview_layout.addWidget(self.preview, 1)
        splitter.addWidget(preview_panel)
        preview_panel.setMinimumWidth(850)
        splitter.setSizes([600, 900])
        root.addWidget(splitter, 1)

        self._build_dataset_page()
        self._build_load_page()
        self._build_prepare_page()
        self._build_geometry_page()
        self._build_export_page()

        self.crop_roi = pg.RectROI((10, 10), (100, 100), pen=pg.mkPen("#ff9f1c", width=2))
        self.crop_roi.addScaleHandle((1, 1), (0, 0))
        self.crop_roi.addScaleHandle((0, 0), (1, 1))
        self.preview.view.getView().addItem(self.crop_roi, ignoreBounds=True)
        self.crop_roi.hide()
        self.dose_roi = pg.RectROI((10, 10), (100, 50), pen=pg.mkPen("#00b4d8", width=2))
        self.dose_roi.addScaleHandle((1, 1), (0, 0))
        self.dose_roi.addScaleHandle((0, 0), (1, 1))
        self.preview.view.getView().addItem(self.dose_roi, ignoreBounds=True)
        self.dose_roi.hide()

        footer = QtWidgets.QFrame(objectName="header")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        self.status_label = QtWidgets.QLabel("Select a tomography dataset to begin.")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMinimumWidth(260)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        footer_layout.addWidget(self.status_label, 1)
        footer_layout.addWidget(self.progress_bar)
        footer_layout.addWidget(self.cancel_button)
        root.addWidget(footer)

    def _page(self, title, text):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(22, 18, 22, 18)
        heading = QtWidgets.QLabel(title)
        font = heading.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        heading.setFont(font)
        introduction = QtWidgets.QLabel(text)
        introduction.setWordWrap(True)
        introduction.setProperty("muted", True)
        layout.addWidget(heading)
        layout.addWidget(introduction)
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        return layout

    @staticmethod
    def _spin(minimum, maximum, value, *, special=None):
        widget = QtWidgets.QSpinBox()
        widget.setRange(int(minimum), int(maximum))
        widget.setValue(int(value))
        if special is not None:
            widget.setSpecialValueText(special)
        return widget

    @staticmethod
    def _double(minimum, maximum, value, decimals=5):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(float(minimum), float(maximum))
        widget.setDecimals(decimals)
        widget.setValue(float(value))
        return widget

    @staticmethod
    def _configure_form(form):
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

    def _build_dataset_page(self):
        layout = self._page("Dataset", "Inspect file structure and projection angles before allocating a full image stack.")
        mode_group = QtWidgets.QGroupBox("Input source")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self.input_mode_combo = QtWidgets.QComboBox()
        self.input_mode_combo.addItem("Dataset folder and patterns", "patterns")
        self.input_mode_combo.addItem("Explicit file lists", "files")
        mode_layout.addWidget(self.input_mode_combo)
        self.input_stack = QtWidgets.QStackedWidget()
        pattern_widget = QtWidgets.QWidget()
        pattern_form = QtWidgets.QFormLayout(pattern_widget)
        self._configure_form(pattern_form)
        directory_row = QtWidgets.QWidget()
        directory_layout = QtWidgets.QHBoxLayout(directory_row)
        directory_layout.setContentsMargins(0, 0, 0, 0)
        self.data_directory_edit = QtWidgets.QLineEdit()
        self.data_directory_button = QtWidgets.QPushButton("Browse…")
        directory_layout.addWidget(self.data_directory_edit, 1)
        directory_layout.addWidget(self.data_directory_button)
        self.projection_pattern_edit = QtWidgets.QLineEdit("StackTomo*.tif")
        self.white_pattern_edit = QtWidgets.QLineEdit("WhiteField*.tif")
        self.dark_pattern_edit = QtWidgets.QLineEdit("DarkField*.tif")
        pattern_form.addRow("Folder", directory_row)
        pattern_form.addRow("Projections", self.projection_pattern_edit)
        pattern_form.addRow("White fields", self.white_pattern_edit)
        pattern_form.addRow("Dark fields", self.dark_pattern_edit)
        self.input_stack.addWidget(pattern_widget)
        explicit = QtWidgets.QWidget()
        explicit_layout = QtWidgets.QVBoxLayout(explicit)
        self.projection_card = FileSelectionCard("Projections", "One image per projection angle.")
        self.white_card = FileSelectionCard("White fields", "Repeated open-beam references.")
        self.dark_card = FileSelectionCard("Dark fields", "Repeated beam-off references.")
        for card in (self.projection_card, self.white_card, self.dark_card):
            card.set_last_directory(self._last_directory)
            explicit_layout.addWidget(card)
        self.input_stack.addWidget(explicit)
        mode_layout.addWidget(self.input_stack)
        layout.addWidget(mode_group)

        angle_group = QtWidgets.QGroupBox("Projection angles")
        angle_form = QtWidgets.QFormLayout(angle_group)
        self._configure_form(angle_form)
        self.angle_mode_combo = QtWidgets.QComboBox()
        self.angle_mode_combo.addItem("Read TIFF metadata", "metadata")
        self.angle_mode_combo.addItem("Manual range", "manual")
        angle_row = QtWidgets.QWidget()
        angle_row_layout = QtWidgets.QHBoxLayout(angle_row)
        angle_row_layout.setContentsMargins(0, 0, 0, 0)
        self.angle_start_spin = self._double(-3600, 3600, 0, 3)
        self.angle_stop_spin = self._double(-3600, 3600, 360, 3)
        self.angle_endpoint_check = QtWidgets.QCheckBox("Include stop")
        angle_row_layout.addWidget(QtWidgets.QLabel("Start"))
        angle_row_layout.addWidget(self.angle_start_spin)
        angle_row_layout.addWidget(QtWidgets.QLabel("Stop"))
        angle_row_layout.addWidget(self.angle_stop_spin)
        angle_row_layout.addWidget(self.angle_endpoint_check)
        angle_form.addRow("Source", self.angle_mode_combo)
        angle_form.addRow("Manual range", angle_row)
        layout.addWidget(angle_group)
        self.scan_button = QtWidgets.QPushButton("Scan dataset", objectName="primaryButton")
        self.dataset_summary = QtWidgets.QPlainTextEdit()
        self.dataset_summary.setReadOnly(True)
        self.dataset_summary.setMaximumHeight(150)
        self.angle_plot = pg.PlotWidget()
        self.angle_plot.setMaximumHeight(130)
        self.angle_plot.setLabel("left", "Angle", units="deg")
        self.angle_plot.setLabel("bottom", "Projection index")
        layout.addWidget(self.scan_button)
        layout.addWidget(self.dataset_summary)
        layout.addWidget(self.angle_plot)
        layout.addStretch(1)

    def _build_load_page(self):
        layout = self._page("Load and crop", "Test native-pixel crop and binning on a small projection subset.")
        group = QtWidgets.QGroupBox("Loading")
        form = QtWidgets.QFormLayout(group)
        self._configure_form(form)
        bin_row = QtWidgets.QWidget()
        bin_layout = QtWidgets.QHBoxLayout(bin_row)
        bin_layout.setContentsMargins(0, 0, 0, 0)
        self.ybin_spin = self._spin(1, 32, 1)
        self.xbin_spin = self._spin(1, 32, 1)
        bin_layout.addWidget(QtWidgets.QLabel("Y"))
        bin_layout.addWidget(self.ybin_spin)
        bin_layout.addWidget(QtWidgets.QLabel("X"))
        bin_layout.addWidget(self.xbin_spin)
        self.skip_spin = self._spin(1, 1000, 1)
        self.max_files_spin = self._spin(0, 1_000_000, 0, special="All")
        self.test_images_spin = self._spin(1, 50, 6)
        self.dtype_combo = QtWidgets.QComboBox()
        for value in ("float32", "uint16"):
            self.dtype_combo.addItem(value, value)
        self.file_workers_spin = self._spin(1, 64, 1)
        self.cores_spin = self._spin(1, 128, 4)
        self.crop_check = QtWidgets.QCheckBox("Use orange ROI as native-pixel crop")
        self.crop_label = QtWidgets.QLabel("No crop")
        form.addRow("Binning", bin_row)
        form.addRow("Projection stride", self.skip_spin)
        form.addRow("Maximum projections", self.max_files_spin)
        form.addRow("Test images", self.test_images_spin)
        form.addRow("Array dtype", self.dtype_combo)
        form.addRow("File workers", self.file_workers_spin)
        form.addRow("Processing cores", self.cores_spin)
        form.addRow(self.crop_check)
        form.addRow("Crop", self.crop_label)
        layout.addWidget(group)
        self.load_preview_button = QtWidgets.QPushButton("Load test subset", objectName="primaryButton")
        layout.addWidget(self.load_preview_button)
        layout.addStretch(1)

    def _build_prepare_page(self):
        layout = self._page("Prepare projections", "Run the standard NIT raw-to-attenuation pipeline in a background worker.")
        group = QtWidgets.QGroupBox("Reference and outlier processing")
        form = QtWidgets.QFormLayout(group)
        self._configure_form(form)
        self.reference_combo = QtWidgets.QComboBox()
        for value in ("mad_adaptive", "mad", "median", "mean"):
            self.reference_combo.addItem(value, value)
        self.outlier_check = QtWidgets.QCheckBox("Filter references and every projection")
        self.outlier_check.setChecked(True)
        self.outlier_size_spin = self._spin(3, 31, 5)
        self.outlier_size_spin.setSingleStep(2)
        self.outlier_sigma_spin = self._double(0.1, 100, 8.0, 2)
        self.calibration_frames_spin = self._spin(1, 1000, 8)
        form.addRow("Reference merge", self.reference_combo)
        form.addRow("Outlier filtering", self.outlier_check)
        form.addRow("Filter window", self.outlier_size_spin)
        form.addRow("Sigma multiplier", self.outlier_sigma_spin)
        form.addRow("Calibration frames", self.calibration_frames_spin)
        layout.addWidget(group)

        correction_group = QtWidgets.QGroupBox("Normalization and stripe correction")
        correction_form = QtWidgets.QFormLayout(correction_group)
        self._configure_form(correction_form)
        self.white_align_check = QtWidgets.QCheckBox("Align merged white field from feature points")
        self.white_points_edit = QtWidgets.QLineEdit()
        self.white_points_edit.setPlaceholderText("Native y,x points: 120,300; 500,900")
        self.dose_check = QtWidgets.QCheckBox("Normalize dose using the blue ROI")
        self.dose_label = QtWidgets.QLabel("No dose ROI")
        self.stripe_check = QtWidgets.QCheckBox("Remove all stripes")
        self.stripe_check.setChecked(True)
        stripe_row = QtWidgets.QWidget()
        stripe_layout = QtWidgets.QHBoxLayout(stripe_row)
        stripe_layout.setContentsMargins(0, 0, 0, 0)
        self.stripe_snr_spin = self._double(0.1, 100, 2.0, 2)
        self.stripe_large_spin = self._spin(3, 2001, 163)
        self.stripe_small_spin = self._spin(3, 501, 31)
        stripe_layout.addWidget(QtWidgets.QLabel("SNR"))
        stripe_layout.addWidget(self.stripe_snr_spin)
        stripe_layout.addWidget(QtWidgets.QLabel("Large"))
        stripe_layout.addWidget(self.stripe_large_spin)
        stripe_layout.addWidget(QtWidgets.QLabel("Small"))
        stripe_layout.addWidget(self.stripe_small_spin)
        diag_row = QtWidgets.QWidget()
        diag_layout = QtWidgets.QHBoxLayout(diag_row)
        diag_layout.setContentsMargins(0, 0, 0, 0)
        self.diag_projection_spin = self._spin(-1, 1_000_000, -1, special="Middle")
        self.diag_sinogram_spin = self._spin(-1, 1_000_000, -1, special="Middle")
        diag_layout.addWidget(QtWidgets.QLabel("Projection"))
        diag_layout.addWidget(self.diag_projection_spin)
        diag_layout.addWidget(QtWidgets.QLabel("Sinogram row"))
        diag_layout.addWidget(self.diag_sinogram_spin)
        correction_form.addRow(self.white_align_check)
        correction_form.addRow("Alignment points", self.white_points_edit)
        correction_form.addRow(self.dose_check)
        correction_form.addRow("Dose ROI", self.dose_label)
        correction_form.addRow("Stripe removal", self.stripe_check)
        correction_form.addRow("Stripe settings", stripe_row)
        correction_form.addRow("Diagnostics", diag_row)
        layout.addWidget(correction_group)
        self.prepare_button = QtWidgets.QPushButton("Prepare full projection stack", objectName="primaryButton")
        layout.addWidget(self.prepare_button)
        self.preparation_summary = QtWidgets.QPlainTextEdit()
        self.preparation_summary.setReadOnly(True)
        self.preparation_summary.setMaximumHeight(110)
        self.dose_plot = pg.PlotWidget()
        self.dose_plot.setMaximumHeight(140)
        self.dose_plot.setLabel("left", "Mean ROI transmission")
        self.dose_plot.setLabel("bottom", "Projection index")
        layout.addWidget(self.preparation_summary)
        layout.addWidget(self.dose_plot)
        layout.addStretch(1)

    def _build_geometry_page(self):
        layout = self._page("Geometry and trials", "Estimate geometry and retain quick reconstruction experiments for comparison.")
        self.capability_label = QtWidgets.QLabel()
        self.capability_label.setWordWrap(True)
        layout.addWidget(self.capability_label)
        geometry_group = QtWidgets.QGroupBox("Geometry")
        form = QtWidgets.QFormLayout(geometry_group)
        self._configure_form(form)
        self.geometry_combo = QtWidgets.QComboBox()
        self.geometry_combo.addItem("Parallel beam", "parallel")
        self.geometry_combo.addItem("Cone beam", "cone")
        self.pixel_source_combo = QtWidgets.QComboBox()
        self.pixel_source_combo.addItem("Read TIFF tag 65040", "tag")
        self.pixel_source_combo.addItem("Manual pixel size", "manual")
        self.pixel_size_spin = self._double(1e-7, 1000, 0.1, 7)
        self.source_object_spin = self._double(1e-6, 1e9, 1000.0, 3)
        self.source_detector_spin = self._double(1e-6, 1e9, 1200.0, 3)
        self.auto_center_check = QtWidgets.QCheckBox("Estimate center")
        self.auto_center_check.setChecked(True)
        self.center_spin = self._double(-1_000_000, 1_000_000, 0, 4)
        self.auto_tilt_check = QtWidgets.QCheckBox("Estimate detector tilt")
        self.auto_tilt_check.setChecked(True)
        self.tilt_spin = self._double(-90, 90, 0, 6)
        form.addRow("Beam geometry", self.geometry_combo)
        form.addRow("Pixel size source", self.pixel_source_combo)
        form.addRow("Pixel size (mm)", self.pixel_size_spin)
        form.addRow("Source-object distance (mm)", self.source_object_spin)
        form.addRow("Source-detector distance (mm)", self.source_detector_spin)
        form.addRow(self.auto_center_check, self.center_spin)
        form.addRow(self.auto_tilt_check, self.tilt_spin)
        layout.addWidget(geometry_group)
        geometry_buttons = QtWidgets.QHBoxLayout()
        self.geometry_button = QtWidgets.QPushButton("Estimate/apply LEAP geometry")
        self.tilt_compare_button = QtWidgets.QPushButton("Compare tilt")
        geometry_buttons.addWidget(self.geometry_button)
        geometry_buttons.addWidget(self.tilt_compare_button)
        layout.addLayout(geometry_buttons)

        trial_group = QtWidgets.QGroupBox("Reconstruction trial")
        trial_form = QtWidgets.QFormLayout(trial_group)
        self._configure_form(trial_form)
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItem("LEAP", "leap")
        self.backend_combo.addItem("TomoPy CPU fallback", "tomopy")
        self.method_combo = QtWidgets.QComboBox()
        self.trial_slice_spin = self._spin(0, 1_000_000, 0)
        self.trial_chunk_spin = self._spin(1, 500, 1)
        self.trial_pad_spin = self._spin(0, 500, 0)
        self.iterations_spin = self._spin(1, 10000, 20)
        self.preconditioner_combo = QtWidgets.QComboBox()
        self.preconditioner_combo.addItems(["SQS", "RAMP", "NONE"])
        self.tv_check = QtWidgets.QCheckBox("Total variation regularization")
        self.tv_beta_spin = self._double(0, 1e9, 10, 5)
        self.tv_delta_spin = self._double(0, 1, 2.5e-6, 8)
        self.trial_name_edit = QtWidgets.QLineEdit()
        self.trial_name_edit.setPlaceholderText("Optional trial name")
        trial_form.addRow("Backend", self.backend_combo)
        trial_form.addRow("Method", self.method_combo)
        trial_form.addRow("Slice / detector row", self.trial_slice_spin)
        trial_form.addRow("Chunk slices", self.trial_chunk_spin)
        trial_form.addRow("Padding per side", self.trial_pad_spin)
        trial_form.addRow("Iterations", self.iterations_spin)
        trial_form.addRow("Preconditioner", self.preconditioner_combo)
        trial_form.addRow(self.tv_check)
        trial_form.addRow("TV beta", self.tv_beta_spin)
        trial_form.addRow("TV delta", self.tv_delta_spin)
        trial_form.addRow("Name", self.trial_name_edit)
        layout.addWidget(trial_group)
        self.run_trial_button = QtWidgets.QPushButton("Run trial", objectName="primaryButton")
        layout.addWidget(self.run_trial_button)
        gallery_group = QtWidgets.QGroupBox("Trial gallery")
        gallery_layout = QtWidgets.QVBoxLayout(gallery_group)
        self.trial_list = QtWidgets.QListWidget()
        self.trial_detail = QtWidgets.QPlainTextEdit()
        self.trial_detail.setReadOnly(True)
        self.trial_detail.setMaximumHeight(100)
        self.promote_trial_button = QtWidgets.QPushButton("Use selected trial in recipe")
        gallery_layout.addWidget(self.trial_list)
        gallery_layout.addWidget(self.trial_detail)
        gallery_layout.addWidget(self.promote_trial_button)
        layout.addWidget(gallery_group)
        layout.addStretch(1)

    def _build_export_page(self):
        layout = self._page("Export and full reconstruction", "Review portable paths and reconstruction size before exporting or running locally.")
        paths_group = QtWidgets.QGroupBox("Local and cluster paths")
        paths_form = QtWidgets.QFormLayout(paths_group)
        self._configure_form(paths_form)
        self.local_data_edit = QtWidgets.QLineEdit()
        self.local_output_edit = QtWidgets.QLineEdit()
        self.cluster_data_edit = QtWidgets.QLineEdit()
        self.cluster_output_edit = QtWidgets.QLineEdit()
        self.output_browse_button = QtWidgets.QPushButton("Choose local output…")
        paths_form.addRow("Local data", self.local_data_edit)
        output_row = QtWidgets.QWidget()
        output_layout = QtWidgets.QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.local_output_edit, 1)
        output_layout.addWidget(self.output_browse_button)
        paths_form.addRow("Local output", output_row)
        paths_form.addRow("Cluster data", self.cluster_data_edit)
        paths_form.addRow("Cluster output", self.cluster_output_edit)
        layout.addWidget(paths_group)

        volume_group = QtWidgets.QGroupBox("Volume reconstruction")
        volume_form = QtWidgets.QFormLayout(volume_group)
        self._configure_form(volume_form)
        self.base_filename_edit = QtWidgets.QLineEdit("recon")
        self.z_range_edit = QtWidgets.QLineEdit()
        self.y_range_edit = QtWidgets.QLineEdit()
        self.x_range_edit = QtWidgets.QLineEdit()
        for widget in (self.z_range_edit, self.y_range_edit, self.x_range_edit):
            widget.setPlaceholderText("start, stop · blank means full range")
        self.chunk_size_spin = self._spin(1, 10000, 40)
        self.chunk_pad_spin = self._spin(0, 10000, 10)
        self.diagnostics_check = QtWidgets.QCheckBox("Save reconstruction diagnostics")
        self.diagnostics_check.setChecked(True)
        self.resume_check = QtWidgets.QCheckBox("Resume compatible reconstruction")
        self.resume_check.setChecked(True)
        self.overwrite_check = QtWidgets.QCheckBox("Overwrite existing reconstruction")
        volume_form.addRow("Base filename", self.base_filename_edit)
        volume_form.addRow("Z range", self.z_range_edit)
        volume_form.addRow("Y crop", self.y_range_edit)
        volume_form.addRow("X crop", self.x_range_edit)
        volume_form.addRow("Chunk size", self.chunk_size_spin)
        volume_form.addRow("Padding per side", self.chunk_pad_spin)
        volume_form.addRow(self.diagnostics_check)
        volume_form.addRow(self.resume_check)
        volume_form.addRow(self.overwrite_check)
        layout.addWidget(volume_group)
        action_row = QtWidgets.QHBoxLayout()
        self.preflight_button = QtWidgets.QPushButton("Full reconstruction preflight")
        self.full_recon_button = QtWidgets.QPushButton("Run full reconstruction", objectName="primaryButton")
        action_row.addWidget(self.preflight_button)
        action_row.addWidget(self.full_recon_button)
        layout.addLayout(action_row)
        self.preflight_summary = QtWidgets.QPlainTextEdit()
        self.preflight_summary.setReadOnly(True)
        self.preflight_summary.setMaximumHeight(180)
        layout.addWidget(self.preflight_summary)
        layout.addStretch(1)

    def _connect_signals(self):
        self.steps.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.input_mode_combo.currentIndexChanged.connect(self.input_stack.setCurrentIndex)
        self.data_directory_button.clicked.connect(self._choose_data_directory)
        self.output_browse_button.clicked.connect(self._choose_output_directory)
        for card in (self.projection_card, self.white_card, self.dark_card):
            card.directoryChanged.connect(self._remember_directory)
            card.previewRequested.connect(self._preview_raw_file)
            card.filesChanged.connect(self._invalidate_dataset)
        self.angle_mode_combo.currentIndexChanged.connect(self._update_angle_controls)
        self.scan_button.clicked.connect(self._scan_dataset)
        self.load_preview_button.clicked.connect(self._load_test_subset)
        self.crop_check.toggled.connect(self._show_preview_selection)
        self.crop_roi.sigRegionChangeFinished.connect(self._roi_changed)
        self.dose_check.toggled.connect(self._show_preview_selection)
        self.dose_roi.sigRegionChangeFinished.connect(self._roi_changed)
        self.prepare_button.clicked.connect(self._prepare)
        self.preview_combo.currentIndexChanged.connect(self._show_preview_selection)
        self.backend_combo.currentIndexChanged.connect(self._populate_methods)
        self.geometry_button.clicked.connect(self._estimate_geometry)
        self.tilt_compare_button.clicked.connect(self._compare_tilt)
        self.run_trial_button.clicked.connect(self._run_trial)
        self.trial_list.currentRowChanged.connect(self._show_trial)
        self.promote_trial_button.clicked.connect(self._promote_trial)
        self.preflight_button.clicked.connect(self._preflight)
        self.full_recon_button.clicked.connect(self._run_full_reconstruction)
        self.import_recipe_button.clicked.connect(self._import_recipe)
        self.export_recipe_button.clicked.connect(self._export_recipe)
        self.cancel_button.clicked.connect(self._runner.cancel)
        self._runner.busyChanged.connect(self._busy_changed)
        self._runner.progress.connect(self._progress)
        self._runner.succeeded.connect(self._job_succeeded)
        self._runner.failed.connect(self._job_failed)
        self._runner.cancelled.connect(self._job_cancelled)
        self.resume_check.toggled.connect(lambda checked: self.overwrite_check.setChecked(False) if checked else None)
        self.overwrite_check.toggled.connect(lambda checked: self.resume_check.setChecked(False) if checked else None)
        self.geometry_combo.currentIndexChanged.connect(self._geometry_type_changed)
        for widget, signal_name in (
            (self.data_directory_edit, "textChanged"),
            (self.projection_pattern_edit, "textChanged"),
            (self.white_pattern_edit, "textChanged"),
            (self.dark_pattern_edit, "textChanged"),
            (self.angle_mode_combo, "currentIndexChanged"),
            (self.angle_start_spin, "valueChanged"),
            (self.angle_stop_spin, "valueChanged"),
            (self.angle_endpoint_check, "toggled"),
        ):
            getattr(widget, signal_name).connect(self._invalidate_dataset)
        for widget, signal_name in (
            (self.ybin_spin, "valueChanged"), (self.xbin_spin, "valueChanged"),
            (self.skip_spin, "valueChanged"), (self.max_files_spin, "valueChanged"),
            (self.dtype_combo, "currentIndexChanged"), (self.crop_check, "toggled"),
        ):
            getattr(widget, signal_name).connect(self._invalidate_preparation)
        for widget, signal_name in (
            (self.reference_combo, "currentIndexChanged"), (self.outlier_check, "toggled"),
            (self.outlier_size_spin, "valueChanged"), (self.outlier_sigma_spin, "valueChanged"),
            (self.calibration_frames_spin, "valueChanged"), (self.white_align_check, "toggled"),
            (self.white_points_edit, "textChanged"), (self.dose_check, "toggled"),
            (self.stripe_check, "toggled"), (self.stripe_snr_spin, "valueChanged"),
            (self.stripe_large_spin, "valueChanged"), (self.stripe_small_spin, "valueChanged"),
        ):
            getattr(widget, signal_name).connect(self._invalidate_preparation)
        for widget, signal_name in (
            (self.geometry_combo, "currentIndexChanged"), (self.pixel_source_combo, "currentIndexChanged"),
            (self.pixel_size_spin, "valueChanged"), (self.auto_center_check, "toggled"),
            (self.center_spin, "valueChanged"), (self.auto_tilt_check, "toggled"),
            (self.tilt_spin, "valueChanged"), (self.source_object_spin, "valueChanged"),
            (self.source_detector_spin, "valueChanged"),
        ):
            getattr(widget, signal_name).connect(self._invalidate_geometry)
        self._update_angle_controls()
        self._geometry_type_changed()
        self._populate_methods()

    def _remember_directory(self, directory):
        self._last_directory = Path(directory).expanduser().resolve()
        for card in (self.projection_card, self.white_card, self.dark_card):
            card.set_last_directory(self._last_directory)
        self.directoryChanged.emit(str(self._last_directory))

    def set_last_directory(self, directory):
        self._last_directory = Path(directory).expanduser().resolve()
        for card in (self.projection_card, self.white_card, self.dark_card):
            card.set_last_directory(self._last_directory)

    def set_external_busy(self, busy):
        self._external_busy = bool(busy)
        self._update_actions()

    def _invalidate_dataset(self, *_args):
        if self._runner.busy:
            return
        self._manifest = None
        self.dataset_summary.clear()
        self.angle_plot.clear()
        self._preview_stack = None
        self._invalidate_preparation()

    def _invalidate_preparation(self, *_args):
        if self._runner.busy:
            return
        self._prepared = None
        self.preparation_summary.clear()
        self.dose_plot.clear()
        self._invalidate_geometry()

    def _invalidate_geometry(self, *_args):
        if self._runner.busy:
            return
        self._geometry = None
        self._tilt_previews = None
        self.preflight_summary.clear()
        self._update_actions()

    def _choose_data_directory(self):
        value = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose tomography dataset", str(self._last_directory))
        if value:
            self.data_directory_edit.setText(value)
            self.local_data_edit.setText(value)
            self._remember_directory(value)

    def _choose_output_directory(self):
        value = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose reconstruction output", str(self._last_directory))
        if value:
            self.local_output_edit.setText(value)
            self._remember_directory(value)

    def _input_spec(self):
        return TomographyInput(
            mode=str(self.input_mode_combo.currentData()), data_dir=self.data_directory_edit.text().strip(),
            projection_pattern=self.projection_pattern_edit.text().strip(), white_pattern=self.white_pattern_edit.text().strip(), dark_pattern=self.dark_pattern_edit.text().strip(),
            projection_files=self.projection_card.paths, white_files=self.white_card.paths, dark_files=self.dark_card.paths,
        )

    def _update_angle_controls(self):
        enabled = self.angle_mode_combo.currentData() == "manual"
        for widget in (self.angle_start_spin, self.angle_stop_spin, self.angle_endpoint_check):
            widget.setEnabled(enabled)

    def _scan_dataset(self):
        spec = self._input_spec()
        mode = str(self.angle_mode_combo.currentData())
        self._start_job("scan", lambda **_kwargs: inspect_dataset(spec, angle_mode=mode, manual_start=self.angle_start_spin.value(), manual_stop=self.angle_stop_spin.value(), manual_endpoint=self.angle_endpoint_check.isChecked()))

    def _preview_raw_file(self, path):
        try:
            self.preview.set_image(np.asarray(pg.imread(path)), f"Raw · {Path(path).name}")
        except Exception:
            try:
                import tifffile
                self.preview.set_image(tifffile.imread(path), f"Raw · {Path(path).name}")
            except Exception as exc:
                self.status_label.setText(str(exc))

    def _current_crop(self):
        if not self.crop_check.isChecked() or self._manifest is None:
            return None
        pos, size = self.crop_roi.pos(), self.crop_roi.size()
        yb, xb = self.ybin_spin.value(), self.xbin_spin.value()
        return [max(0, round(pos.y() * yb)), max(0, round(pos.x() * xb)), min(self._manifest.shape[0], round((pos.y() + size.y()) * yb)), min(self._manifest.shape[1], round((pos.x() + size.x()) * xb))]

    def _current_dose_roi(self):
        if not self.dose_check.isChecked() or self._manifest is None:
            return None
        pos, size = self.dose_roi.pos(), self.dose_roi.size()
        yb, xb = self.ybin_spin.value(), self.xbin_spin.value()
        return [max(0, round(pos.y() * yb)), max(0, round(pos.x() * xb)), min(self._manifest.shape[0], round((pos.y() + size.y()) * yb)), min(self._manifest.shape[1], round((pos.x() + size.x()) * xb))]

    def _roi_changed(self):
        crop = self._current_crop()
        dose = self._current_dose_roi()
        self.crop_label.setText("No crop" if crop is None else f"native y/x: {crop} · binned: {self._scale_roi(crop)}")
        self.dose_label.setText("No dose ROI" if dose is None else f"native y/x: {dose} · binned: {self._scale_roi(dose)}")
        self._prepared = None
        self._geometry = None
        self._update_actions()

    def _scale_roi(self, roi):
        if roi is None:
            return None
        yb, xb = self.ybin_spin.value(), self.xbin_spin.value()
        return [roi[0] // yb, roi[1] // xb, math.ceil(roi[2] / yb), math.ceil(roi[3] / xb)]

    def _load_test_subset(self):
        if self._manifest is None:
            return
        crop, binning, maximum = self._current_crop(), (self.ybin_spin.value(), self.xbin_spin.value()), self.test_images_spin.value()
        def load_previews(**_kwargs):
            full = load_preview_stack(
                self._manifest, crop=None, binning=binning, max_images=maximum
            )
            cropped = None
            if crop is not None:
                cropped = load_preview_stack(
                    self._manifest, crop=crop, binning=binning, max_images=maximum
                )
            return {"full": full, "cropped": cropped, "crop": crop}
        self._start_job("preview", load_previews)

    def _loading_settings(self):
        maximum = self.max_files_spin.value() or None
        return {"crop": self._current_crop(), "binning": [self.ybin_spin.value(), self.xbin_spin.value()], "skip_files": self.skip_spin.value(), "max_files": maximum, "white_skip_files": 1, "dark_skip_files": 1, "dtype": str(self.dtype_combo.currentData()), "file_workers": self.file_workers_spin.value()}

    def _parse_points(self):
        text = self.white_points_edit.text().strip()
        if not text:
            return []
        points = []
        for pair in text.split(";"):
            y, x = (float(value.strip()) for value in pair.split(",", 1))
            points.append([y, x])
        return points

    def _preparation_settings(self):
        stripe = {"enabled": self.stripe_check.isChecked(), "snr": self.stripe_snr_spin.value(), "la_size": self.stripe_large_spin.value(), "sm_size": self.stripe_small_spin.value(), "sizes_are_binned": False}
        outlier = {"size": self.outlier_size_spin.value(), "dif": "auto", "sigma_multiplier": self.outlier_sigma_spin.value(), "backend": "auto", "threshold_mode": "shared", "calibration_frames": self.calibration_frames_spin.value()}
        if not self.outlier_check.isChecked():
            outlier["fields"] = ()
        return {"loading": self._loading_settings(), "reference_method": str(self.reference_combo.currentData()), "outlier": outlier, "white_alignment": {"enabled": self.white_align_check.isChecked(), "points_yx": self._parse_points()}, "dose": {"enabled": self.dose_check.isChecked(), "roi": self._current_dose_roi()}, "stripe": stripe, "diagnostic_projection_index": None if self.diag_projection_spin.value() < 0 else self.diag_projection_spin.value(), "diagnostic_sinogram_row": None if self.diag_sinogram_spin.value() < 0 else self.diag_sinogram_spin.value(), "ncore": self.cores_spin.value()}

    def _prepare(self):
        if self._manifest is None:
            return
        settings = self._preparation_settings()
        self._start_job("prepare", lambda **kwargs: prepare_tomography(self._manifest, settings, **kwargs))

    def _geometry_settings(self):
        return {"binning": [self.ybin_spin.value(), self.xbin_spin.value()], "geometry": str(self.geometry_combo.currentData()), "pixel_size_mm": self.pixel_size_spin.value() if self.pixel_source_combo.currentData() == "manual" else None, "pixel_size_tag": "65040", "pixel_size_unit_scale": 0.001, "conebeam_kwargs": {"sod": self.source_object_spin.value(), "sdd": self.source_detector_spin.value()}, "estimate_center": self.auto_center_check.isChecked(), "estimate_tilt": self.auto_tilt_check.isChecked(), "center_col": None if self.auto_center_check.isChecked() else self.center_spin.value(), "tilt_degrees": None if self.auto_tilt_check.isChecked() else self.tilt_spin.value()}

    def _geometry_type_changed(self, *_args):
        cone = self.geometry_combo.currentData() == "cone"
        self.source_object_spin.setEnabled(cone)
        self.source_detector_spin.setEnabled(cone)
        self._update_actions()

    def _estimate_geometry(self):
        if self._prepared is None:
            return
        settings = self._geometry_settings()
        self._start_job("geometry", lambda **_kwargs: create_leap_geometry(self._prepared, settings))

    def _compare_tilt(self):
        if self._prepared is None or self._geometry is None:
            return
        index = self.trial_slice_spin.value()
        self._start_job("tilt", lambda **_kwargs: leap_tilt_comparison(self._prepared, self._geometry, slice_index=index))

    def _update_capabilities(self):
        if self._capabilities["leap"]:
            leap_text = f"available ({self._capabilities['leap_gpu_count']} GPU(s))"
        elif self._capabilities.get("leap_installed"):
            leap_text = "installed, but no usable GPU was detected"
        else:
            leap_text = "not installed in this environment"
        tomo_text = "available" if self._capabilities["tomopy"] else "not installed"
        self.capability_label.setText(f"LEAP: {leap_text}. TomoPy CPU fallback: {tomo_text}. CPU trials are screening results and are not numerically equivalent to LEAP cluster reconstructions.")
        leap_index = self.backend_combo.findData("leap")
        if leap_index >= 0:
            self.backend_combo.model().item(leap_index).setEnabled(self._capabilities["leap"])
        if not self._capabilities["leap"] and self._capabilities["tomopy"]:
            self.backend_combo.setCurrentIndex(self.backend_combo.findData("tomopy"))

    def _populate_methods(self):
        backend = self.backend_combo.currentData()
        current = self.method_combo.currentData()
        self.method_combo.clear()
        methods = ("FBP", "RWLS", "SIRT", "MLTR", "ASDPOCS") if backend == "leap" else ("gridrec", "fbp", "sirt")
        for method in methods:
            self.method_combo.addItem(method, method)
        index = self.method_combo.findData(current)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
        leap = backend == "leap"
        for widget in (self.trial_chunk_spin, self.trial_pad_spin, self.preconditioner_combo, self.tv_check, self.tv_beta_spin, self.tv_delta_spin):
            widget.setEnabled(leap)

    def _trial_regularization(self):
        return {"tv_enabled": self.tv_check.isChecked(), "beta": self.tv_beta_spin.value(), "delta": self.tv_delta_spin.value(), "p": 1.2, "weight": 1.0, "neighbors": 26}

    def _run_trial(self):
        if self._prepared is None:
            return
        backend, method = str(self.backend_combo.currentData()), str(self.method_combo.currentData())
        name = self.trial_name_edit.text().strip() or None
        if backend == "leap":
            if self._geometry is None:
                QtWidgets.QMessageBox.warning(self, "Geometry required", "Estimate or apply LEAP geometry first.")
                return
            self._start_job("trial", lambda **_kwargs: run_leap_trial(self._prepared, self._geometry, method=method, slice_index=self.trial_slice_spin.value(), chunk_size=self.trial_chunk_spin.value(), pad_each=self.trial_pad_spin.value(), num_iter=self.iterations_spin.value(), preconditioner=self.preconditioner_combo.currentText(), regularization=self._trial_regularization(), name=name))
        else:
            center = None if self.auto_center_check.isChecked() else self.center_spin.value()
            self._start_job("trial", lambda **_kwargs: run_tomopy_trial(self._prepared, method=method, slice_index=self.trial_slice_spin.value(), center=center, num_iter=self.iterations_spin.value(), ncore=self.cores_spin.value(), name=name))

    def _show_trial(self, row):
        if not 0 <= row < len(self._trials):
            self.trial_detail.clear()
            return
        trial = self._trials[row]
        self.preview_combo.blockSignals(True)
        self.preview_combo.setCurrentIndex(-1)
        self.preview_combo.setPlaceholderText(f"Trial · {trial.name}")
        self.preview_combo.blockSignals(False)
        self.preview.set_image(trial.image, trial.name)
        self.trial_detail.setPlainText(json.dumps(trial_summary(trial), indent=2))

    def _promote_trial(self):
        row = self.trial_list.currentRow()
        if 0 <= row < len(self._trials):
            self._selected_trial_index = row
            self.status_label.setText(f"Selected reconstruction recipe: {self._trials[row].name}")
            self._refresh_trial_labels()

    def _refresh_trial_labels(self):
        for index, trial in enumerate(self._trials):
            prefix = "★ " if index == self._selected_trial_index else ""
            self.trial_list.item(index).setText(f"{prefix}{trial.name} · {trial.elapsed_seconds:.2f} s")

    @staticmethod
    def _parse_range(text):
        text = text.strip()
        if not text:
            return None
        values = [int(value.strip()) for value in text.split(",")]
        if len(values) != 2 or values[0] >= values[1]:
            raise ValueError("Ranges must be entered as 'start, stop' with start < stop.")
        return values

    def _volume_settings(self):
        selected = self._trials[self._selected_trial_index] if self._selected_trial_index is not None else None
        method = selected.method if selected and selected.backend == "leap" else "FBP"
        parameters = selected.parameters if selected and selected.backend == "leap" else {}
        return {"output_dir": self.local_output_edit.text().strip(), "base_filename": self.base_filename_edit.text().strip() or "recon", "z_range": self._parse_range(self.z_range_edit.text()), "y_range": self._parse_range(self.y_range_edit.text()), "x_range": self._parse_range(self.x_range_edit.text()), "chunk_size": self.chunk_size_spin.value(), "pad_each": self.chunk_pad_spin.value(), "diagnostics": self.diagnostics_check.isChecked(), "resume": self.resume_check.isChecked(), "overwrite": self.overwrite_check.isChecked(), "method": method, "num_iter": parameters.get("num_iter", self.iterations_spin.value()), "preconditioner": parameters.get("preconditioner", self.preconditioner_combo.currentText()), "regularization": parameters.get("regularization", self._trial_regularization())}

    def _preflight(self):
        if self._prepared is None or self._geometry is None:
            QtWidgets.QMessageBox.warning(self, "LEAP result required", "Prepare data and apply LEAP geometry before full reconstruction preflight.")
            return None
        try:
            report = reconstruction_preflight(self._prepared, self._geometry, self._volume_settings())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Preflight failed", str(exc))
            return None
        self.preflight_summary.setPlainText("\n".join((f"Backend/GPU: {report['backend']} · {report['gpu_count']} GPU(s)", f"Projection shape: {tuple(report['projection_shape'])}", f"Output shape: {tuple(report['output_shape'])}", f"Estimated output: {report['output_bytes'] / 1024**3:.3f} GiB", f"Chunks: {report['chunk_count']}", f"Free disk: {report['free_disk_bytes'] / 1024**3:.3f} GiB", f"Disk margin: {'OK' if report['enough_disk'] else 'INSUFFICIENT'}", f"Destination: {report['output_dir']}")))
        return report

    def _run_full_reconstruction(self):
        report = self._preflight()
        if report is None or report["gpu_count"] < 1 or not report["enough_disk"]:
            return
        answer = QtWidgets.QMessageBox.question(self, "Start full reconstruction?", f"Reconstruct {tuple(report['output_shape'])} float32 volume in {report['chunk_count']} chunk(s)?\n\nOutput: {report['output_dir']}")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        settings = self._volume_settings()
        recipe = self.build_recipe()
        self._full_progress_timer.start()
        self._start_job("full", lambda **_kwargs: run_full_reconstruction(self._prepared, self._geometry, settings, manifest_metadata=recipe))

    def _poll_full_progress(self):
        try:
            settings = self._volume_settings()
            manifest_path = Path(settings["output_dir"]) / f"{settings['base_filename']}_progress.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks = payload.get("chunks", [])
            complete = sum(chunk.get("status") == "complete" for chunk in chunks)
            if chunks:
                self.progress_bar.setValue(round(100 * complete / len(chunks)))
                self.status_label.setText(f"Full reconstruction · {complete}/{len(chunks)} chunks complete")
        except (OSError, ValueError, TypeError):
            return

    def build_recipe(self):
        recipe = default_recipe()
        spec = self._input_spec()
        recipe["paths"] = {"local": {"data_dir": self.local_data_edit.text().strip() or spec.data_dir, "output_dir": self.local_output_edit.text().strip()}, "cluster": {"data_dir": self.cluster_data_edit.text().strip(), "output_dir": self.cluster_output_edit.text().strip()}}
        recipe["input"].update({"mode": spec.mode, "projection_pattern": spec.projection_pattern, "white_pattern": spec.white_pattern, "dark_pattern": spec.dark_pattern, "projection_files": list(spec.projection_files), "white_files": list(spec.white_files), "dark_files": list(spec.dark_files), "angle_source": str(self.angle_mode_combo.currentData()), "manual_angles": {"start": self.angle_start_spin.value(), "stop": self.angle_stop_spin.value(), "endpoint": self.angle_endpoint_check.isChecked()}})
        recipe["loading"] = self._loading_settings()
        preparation = self._preparation_settings()
        preparation.pop("loading", None)
        preparation.pop("ncore", None)
        recipe["preparation"] = preparation
        recipe["geometry"] = self._geometry_settings()
        selected = self._trials[self._selected_trial_index] if self._selected_trial_index is not None else None
        if selected:
            recipe["reconstruction"].update({"backend": selected.backend, "method": selected.method, **selected.parameters})
        recipe["volume"] = self._volume_settings()
        recipe["volume"]["output_dir"] = (
            self.cluster_output_edit.text().strip()
            or self.local_output_edit.text().strip()
        )
        recipe["selected_trial"] = trial_summary(selected)
        if self._manifest is not None:
            recipe["provenance"]["source_summary"] = {"counts": {key: len(value) for key, value in self._manifest.files.items()}, "shape": list(self._manifest.shape), "dtype": self._manifest.dtype, "estimated_bytes": self._manifest.estimated_bytes, "angles": self._manifest.angle_summary}
        return recipe

    def _export_recipe(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export tomography recipe", str(self._last_directory / "tomography_recipe.json"), "JSON recipe (*.json)")
        if not path:
            return
        try:
            destination = save_recipe(self.build_recipe(), path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Recipe export failed", str(exc))
            return
        self._remember_directory(destination.parent)
        self.status_label.setText(f"Exported recipe to {destination}")

    def _import_recipe(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import tomography recipe", str(self._last_directory), "JSON recipe (*.json)")
        if not path:
            return
        try:
            recipe = load_recipe(path)
            self.apply_recipe(recipe)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Recipe import failed", str(exc))
            return
        self._remember_directory(Path(path).parent)
        self.status_label.setText("Recipe loaded. Dataset has not been scanned or loaded.")

    def apply_recipe(self, recipe):
        self._recipe = copy.deepcopy(recipe)
        paths, inputs = recipe["paths"], recipe["input"]
        self.data_directory_edit.setText(paths["local"].get("data_dir", ""))
        self.local_data_edit.setText(paths["local"].get("data_dir", ""))
        self.local_output_edit.setText(paths["local"].get("output_dir", ""))
        self.cluster_data_edit.setText(paths["cluster"].get("data_dir", ""))
        self.cluster_output_edit.setText(paths["cluster"].get("output_dir", ""))
        self.input_mode_combo.setCurrentIndex(max(0, self.input_mode_combo.findData(inputs.get("mode", "patterns"))))
        self.projection_pattern_edit.setText(inputs.get("projection_pattern", "StackTomo*.tif"))
        self.white_pattern_edit.setText(inputs.get("white_pattern", "WhiteField*.tif"))
        self.dark_pattern_edit.setText(inputs.get("dark_pattern", "DarkField*.tif"))
        for card, values in ((self.projection_card, inputs.get("projection_files", [])), (self.white_card, inputs.get("white_files", [])), (self.dark_card, inputs.get("dark_files", []))):
            card.clear()
            if values:
                card.add_paths(values)
        self.angle_mode_combo.setCurrentIndex(max(0, self.angle_mode_combo.findData(inputs.get("angle_source", "metadata"))))
        manual = inputs.get("manual_angles", {})
        self.angle_start_spin.setValue(float(manual.get("start", 0)))
        self.angle_stop_spin.setValue(float(manual.get("stop", 360)))
        self.angle_endpoint_check.setChecked(bool(manual.get("endpoint", False)))
        loading = recipe["loading"]
        self.ybin_spin.setValue(int(loading.get("binning", [1, 1])[0]))
        self.xbin_spin.setValue(int(loading.get("binning", [1, 1])[1]))
        self.skip_spin.setValue(int(loading.get("skip_files", 1)))
        self.max_files_spin.setValue(int(loading.get("max_files") or 0))
        self.dtype_combo.setCurrentIndex(max(0, self.dtype_combo.findData(loading.get("dtype", "float32"))))
        self.file_workers_spin.setValue(int(loading.get("file_workers", 1)))
        preparation = recipe["preparation"]
        self.reference_combo.setCurrentIndex(max(0, self.reference_combo.findData(preparation.get("reference_method", "mad_adaptive"))))
        outlier = preparation.get("outlier", {})
        self.outlier_size_spin.setValue(int(outlier.get("size", 5)))
        self.outlier_sigma_spin.setValue(float(outlier.get("sigma_multiplier", 8)))
        self.calibration_frames_spin.setValue(int(outlier.get("calibration_frames", 8)))
        alignment = preparation.get("white_alignment", {})
        self.white_align_check.setChecked(bool(alignment.get("enabled", False)))
        self.white_points_edit.setText("; ".join(f"{point[0]},{point[1]}" for point in alignment.get("points_yx", [])))
        dose = preparation.get("dose", {})
        self.dose_check.setChecked(bool(dose.get("enabled", False)))
        self._pending_crop = loading.get("crop")
        self._pending_dose_roi = dose.get("roi")
        self.crop_check.setChecked(self._pending_crop is not None)
        stripe = preparation.get("stripe", {})
        self.stripe_check.setChecked(bool(stripe.get("enabled", True)))
        self.stripe_snr_spin.setValue(float(stripe.get("snr", 2)))
        self.stripe_large_spin.setValue(int(stripe.get("la_size", 163)))
        self.stripe_small_spin.setValue(int(stripe.get("sm_size", 31)))
        geometry = recipe["geometry"]
        self.geometry_combo.setCurrentIndex(max(0, self.geometry_combo.findData(geometry.get("geometry", "parallel"))))
        if geometry.get("pixel_size_mm") is not None:
            self.pixel_source_combo.setCurrentIndex(self.pixel_source_combo.findData("manual"))
            self.pixel_size_spin.setValue(float(geometry["pixel_size_mm"]))
        cone = geometry.get("conebeam_kwargs", {})
        self.source_object_spin.setValue(float(cone.get("sod", 1000.0)))
        self.source_detector_spin.setValue(float(cone.get("sdd", 1200.0)))
        self.auto_center_check.setChecked(bool(geometry.get("estimate_center", True)))
        self.auto_tilt_check.setChecked(bool(geometry.get("estimate_tilt", True)))
        if geometry.get("center_col") is not None:
            self.center_spin.setValue(float(geometry["center_col"]))
        if geometry.get("tilt_degrees") is not None:
            self.tilt_spin.setValue(float(geometry["tilt_degrees"]))
        reconstruction = recipe["reconstruction"]
        backend_index = self.backend_combo.findData(reconstruction.get("backend", "leap"))
        if backend_index >= 0:
            self.backend_combo.setCurrentIndex(backend_index)
        self._populate_methods()
        method_index = self.method_combo.findData(reconstruction.get("method", "FBP"))
        if method_index >= 0:
            self.method_combo.setCurrentIndex(method_index)
        if reconstruction.get("slice_index") is not None:
            self.trial_slice_spin.setValue(int(reconstruction["slice_index"]))
        self.iterations_spin.setValue(int(reconstruction.get("num_iter", 20)))
        preconditioner_index = self.preconditioner_combo.findText(
            reconstruction.get("preconditioner", "SQS")
        )
        if preconditioner_index >= 0:
            self.preconditioner_combo.setCurrentIndex(preconditioner_index)
        regularization = reconstruction.get("regularization", {})
        self.tv_check.setChecked(bool(regularization.get("tv_enabled", False)))
        self.tv_beta_spin.setValue(float(regularization.get("beta", 10.0)))
        self.tv_delta_spin.setValue(float(regularization.get("delta", 2.5e-6)))
        volume = recipe["volume"]
        self.base_filename_edit.setText(volume.get("base_filename", "recon"))
        for widget, key in ((self.z_range_edit, "z_range"), (self.y_range_edit, "y_range"), (self.x_range_edit, "x_range")):
            value = volume.get(key)
            widget.setText("" if value is None else f"{value[0]}, {value[1]}")
        self.chunk_size_spin.setValue(int(volume.get("chunk_size", 40)))
        self.chunk_pad_spin.setValue(int(volume.get("pad_each", 10)))
        self.diagnostics_check.setChecked(bool(volume.get("diagnostics", True)))
        self.resume_check.setChecked(bool(volume.get("resume", True)))
        self.overwrite_check.setChecked(bool(volume.get("overwrite", False)))
        self._manifest = self._preview_stack = self._prepared = self._geometry = None
        self._trials.clear()
        self.trial_list.clear()
        self._selected_trial_index = None
        self._update_actions()

    def _start_job(self, kind, function):
        if self._external_busy:
            return
        try:
            self._runner.start(kind, function)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Cannot start tomography operation", str(exc))

    def _busy_changed(self, busy):
        self.cancel_button.setEnabled(busy)
        self.busyChanged.emit(bool(busy))
        self._update_actions()

    def _progress(self, stage, current, total, message):
        self.status_label.setText(message)
        self.progress_bar.setValue(round(100 * current / max(1, total)))

    def _job_succeeded(self, kind, result):
        if kind == "full":
            self._full_progress_timer.stop()
        self.progress_bar.setValue(100)
        if kind == "scan":
            self._manifest = result
            summary = result.angle_summary
            self.dataset_summary.setPlainText("\n".join((f"Projections: {len(result.files['projections'])}", f"White fields: {len(result.files['white'])}", f"Dark fields: {len(result.files['dark'])}", f"Image: {result.shape[1]} × {result.shape[0]} · {result.dtype}", f"Estimated raw input: {result.estimated_bytes / 1024**3:.3f} GiB", f"Angles ({result.angle_source}): {summary['finite_count']}/{summary['count']} · {summary['minimum']} to {summary['maximum']} · median step {summary['median_step']}")))
            self.angle_plot.clear()
            self.angle_plot.plot(np.arange(len(result.angles)), result.angles, pen=pg.mkPen("#2f6f9f", width=2), symbol="o", symbolSize=4)
            self.local_data_edit.setText(self.data_directory_edit.text().strip())
            self.steps.setCurrentRow(1)
            self.status_label.setText("Dataset inspection complete.")
        elif kind == "preview":
            self._preview_stack = result["full"]
            images = {
                f"Raw test projection {index}": image
                for index, image in enumerate(result["full"])
            }
            if result["cropped"] is not None:
                images.update(
                    {
                        f"Cropped test projection {index}": image
                        for index, image in enumerate(result["cropped"])
                    }
                )
            if result["crop"] is not None:
                self._pending_crop = result["crop"]
            self._set_preview_collection(images)
            self._initialize_rois(result["full"][0].shape)
            self.status_label.setText(f"Loaded {len(result['full'])} test projection(s).")
        elif kind == "prepare":
            self._prepared = result
            self._geometry = None
            self._set_preview_collection(result.previews)
            outlier_lines = []
            for field, values in (result.diagnostics.get("outliers") or {}).items():
                changed = values.get("changed_fraction")
                if changed is not None:
                    outlier_lines.append(f"{field}: {100 * float(np.mean(changed)):.5f}% pixels changed")
            self.preparation_summary.setPlainText("\n".join(outlier_lines) or "Preparation completed; no outlier summary was returned.")
            self.dose_plot.clear()
            before = result.diagnostics.get("dose_before")
            after = result.diagnostics.get("dose_after")
            if before is not None:
                self.dose_plot.plot(np.arange(len(before)), before, pen=pg.mkPen("#d95f02", width=2), name="Before")
                self.dose_plot.plot(np.arange(len(after)), after, pen=pg.mkPen("#1b9e77", width=2), name="After")
            self.steps.setCurrentRow(3)
            self.status_label.setText(f"Prepared {result.data.projections.shape[0]} projections in {result.elapsed_seconds:.2f} s.")
            self.trial_slice_spin.setMaximum(max(result.data.projections.shape[1] - 1, 0))
            self.trial_slice_spin.setValue(result.data.projections.shape[1] // 2)
        elif kind == "geometry":
            self._geometry = result
            if result.center_col is not None:
                self.center_spin.setValue(float(result.center_col))
            if result.tilt_degrees is not None:
                self.tilt_spin.setValue(float(result.tilt_degrees))
            total = int(np.asarray(result.model.z_samples()).size)
            self.trial_slice_spin.setMaximum(max(0, total - 1))
            self.trial_slice_spin.setValue(total // 2)
            self.status_label.setText(f"LEAP geometry ready · center {result.center_col} · tilt {result.tilt_degrees}°")
        elif kind == "tilt":
            self._tilt_previews = result
            self._set_preview_collection({key: value for key, value in result.items() if isinstance(value, np.ndarray)})
            self.status_label.setText(f"Tilt comparison reconstructed at z={result['slice_index']}.")
        elif kind == "trial":
            reference = None
            if self._selected_trial_index is not None:
                reference = self._trials[self._selected_trial_index]
            elif self._trials:
                reference = self._trials[-1]
            if reference is not None and reference.image.shape == result.image.shape:
                difference = np.asarray(result.image, dtype=float) - np.asarray(reference.image, dtype=float)
                finite = difference[np.isfinite(difference)]
                if finite.size:
                    result.metrics.update(
                        {
                            "difference_reference": reference.name,
                            "mean_absolute_difference": float(np.mean(np.abs(finite))),
                            "root_mean_square_difference": float(np.sqrt(np.mean(finite**2))),
                        }
                    )
            self._trials.append(result)
            self.trial_list.addItem("")
            self._refresh_trial_labels()
            self.trial_list.setCurrentRow(len(self._trials) - 1)
            self.status_label.setText(f"Trial completed: {result.name}")
        elif kind == "full":
            self.status_label.setText(f"Full reconstruction complete: {result.completed_chunks}/{result.total_chunks} chunks.")
        self._update_actions()

    def _job_failed(self, kind, message, trace):
        if kind == "full":
            self._full_progress_timer.stop()
        log.error("Tomography %s failed:\n%s", kind, trace)
        self.status_label.setText(f"{kind.capitalize()} failed: {message}")
        QtWidgets.QMessageBox.critical(self, f"Tomography {kind} failed", message)

    def _job_cancelled(self, kind):
        if kind == "full":
            self._full_progress_timer.stop()
        self.status_label.setText(f"{kind.capitalize()} cancelled.")

    def _set_preview_collection(self, images):
        self._preview_images = dict(images)
        self.preview_combo.blockSignals(True)
        self.preview_combo.clear()
        self.preview_combo.setPlaceholderText("")
        for name in self._preview_images:
            self.preview_combo.addItem(name, name)
        self.preview_combo.blockSignals(False)
        if self.preview_combo.count():
            self.preview_combo.setCurrentIndex(0)
            self._show_preview_selection()

    def _show_preview_selection(self, *_args):
        name = self.preview_combo.currentData()
        if name and hasattr(self, "_preview_images") and name in self._preview_images:
            self.preview.set_image(self._preview_images[name], str(name))
            raw_test = str(name).startswith("Raw test")
            self.crop_roi.setVisible(self.crop_check.isChecked() and raw_test)
            self.dose_roi.setVisible(self.dose_check.isChecked() and raw_test)

    def _initialize_rois(self, binned_shape):
        yb, xb = self.ybin_spin.value(), self.xbin_spin.value()
        height, width = binned_shape
        if self._pending_crop is not None:
            y0, x0, y1, x1 = self._pending_crop
            self.crop_roi.setPos((x0 / xb, y0 / yb), update=False)
            self.crop_roi.setSize(((x1 - x0) / xb, (y1 - y0) / yb), update=True)
            self._pending_crop = None
        else:
            self.crop_roi.setPos((0, 0), update=False)
            self.crop_roi.setSize((width, height), update=True)
        if self._pending_dose_roi is not None:
            y0, x0, y1, x1 = self._pending_dose_roi
            self.dose_roi.setPos((x0 / xb, y0 / yb), update=False)
            self.dose_roi.setSize(((x1 - x0) / xb, (y1 - y0) / yb), update=True)
            self._pending_dose_roi = None
        else:
            dose_width, dose_height = max(2, width // 5), max(2, height // 8)
            self.dose_roi.setPos((0, 0), update=False)
            self.dose_roi.setSize((dose_width, dose_height), update=True)
        self._roi_changed()

    def _update_actions(self):
        busy = self._runner.busy or self._external_busy
        self.scan_button.setEnabled(not busy)
        self.load_preview_button.setEnabled(not busy and self._manifest is not None)
        self.prepare_button.setEnabled(not busy and self._manifest is not None)
        self.geometry_button.setEnabled(not busy and self._prepared is not None and self._capabilities["leap"])
        self.tilt_compare_button.setEnabled(not busy and self._geometry is not None and self._capabilities["leap"] and self.geometry_combo.currentData() == "parallel")
        backend = self.backend_combo.currentData()
        backend_ready = self._capabilities.get(str(backend), False)
        self.run_trial_button.setEnabled(not busy and self._prepared is not None and backend_ready and (backend != "leap" or self._geometry is not None))
        self.promote_trial_button.setEnabled(not busy and self.trial_list.currentRow() >= 0)
        full_ready = not busy and self._prepared is not None and self._geometry is not None and self._capabilities["leap"]
        self.preflight_button.setEnabled(full_ready)
        self.full_recon_button.setEnabled(full_ready)
        self.import_recipe_button.setEnabled(not busy)
        self.export_recipe_button.setEnabled(not busy)

    def prepare_close(self):
        self._full_progress_timer.stop()
        self.preview.prepare_close()
        self._runner.shutdown()
