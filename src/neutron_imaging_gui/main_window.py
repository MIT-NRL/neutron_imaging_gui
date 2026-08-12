"""Main window and staged workflow layout."""

from __future__ import annotations

from pathlib import Path
import logging
import os

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from .processing import ReductionConfig, ReductionResult, group_repeated_files, load_image
from .export_dialogs import BatchExportDialog, CurrentImageExportDialog
from .exporting import (
    build_export_manifest,
    crop_image,
    export_reduction_batch,
    image_export_metadata,
    safe_name,
    write_json,
    write_png,
    write_tiff,
)
from .widgets import FileSelectionCard, ImagePreview, StepList
from .workers import ReductionQueue
from .tomography_workspace import TomographyWorkspace
from .theme import apply_theme, saved_theme_mode


log = logging.getLogger(__name__)


STYLE = """
QMainWindow { background: palette(window); }
QFrame#header { background: palette(base); border-bottom: 1px solid palette(mid); }
QLabel#title { font-size: 22px; font-weight: 600; }
QLabel#subtitle, QLabel[muted="true"] { color: palette(mid); }
QListWidget#stepList { border: none; background: palette(base); padding: 8px; }
QListWidget#stepList::item { border-radius: 6px; padding: 8px; }
QListWidget#stepList::item:selected { background: #2f6f9f; color: white; }
QGroupBox { font-weight: 600; margin-top: 12px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { padding: 6px 12px; }
QPushButton#primaryButton { background: #2f6f9f; color: white; font-weight: 600; border-radius: 5px; }
QProgressBar { min-height: 18px; text-align: center; }
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_sample_paths=(), parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Neutron Imaging Reduction")
        self.resize(1500, 850)
        self.setMinimumSize(1450, 700)
        self.setStyleSheet(STYLE)
        self._result: ReductionResult | None = None
        self._current_job_id: str | None = None
        self._preview_paths: tuple[str, ...] = ()
        self._preview_index = -1
        self._preview_mode = "raw"
        self._shared_roi = (10, 10, 100, 100)
        self._group_rois: dict[str, tuple[int, int, int, int]] = {}
        self._last_directory = Path.home()
        self._tomography_busy = False
        self._queue = ReductionQueue(self)
        self._build_ui()
        self._build_menus()
        self._connect_signals()
        if initial_sample_paths:
            self.sample_card.add_paths(initial_sample_paths)
        self._update_run_state()

    def _build_ui(self) -> None:
        self.workspace_tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.workspace_tabs)
        central = QtWidgets.QWidget()
        self.workspace_tabs.addTab(central, "Radiography")
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QtWidgets.QFrame(objectName="header")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(24, 14, 24, 14)
        title = QtWidgets.QLabel("Neutron Imaging Reduction", objectName="title")
        title_row = QtWidgets.QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.multiprocessing_check = QtWidgets.QCheckBox("Background processes")
        self.multiprocessing_check.setChecked(True)
        self.multiprocessing_check.setToolTip(
            "Prepare independent white, dark, and sample groups in separate processes."
        )
        self.process_count_label = QtWidgets.QLabel("Cores")
        self.process_count_spin = QtWidgets.QSpinBox()
        self.process_count_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.process_count_spin.setValue(min(4, self.process_count_spin.maximum()))
        self.process_count_spin.setToolTip(
            "Maximum background processes. Native worker threads are limited "
            "to avoid oversubscription."
        )
        title_row.addWidget(self.multiprocessing_check)
        title_row.addWidget(self.process_count_label)
        title_row.addWidget(self.process_count_spin)
        subtitle = QtWidgets.QLabel(
            "Build references, combine repeated exposures, normalize transmission, and inspect attenuation.",
            objectName="subtitle",
        )
        header_layout.addLayout(title_row)
        header_layout.addWidget(subtitle)
        root.addWidget(header)

        horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(horizontal, 1)

        workflow = QtWidgets.QWidget()
        workflow_layout = QtWidgets.QHBoxLayout(workflow)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        self.steps = StepList()
        self.pages = QtWidgets.QStackedWidget()
        workflow_layout.addWidget(self.steps)
        workflow_layout.addWidget(self.pages, 1)
        horizontal.addWidget(workflow)

        preview_panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_header = QtWidgets.QHBoxLayout()
        preview_header.addWidget(QtWidgets.QLabel("Image preview"))
        preview_header.addStretch(1)
        preview_layout.addLayout(preview_header)
        self.preview = ImagePreview()
        preview_layout.addWidget(self.preview, 1)

        display_row = QtWidgets.QHBoxLayout()
        display_row.addWidget(QtWidgets.QLabel("Displayed result"))
        self.result_combo = QtWidgets.QComboBox()
        self.result_combo.setMinimumWidth(300)
        self.result_combo.setEnabled(False)
        display_row.addWidget(self.result_combo, 1)
        preview_layout.addLayout(display_row)
        horizontal.addWidget(preview_panel)
        horizontal.setSizes([600, 900])

        self._build_input_page()
        self._build_processing_page()
        self._build_normalization_page()
        self._build_review_page()

        self.progress_log = QtWidgets.QPlainTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMaximumBlockCount(1000)
        self.progress_log.setMaximumHeight(105)
        self.progress_log.setPlaceholderText("Detailed processing progress will appear here.")
        root.addWidget(self.progress_log)

        footer = QtWidgets.QFrame(objectName="header")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 10, 18, 10)
        self.status_label = QtWidgets.QLabel("Select input images to begin.")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumWidth(260)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.run_button = QtWidgets.QPushButton("Run reduction", objectName="primaryButton")
        footer_layout.addWidget(self.status_label, 1)
        footer_layout.addWidget(self.progress_bar)
        footer_layout.addWidget(self.cancel_button)
        footer_layout.addWidget(self.run_button)
        root.addWidget(footer)

        self.tomography = TomographyWorkspace(initial_directory=self._last_directory)
        self.workspace_tabs.addTab(self.tomography, "Tomography")

    def _build_menus(self) -> None:
        view_menu = self.menuBar().addMenu("View")
        theme_menu = view_menu.addMenu("Theme")
        self.theme_action_group = QtGui.QActionGroup(theme_menu)
        self.theme_action_group.setExclusive(True)
        self.theme_actions = {}
        current = saved_theme_mode()
        for label, mode, tooltip in (
            ("Light", "light", "Always use the light application theme."),
            ("Dark", "dark", "Always use the dark application theme."),
            ("System", "system", "Follow the desktop theme when the application starts."),
        ):
            action = theme_menu.addAction(label)
            action.setCheckable(True)
            action.setData(mode)
            action.setToolTip(tooltip)
            action.setChecked(mode == current)
            action.triggered.connect(
                lambda checked=False, selected=mode: self.set_theme_mode(selected)
            )
            self.theme_action_group.addAction(action)
            self.theme_actions[mode] = action

    def set_theme_mode(self, mode: str) -> None:
        apply_theme(mode, persist=True, root=self)
        # Qt resolves palette(...) references when a stylesheet is polished.
        # Reapply it so live theme changes update styled headers and navigation.
        self.setStyleSheet("")
        self.setStyleSheet(STYLE)
        for key, action in self.theme_actions.items():
            action.setChecked(key == mode)

    def _page(self, title: str, intro: str):
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
        intro_label = QtWidgets.QLabel(intro)
        intro_label.setWordWrap(True)
        intro_label.setProperty("muted", True)
        layout.addWidget(heading)
        layout.addWidget(intro_label)
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        return layout

    def _build_input_page(self):
        layout = self._page(
            "Select inputs",
            "Choose repeated white, dark, and sample exposures. Double-click any file to preview it.",
        )
        self.white_card = FileSelectionCard(
            "White field", "Open-beam images used to describe the incident beam and detector response."
        )
        self.dark_card = FileSelectionCard(
            "Dark field", "Beam-off images used to remove detector offset and dark current."
        )
        self.sample_card = FileSelectionCard(
            "Sample data", "Repeated exposures are grouped by the final numeric filename suffix."
        )
        layout.addWidget(self.white_card)
        layout.addWidget(self.dark_card)
        layout.addWidget(self.sample_card)
        layout.addStretch(1)

    def _build_processing_page(self):
        layout = self._page(
            "Prepare images",
            "References and each sample group use the same merge and detector-artifact settings.",
        )
        group = QtWidgets.QGroupBox("Merging and filtering")
        form = QtWidgets.QFormLayout(group)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.merge_combo = QtWidgets.QComboBox()
        self.merge_combo.addItem("MAD adaptive (recommended)", "mad_adaptive")
        self.merge_combo.addItem("MAD average", "mad")
        self.merge_combo.addItem("Median", "median")
        self.merge_combo.addItem("Mean", "mean")
        self.gamma_check = QtWidgets.QCheckBox("Remove bright and dark gamma events after merging")
        self.gamma_check.setChecked(True)
        self.gamma_size_spin = QtWidgets.QSpinBox()
        self.gamma_size_spin.setRange(3, 31)
        self.gamma_size_spin.setSingleStep(2)
        self.gamma_size_spin.setValue(5)
        self.combine_scans_check = QtWidgets.QCheckBox(
            "Combine matching sample scans across UIDs"
        )
        self.combine_scans_check.setToolTip(
            "For files such as sample_ab12cd34_0000.tif and sample_12ef5678_0000.tif, "
            "merge all exposures into one result named sample."
        )
        self.grouping_label = QtWidgets.QLabel("No sample groups selected.")
        self.grouping_label.setWordWrap(True)
        self.grouping_label.setProperty("muted", True)
        form.addRow("Merge method", self.merge_combo)
        form.addRow("Gamma filtering", self.gamma_check)
        form.addRow("Filter window", self.gamma_size_spin)
        form.addRow("Multiple scans", self.combine_scans_check)
        form.addRow("Result grouping", self.grouping_label)
        layout.addWidget(group)

        note = QtWidgets.QLabel(
            "Reduction jobs stay in one FIFO lane. Within the active job, independent "
            "reference and sample groups can use the background-process budget selected above."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        layout.addWidget(note)
        layout.addStretch(1)

    def _build_normalization_page(self):
        layout = self._page(
            "Normalize",
            "Transmission is calculated as (sample − dark) / (white − dark). An optional open-beam ROI can correct beam-dose drift first.",
        )
        dose_group = QtWidgets.QGroupBox("Dose normalization")
        dose_form = QtWidgets.QFormLayout(dose_group)
        dose_form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        dose_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.dose_check = QtWidgets.QCheckBox("Enable open-beam ROI dose correction")
        self.dose_stat_combo = QtWidgets.QComboBox()
        self.dose_stat_combo.addItem("Median", "median")
        self.dose_stat_combo.addItem("Mean", "mean")
        self.roi_mode_combo = QtWidgets.QComboBox()
        self.roi_mode_combo.addItem("Shared across all samples", "shared")
        self.roi_mode_combo.addItem("Different for each sample group", "per_group")
        self.roi_mode_combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.roi_sample_combo = QtWidgets.QComboBox()
        self.roi_sample_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
        )
        self.roi_sample_combo.setToolTip(
            "Choose a representative sample group, then position the orange ROI in the image preview."
        )
        self.roi_label = QtWidgets.QLabel("ROI: 10, 10, 100 × 100 px")
        self.roi_label.setWordWrap(True)
        dose_form.addRow(self.dose_check)
        dose_form.addRow("Statistic", self.dose_stat_combo)
        dose_form.addRow("ROI assignment", self.roi_mode_combo)
        dose_form.addRow("Preview sample", self.roi_sample_combo)
        dose_form.addRow("Preview selection", self.roi_label)
        roi_note = QtWidgets.QLabel(
            "The orange rectangle is editable directly on the sample preview. Page through raw images "
            "with the controls below the viewer to confirm that the selected region remains open beam."
        )
        roi_note.setWordWrap(True)
        roi_note.setProperty("muted", True)
        dose_form.addRow(roi_note)
        layout.addWidget(dose_group)

        attenuation_group = QtWidgets.QGroupBox("Attenuation")
        attenuation_form = QtWidgets.QFormLayout(attenuation_group)
        attenuation_form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        attenuation_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.attenuation_check = QtWidgets.QCheckBox("Calculate attenuation, −log(transmission)")
        self.attenuation_check.setChecked(True)
        self.clip_spin = QtWidgets.QDoubleSpinBox()
        self.clip_spin.setDecimals(8)
        self.clip_spin.setRange(1e-12, 1.0)
        self.clip_spin.setValue(1e-6)
        self.clip_spin.setSingleStep(1e-6)
        attenuation_form.addRow(self.attenuation_check)
        attenuation_form.addRow("Minimum transmission", self.clip_spin)
        layout.addWidget(attenuation_group)
        layout.addStretch(1)

    def _build_review_page(self):
        layout = self._page(
            "Review results",
            "Run the queued reduction, inspect each intermediate or final image, then export selected results.",
        )
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Processing summary will appear here.")
        self.summary.setMinimumHeight(220)
        layout.addWidget(self.summary)
        export_group = QtWidgets.QGroupBox("Export")
        export_layout = QtWidgets.QHBoxLayout(export_group)
        self.export_selected_button = QtWidgets.QPushButton("Export current image…")
        self.export_all_button = QtWidgets.QPushButton("Export all results…")
        self.export_selected_button.setEnabled(False)
        self.export_all_button.setEnabled(False)
        export_layout.addWidget(self.export_selected_button)
        export_layout.addWidget(self.export_all_button)
        export_layout.addStretch(1)
        layout.addWidget(export_group)
        layout.addStretch(1)

    def _connect_signals(self):
        self.steps.currentRowChanged.connect(self._stage_changed)
        for card in (self.white_card, self.dark_card, self.sample_card):
            card.filesChanged.connect(self._update_run_state)
            card.directoryChanged.connect(self._remember_directory)
            card.previewRequested.connect(
                lambda path, source=card: self._preview_file(path, source.paths)
            )
        self.sample_card.filesChanged.connect(self._refresh_sample_groups)
        self.combine_scans_check.toggled.connect(self._refresh_sample_groups)
        self.multiprocessing_check.toggled.connect(self._multiprocessing_toggled)
        self.roi_mode_combo.currentIndexChanged.connect(self._roi_mode_changed)
        self.roi_sample_combo.currentIndexChanged.connect(self._preview_roi_sample)
        self.preview.roiChanged.connect(self._update_roi_label)
        self.preview.previousRequested.connect(self._previous_image)
        self.preview.nextRequested.connect(self._next_image)
        self.run_button.clicked.connect(self._submit_reduction)
        self.cancel_button.clicked.connect(lambda: self._queue.cancel(self._current_job_id))
        self.result_combo.currentIndexChanged.connect(self._show_selected_result)
        self.export_selected_button.clicked.connect(self._export_selected)
        self.export_all_button.clicked.connect(self._export_all)

        self._queue.jobStarted.connect(self._job_started)
        self._queue.progress.connect(self._job_progress)
        self._queue.succeeded.connect(self._job_succeeded)
        self._queue.failed.connect(self._job_failed)
        self._queue.cancelled.connect(self._job_cancelled)
        self._queue.queueChanged.connect(self._radiography_queue_changed)
        self.tomography.busyChanged.connect(self._tomography_busy_changed)
        self.tomography.directoryChanged.connect(self._remember_directory)

    def _remember_directory(self, directory) -> None:
        path = Path(directory).expanduser().resolve()
        self._last_directory = path
        for card in (self.white_card, self.dark_card, self.sample_card):
            card.set_last_directory(path)
        self.tomography.set_last_directory(path)

    def _radiography_queue_changed(self, count):
        self.tomography.set_external_busy(bool(count))

    def _tomography_busy_changed(self, busy):
        self._tomography_busy = bool(busy)
        self._update_run_state()

    def _build_config(self) -> ReductionConfig:
        return ReductionConfig(
            white_files=self.white_card.paths,
            dark_files=self.dark_card.paths,
            sample_files=self.sample_card.paths,
            combine_matching_scans=self.combine_scans_check.isChecked(),
            merge_method=str(self.merge_combo.currentData()),
            gamma_filter=self.gamma_check.isChecked(),
            gamma_size=self.gamma_size_spin.value(),
            dose_normalization=self.dose_check.isChecked(),
            dose_roi=self._shared_roi,
            dose_rois=(
                dict(self._group_rois)
                if self.roi_mode_combo.currentData() == "per_group"
                else None
            ),
            dose_statistic=str(self.dose_stat_combo.currentData()),
            calculate_attenuation=self.attenuation_check.isChecked(),
            attenuation_clip_min=self.clip_spin.value(),
            use_multiprocessing=self.multiprocessing_check.isChecked(),
            process_count=self.process_count_spin.value(),
        )

    def _multiprocessing_toggled(self, enabled):
        self.process_count_label.setEnabled(enabled)
        self.process_count_spin.setEnabled(enabled)

    def _update_run_state(self):
        ready = bool(self.white_card.paths and self.dark_card.paths and self.sample_card.paths)
        self.run_button.setEnabled(
            ready and self._current_job_id is None and not self._tomography_busy
        )
        if self._current_job_id is None and self._result is None:
            self.status_label.setText(
                "Ready to process selected images." if ready else "Select input images to begin."
            )

    def _preview_file(self, path: str, sequence=None):
        try:
            image = load_image(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Preview failed", str(exc))
            return
        if sequence is None:
            sequence = next(
                (
                    card.paths
                    for card in (self.white_card, self.dark_card, self.sample_card)
                    if path in card.paths
                ),
                (path,),
            )
        self._preview_paths = tuple(sequence)
        self._preview_index = self._preview_paths.index(path) if path in self._preview_paths else 0
        self._preview_mode = "raw"
        self.preview.set_position(self._preview_index + 1, len(self._preview_paths))
        self.preview.set_image(image, f"Raw input · {Path(path).name} · {image.shape[1]} × {image.shape[0]}")

    def _update_roi_label(self, roi):
        x, y, width, height = roi
        self.roi_label.setText(f"ROI: {x}, {y}, {width} × {height} px")
        bounds = (x, y, width, height)
        if self.roi_mode_combo.currentData() == "per_group":
            name = self.roi_sample_combo.currentData()
            if name:
                self._group_rois[str(name)] = bounds
        else:
            self._shared_roi = bounds

    def _previous_image(self):
        if self._preview_mode == "result" and self.result_combo.count() > 1:
            self.result_combo.setCurrentIndex(
                (self.result_combo.currentIndex() - 1) % self.result_combo.count()
            )
            return
        if self._preview_paths:
            index = (self._preview_index - 1) % len(self._preview_paths)
            self._preview_file(self._preview_paths[index], self._preview_paths)

    def _next_image(self):
        if self._preview_mode == "result" and self.result_combo.count() > 1:
            self.result_combo.setCurrentIndex(
                (self.result_combo.currentIndex() + 1) % self.result_combo.count()
            )
            return
        if self._preview_paths:
            index = (self._preview_index + 1) % len(self._preview_paths)
            self._preview_file(self._preview_paths[index], self._preview_paths)

    def _stage_changed(self, index):
        self.pages.setCurrentIndex(index)
        on_normalize_page = index == 2
        self.preview.set_roi_visible(on_normalize_page)
        if on_normalize_page:
            self._refresh_sample_groups()
            self._preview_roi_sample()

    def _refresh_sample_groups(self):
        groups = group_repeated_files(
            self.sample_card.paths,
            combine_scans=self.combine_scans_check.isChecked(),
        )
        scan_word = "combined result" if self.combine_scans_check.isChecked() else "result"
        exposures = sum(len(files) for files in groups.values())
        self.grouping_label.setText(
            f"{exposures} exposure(s) will produce {len(groups)} {scan_word}(s)."
            if groups
            else "No sample groups selected."
        )
        previous = self.roi_sample_combo.currentData()
        self.roi_sample_combo.blockSignals(True)
        self.roi_sample_combo.clear()
        for name, files in groups.items():
            self.roi_sample_combo.addItem(
                f"{name} ({len(files)} exposure{'s' if len(files) != 1 else ''})",
                name,
            )
            index = self.roi_sample_combo.count() - 1
            self.roi_sample_combo.setItemData(index, files[0], QtCore.Qt.UserRole + 1)
        if previous is not None:
            previous_index = self.roi_sample_combo.findData(previous)
            if previous_index >= 0:
                self.roi_sample_combo.setCurrentIndex(previous_index)
        self.roi_sample_combo.blockSignals(False)
        self.roi_sample_combo.setEnabled(bool(groups))

    def _roi_mode_changed(self):
        self.roi_sample_combo.setEnabled(self.roi_sample_combo.count() > 0)
        self._apply_roi_for_current_group()

    def _apply_roi_for_current_group(self):
        bounds = self._shared_roi
        if self.roi_mode_combo.currentData() == "per_group":
            name = self.roi_sample_combo.currentData()
            if name:
                bounds = self._group_rois.setdefault(str(name), self._shared_roi)
        self.preview.set_roi_bounds(bounds)
        self._update_roi_label(bounds)

    def _preview_roi_sample(self):
        index = self.roi_sample_combo.currentIndex()
        if index < 0:
            return
        path = self.roi_sample_combo.itemData(index, QtCore.Qt.UserRole + 1)
        if path:
            self._preview_file(str(path), self.sample_card.paths)
        self._apply_roi_for_current_group()

    def _submit_reduction(self):
        try:
            config = self._build_config()
            config.validate()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot run reduction", str(exc))
            return
        self._result = None
        self.result_combo.clear()
        self.result_combo.setEnabled(False)
        self.summary.clear()
        self.progress_log.clear()
        self.progress_bar.setValue(0)
        self.progress_log.appendPlainText(
            "Queued reduction: "
            f"{len(config.white_files)} white, {len(config.dark_files)} dark, "
            f"{len(config.sample_files)} sample image(s); merge={config.merge_method}; "
            f"gamma={'on' if config.gamma_filter else 'off'}; "
            f"combine scans={'on' if config.combine_matching_scans else 'off'}; "
            f"processes={config.process_count if config.use_multiprocessing else 1}."
        )
        self._current_job_id = self._queue.submit(config)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Reduction queued…")

    @QtCore.Slot(str)
    def _job_started(self, job_id):
        if job_id == self._current_job_id:
            self.status_label.setText("Starting reduction…")

    @QtCore.Slot(str, str, int, int, str)
    def _job_progress(self, job_id, stage, current, total, message):
        if job_id != self._current_job_id:
            return
        percent = round(100 * current / total) if total else 0
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.status_label.setText(message)
        self.progress_log.appendPlainText(
            f"[{stage}] {current}/{total} ({percent:3d}%)  {message}"
        )
        scrollbar = self.progress_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @QtCore.Slot(str, object)
    def _job_succeeded(self, job_id, result):
        if job_id != self._current_job_id:
            return
        self._result = result
        self._current_job_id = None
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Finished {len(result.products)} sample group(s).")
        self.cancel_button.setEnabled(False)
        self._populate_results()
        self.steps.setCurrentRow(3)
        self._update_run_state()

    @QtCore.Slot(str, str, str)
    def _job_failed(self, job_id, message, trace):
        log.error("Reduction failed:\n%s", trace)
        if job_id != self._current_job_id:
            return
        self._current_job_id = None
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Reduction failed.")
        self.progress_log.appendPlainText(f"[failed] {message}")
        self._update_run_state()
        QtWidgets.QMessageBox.critical(self, "Reduction failed", message)

    @QtCore.Slot(str)
    def _job_cancelled(self, job_id):
        if job_id != self._current_job_id:
            return
        self._current_job_id = None
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Reduction cancelled.")
        self.progress_log.appendPlainText("[cancelled] Processing cancelled by user.")
        self._update_run_state()

    def _populate_results(self):
        result = self._result
        if result is None:
            return
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        self.result_combo.addItem("Reference · White", ("reference", "white"))
        self.result_combo.addItem("Reference · Dark", ("reference", "dark"))
        for name, product in result.products.items():
            self.result_combo.addItem(f"{name} · Combined", (name, "combined"))
            self.result_combo.addItem(f"{name} · Transmission", (name, "transmission"))
            if product.attenuation is not None:
                self.result_combo.addItem(f"{name} · Attenuation", (name, "attenuation"))
        self.result_combo.blockSignals(False)
        self.result_combo.setEnabled(True)
        self.export_selected_button.setEnabled(True)
        self.export_all_button.setEnabled(True)
        lines = [
            f"White fields: {len(result.config.white_files)}",
            f"Dark fields: {len(result.config.dark_files)}",
            f"Sample groups: {len(result.products)}",
            f"Combine matching scans: {'on' if result.config.combine_matching_scans else 'off'}",
            f"Merge method: {result.config.merge_method}",
            f"Gamma filter: {'on' if result.config.gamma_filter else 'off'}",
            f"Background processes: "
            f"{result.config.process_count if result.config.use_multiprocessing else 'off'}",
            f"Dose normalization: {'on' if result.config.dose_normalization else 'off'}",
            "",
        ]
        for name, product in result.products.items():
            detail = f"{name}: {len(product.files)} exposure(s)"
            if product.dose_scale is not None:
                detail += f", dose scale={product.dose_scale:.6g}"
            lines.append(detail)
        self.summary.setPlainText("\n".join(lines))
        self.result_combo.setCurrentIndex(0)
        self._show_selected_result()

    def _selected_array(self):
        if self._result is None:
            return None, ""
        key = self.result_combo.currentData()
        if not key:
            return None, ""
        name, kind = key
        if name == "reference":
            return getattr(self._result, kind), f"reference_{kind}"
        return getattr(self._result.products[name], kind), f"{name}_{kind}"

    def _show_selected_result(self):
        array, name = self._selected_array()
        if array is not None:
            self._preview_mode = "result"
            self.preview.set_position(self.result_combo.currentIndex() + 1, self.result_combo.count())
            self.preview.set_image(array, f"Processed result · {name}")

    def _export_selected(self):
        array, name = self._selected_array()
        if array is None or self._result is None:
            return
        profile_bounds = (
            self.preview.profile_bounds() if self.preview.profile_check.isChecked() else None
        )
        dialog = CurrentImageExportDialog(
            image=array,
            image_name=safe_name(name),
            initial_directory=self._last_directory,
            colormap=str(self.preview.colormap_combo.currentData()),
            viewer_levels=self.preview.levels(),
            profile_bounds=profile_bounds,
            parent=self,
        )
        dialog.directorySelected.connect(self._remember_directory)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        options = dialog.options()
        try:
            exported = crop_image(array, options.crop_bounds)
            manifest = build_export_manifest(self._result)
            metadata = image_export_metadata(
                manifest,
                name,
                exported,
                crop_bounds=options.crop_bounds,
            )
            if options.format == "tiff":
                write_tiff(
                    options.path,
                    exported,
                    metadata if options.embed_tiff_metadata else None,
                )
            else:
                levels = (
                    self.preview.levels()
                    if options.use_viewer_levels
                    else options.manual_levels
                )
                write_png(
                    options.path,
                    exported,
                    cmap=options.cmap,
                    levels=levels,
                    styled=options.format == "png_styled",
                    colorbar=options.colorbar,
                    title=options.title,
                    dpi=options.dpi,
                )
            if options.companion_json:
                payload = dict(manifest)
                payload["export"] = metadata
                write_json(options.path.with_suffix(".json"), payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._remember_directory(options.path.parent)
        self.status_label.setText(f"Saved {options.path}")

    def _export_all(self):
        if self._result is None:
            return
        dialog = BatchExportDialog(
            initial_directory=self._last_directory,
            parent=self,
        )
        dialog.directorySelected.connect(self._remember_directory)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        options = dialog.options()
        if not options.categories:
            QtWidgets.QMessageBox.warning(self, "Nothing to export", "Select at least one product.")
            return
        try:
            exported = export_reduction_batch(
                self._result,
                options.directory,
                categories=options.categories,
                use_subfolders=options.use_subfolders,
                embed_tiff_metadata=options.embed_tiff_metadata,
                companion_json=options.companion_json,
                overwrite=options.overwrite,
            )
        except FileExistsError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Export files already exist",
                str(exc),
            )
            return
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._remember_directory(options.directory)
        self.status_label.setText(
            f"Exported {len(exported['files'])} image(s) to {options.directory}"
        )

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.preview.prepare_close()
        self.tomography.prepare_close()
        self._queue.shutdown()
        super().closeEvent(event)
