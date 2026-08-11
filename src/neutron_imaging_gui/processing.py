"""Pure processing model for the initial radiography workflow."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import re
import threading
from typing import Callable, Iterable

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}
_REPEATED_EXPOSURE = re.compile(r"^(?P<group>.+)_(?P<index>\d+)$")
_SCAN_UID_SUFFIX = re.compile(r"^(?P<sample>.+)_(?P<uid>[0-9a-fA-F]{8})$")


class ProcessingCancelled(RuntimeError):
    """Raised when a queued reduction is cancelled cooperatively."""


@dataclass(frozen=True)
class ReductionConfig:
    white_files: tuple[str, ...]
    dark_files: tuple[str, ...]
    sample_files: tuple[str, ...]
    combine_matching_scans: bool = False
    merge_method: str = "mad_adaptive"
    gamma_filter: bool = True
    gamma_size: int = 5
    dose_normalization: bool = False
    dose_roi: tuple[int, int, int, int] | None = None  # x, y, width, height
    dose_rois: dict[str, tuple[int, int, int, int]] | None = None
    dose_statistic: str = "median"
    calculate_attenuation: bool = True
    attenuation_clip_min: float = 1e-6

    def validate(self) -> None:
        if not self.white_files:
            raise ValueError("Select at least one white-field image.")
        if not self.dark_files:
            raise ValueError("Select at least one dark-field image.")
        if not self.sample_files:
            raise ValueError("Select at least one sample image.")
        if self.merge_method not in {"mad_adaptive", "mad", "median", "mean"}:
            raise ValueError(f"Unsupported merge method: {self.merge_method}")
        if self.gamma_size < 1 or self.gamma_size % 2 == 0:
            raise ValueError("Gamma filter size must be a positive odd integer.")
        if self.dose_normalization:
            if self.dose_roi is None or self.dose_roi[2] <= 0 or self.dose_roi[3] <= 0:
                raise ValueError("Dose normalization requires a non-empty ROI.")
            if self.dose_rois is not None:
                for name, roi in self.dose_rois.items():
                    if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
                        raise ValueError(f"Dose ROI for '{name}' is empty.")
        if self.dose_statistic not in {"mean", "median"}:
            raise ValueError("Dose statistic must be mean or median.")
        if self.attenuation_clip_min <= 0:
            raise ValueError("Attenuation clip minimum must be positive.")


@dataclass
class ImageProducts:
    name: str
    files: tuple[str, ...]
    combined: np.ndarray
    transmission: np.ndarray
    attenuation: np.ndarray | None
    dose_scale: float | None = None


@dataclass
class ReductionResult:
    white: np.ndarray
    dark: np.ndarray
    products: "OrderedDict[str, ImageProducts]" = field(default_factory=OrderedDict)
    config: ReductionConfig | None = None


ProgressCallback = Callable[[str, int, int, str], None]


def discover_tiffs(paths: Iterable[str | Path]) -> list[str]:
    discovered: list[Path] = []
    for value in paths:
        path = Path(value).expanduser()
        if path.is_dir():
            discovered.extend(
                item for item in path.iterdir() if item.is_file() and item.suffix.lower() in TIFF_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in TIFF_SUFFIXES:
            discovered.append(path)
    return [str(path.resolve()) for path in sorted(set(discovered), key=_natural_key)]


def group_repeated_files(
    paths: Iterable[str | Path], *, combine_scans: bool = False
) -> "OrderedDict[str, tuple[str, ...]]":
    groups: dict[str, list[str]] = {}
    for value in sorted((str(Path(p)) for p in paths), key=_natural_key):
        path = Path(value)
        match = _REPEATED_EXPOSURE.match(path.stem)
        name = match.group("group") if match else path.stem
        if combine_scans:
            uid_match = _SCAN_UID_SUFFIX.match(name)
            if uid_match:
                name = uid_match.group("sample")
        groups.setdefault(name, []).append(str(path))
    return OrderedDict((name, tuple(files)) for name, files in sorted(groups.items()))


def load_image(path: str | Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(str(path)))
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = np.median(image[..., :3], axis=-1)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF image, got {image.shape} from {path}.")
    return np.asarray(image, dtype=np.float32)


def run_reduction(
    config: ReductionConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> ReductionResult:
    config.validate()
    groups = group_repeated_files(
        config.sample_files,
        combine_scans=config.combine_matching_scans,
    )
    reference_steps = 2 * (1 + int(config.gamma_filter))
    per_group_steps = 2 + int(config.gamma_filter) + int(config.calculate_attenuation)
    total = (
        len(config.white_files)
        + len(config.dark_files)
        + len(config.sample_files)
        + reference_steps
        + len(groups) * per_group_steps
    )
    completed = 0

    def update(stage: str, message: str, increment: int = 0) -> None:
        nonlocal completed
        _check_cancelled(cancel_event)
        completed += increment
        if progress is not None:
            progress(stage, completed, total, message)

    update("white", "Loading white-field images")
    white_stack = _load_stack(config.white_files, cancel_event, update, "white")
    white = _prepare_stack(white_stack, config, update, "white", "white field")

    update("dark", "Loading dark-field images")
    dark_stack = _load_stack(config.dark_files, cancel_event, update, "dark")
    dark = _prepare_stack(dark_stack, config, update, "dark", "dark field")

    if white.shape != dark.shape:
        raise ValueError(f"White and dark shapes differ: {white.shape} and {dark.shape}.")
    denominator = white - dark
    valid_denominator = np.isfinite(denominator) & (np.abs(denominator) > np.finfo(np.float32).eps)
    if not np.any(valid_denominator):
        raise ValueError("White minus dark contains no usable pixels.")

    products: "OrderedDict[str, ImageProducts]" = OrderedDict()
    for group_name, files in groups.items():
        update("samples", f"Loading {group_name}")
        stack = _load_stack(files, cancel_event, update, "samples")
        combined = _prepare_stack(stack, config, update, "samples", group_name)
        if combined.shape != white.shape:
            raise ValueError(
                f"Sample group '{group_name}' has shape {combined.shape}; expected {white.shape}."
            )
        sample_minus_dark = combined - dark
        dose_scale = None
        if config.dose_normalization:
            selected_roi = config.dose_roi
            if config.dose_rois:
                selected_roi = config.dose_rois.get(group_name, selected_roi)
            roi = _roi_slice(selected_roi, combined.shape)
            statistic = np.mean if config.dose_statistic == "mean" else np.median
            sample_dose = float(statistic(sample_minus_dark[roi]))
            white_dose = float(statistic(denominator[roi]))
            if not np.isfinite(sample_dose) or not np.isfinite(white_dose) or white_dose == 0:
                raise ValueError(f"Dose ROI is invalid for sample group '{group_name}'.")
            dose_scale = sample_dose / white_dose
            if not np.isfinite(dose_scale) or dose_scale <= 0:
                raise ValueError(f"Dose scale is not positive for sample group '{group_name}'.")
            sample_minus_dark = sample_minus_dark / dose_scale

        update("normalize", f"Normalizing {group_name}")
        transmission = np.full(combined.shape, np.nan, dtype=np.float32)
        np.divide(sample_minus_dark, denominator, out=transmission, where=valid_denominator)
        update("normalize", f"Transmission ready for {group_name}", 1)
        attenuation = None
        if config.calculate_attenuation:
            update("attenuation", f"Calculating attenuation for {group_name}")
            attenuation = -np.log(np.clip(transmission, config.attenuation_clip_min, None))
            attenuation = np.asarray(attenuation, dtype=np.float32)
            update("attenuation", f"Attenuation ready for {group_name}", 1)
        products[group_name] = ImageProducts(
            name=group_name,
            files=files,
            combined=np.asarray(combined, dtype=np.float32),
            transmission=transmission,
            attenuation=attenuation,
            dose_scale=dose_scale,
        )

    update("done", f"Finished {len(products)} sample group(s)")
    return ReductionResult(white=white, dark=dark, products=products, config=config)


def _load_stack(files, cancel_event, update, stage):
    images = []
    shape = None
    for file_path in files:
        _check_cancelled(cancel_event)
        image = load_image(file_path)
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(f"Image shape mismatch: {image.shape} versus {shape} for {file_path}.")
        images.append(image)
        if update is not None:
            update(stage, f"Loaded {Path(file_path).name}", 1)
    return np.stack(images, axis=0)


def _prepare_stack(stack, config, update=None, stage="processing", label="image group"):
    from neutron_imaging_tools.merging import combine_images

    if update is not None:
        update(stage, f"Merging {label} with {config.merge_method}")
    merged = combine_images(stack, method=config.merge_method, axis=0)
    merged = np.asarray(merged, dtype=np.float32)
    if update is not None:
        update(stage, f"Merged {label}", 1)
    if config.gamma_filter:
        from neutron_imaging_tools.filtering import remove_gammas

        if update is not None:
            update(stage, f"Gamma filtering {label}")
        merged = remove_gammas(
            merged,
            size=config.gamma_size,
            axis=0,
            symmetric=True,
            preserve_dtype=False,
        )
        if update is not None:
            update(stage, f"Gamma filtered {label}", 1)
    return np.asarray(merged, dtype=np.float32)


def _roi_slice(roi, shape):
    if roi is None:
        raise ValueError("ROI is required.")
    x, y, width, height = (int(value) for value in roi)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(shape[1], x + width), min(shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Dose ROI does not overlap the image.")
    return np.s_[y0:y1, x0:x1]


def _check_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled("Processing cancelled.")


def _natural_key(value: str | Path):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(value))]
