"""Tomography dataset, preparation, reconstruction-trial, and recipe helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import concurrent.futures
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable

import numpy as np
import tifffile

from . import __version__
from .processing import ProcessingCancelled


TOMOGRAPHY_RECIPE_SCHEMA = "neutron-imaging-gui/tomography-recipe-v1"
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class TomographyInput:
    mode: str = "patterns"
    data_dir: str = ""
    projection_pattern: str = "StackTomo*.tif"
    white_pattern: str = "WhiteField*.tif"
    dark_pattern: str = "DarkField*.tif"
    projection_files: tuple[str, ...] = ()
    white_files: tuple[str, ...] = ()
    dark_files: tuple[str, ...] = ()

    def resolve_files(self) -> dict[str, tuple[str, ...]]:
        if self.mode not in {"patterns", "files"}:
            raise ValueError("Input mode must be 'patterns' or 'files'.")
        if self.mode == "patterns":
            root = Path(self.data_dir).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"Tomography data directory does not exist: {root}")
            groups = {
                "projections": tuple(str(path.resolve()) for path in sorted(root.glob(self.projection_pattern))),
                "white": tuple(str(path.resolve()) for path in sorted(root.glob(self.white_pattern))),
                "dark": tuple(str(path.resolve()) for path in sorted(root.glob(self.dark_pattern))),
            }
        else:
            groups = {
                "projections": _resolve_existing(self.projection_files),
                "white": _resolve_existing(self.white_files),
                "dark": _resolve_existing(self.dark_files),
            }
        for name, files in groups.items():
            if not files:
                raise FileNotFoundError(f"No {name} TIFF images were selected or matched.")
        return groups


@dataclass(frozen=True)
class DatasetManifest:
    files: dict[str, tuple[str, ...]]
    shape: tuple[int, int]
    dtype: str
    estimated_bytes: int
    angles: np.ndarray
    angle_source: str
    angle_summary: dict[str, float | int | None]


@dataclass
class PreparedTomography:
    data: Any
    previews: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    elapsed_seconds: float


@dataclass
class ReconstructionTrial:
    name: str
    backend: str
    method: str
    image: np.ndarray
    parameters: dict[str, Any]
    elapsed_seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def backend_capabilities() -> dict[str, Any]:
    """Report installed and actually usable reconstruction backends.

    LEAP can import successfully on a CPU-only machine while its reconstruction
    entry points still require an available GPU.  Treat that state as installed
    but unavailable so the GUI does not expose an operation that can terminate
    the Python process inside the native LEAP library.
    """
    leap_installed = importlib.util.find_spec("leapctype") is not None
    leap_gpu_count = 0
    leap_error = None
    if leap_installed:
        try:
            leap = importlib.import_module("leapctype")
            leap_gpu_count = max(0, int(leap.tomographicModels().number_of_gpus()))
        except Exception as exc:  # Native backend diagnostics vary by build.
            leap_error = str(exc)
    return {
        "tomopy": importlib.util.find_spec("tomopy") is not None,
        "leap": leap_installed and leap_gpu_count > 0,
        "leap_installed": leap_installed,
        "leap_gpu_count": leap_gpu_count,
        "leap_error": leap_error,
    }


def _patch_tomopy_numpy2():
    """Apply narrow NumPy/SciPy compatibility needed by older TomoPy releases."""
    if importlib.util.find_spec("tomopy") is None:
        return
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        from tomopy.util import dtype as tomopy_dtype

        if not getattr(tomopy_dtype.as_ndarray, "_neutron_gui_numpy2", False):
            def compatible_as_ndarray(value, dtype=None, copy=False):
                if not isinstance(value, np.ndarray):
                    return np.asarray(value, dtype=dtype)
                return value.copy() if copy else value

            compatible_as_ndarray._neutron_gui_numpy2 = True
            tomopy_dtype.as_ndarray = compatible_as_ndarray

    # TomoPy's remove_all_stripe still uses scipy.interpolate.interp2d,
    # removed in SciPy 1.14. This adapter preserves the regular-grid call form.
    import scipy
    if np.lib.NumpyVersion(scipy.__version__) >= "1.14.0":
        from scipy import interpolate

        if not getattr(interpolate.interp2d, "_neutron_gui_scipy114", False):
            def compatible_interp2d(x, y, z, kind="linear", **_kwargs):
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                z = np.asarray(z, dtype=float)
                order = 1 if kind == "linear" else 3
                spline = interpolate.RectBivariateSpline(
                    y,
                    x,
                    z,
                    kx=min(order, max(1, len(y) - 1)),
                    ky=min(order, max(1, len(x) - 1)),
                )
                return lambda new_x, new_y: spline(
                    np.asarray(new_y, dtype=float),
                    np.asarray(new_x, dtype=float),
                )

            compatible_interp2d._neutron_gui_scipy114 = True
            interpolate.interp2d = compatible_interp2d


def _resolve_existing(values) -> tuple[str, ...]:
    paths = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path))
    return tuple(paths)


def _first_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not matches:
        return None
    number = float(matches[-1])
    return number if math.isfinite(number) else None


def _angle_from_file(path) -> float | None:
    try:
        with tifffile.TiffFile(path) as tif:
            tags = {str(tag.name): tag.value for tag in tif.pages[0].tags.values()}
    except Exception:
        return None
    for key, value in tags.items():
        if re.search(r"angle|rotation|theta", f"{key} {value}", re.IGNORECASE):
            angle = _first_number(value)
            if angle is not None:
                return angle
    description = tags.get("ImageDescription")
    try:
        description = json.loads(description) if description else {}
    except (TypeError, ValueError):
        description = {}
    if isinstance(description, dict):
        for key, value in description.items():
            if re.search(r"angle|rotation|theta", str(key), re.IGNORECASE):
                angle = _first_number(value)
                if angle is not None:
                    return angle
    return None


def manual_angles(count: int, start: float, stop: float, *, endpoint=False) -> np.ndarray:
    if count < 1:
        raise ValueError("At least one projection is required.")
    if not math.isfinite(start) or not math.isfinite(stop) or start == stop:
        raise ValueError("Manual angle start and stop must be finite and different.")
    return np.linspace(float(start), float(stop), int(count), endpoint=bool(endpoint), dtype=np.float32)


def summarize_angles(angles) -> dict[str, float | int | None]:
    values = np.asarray(angles, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite_count": 0, "minimum": None, "maximum": None, "median_step": None}
    steps = np.diff(finite)
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "median_step": float(np.median(steps)) if steps.size else None,
    }


def inspect_dataset(
    inputs: TomographyInput,
    *,
    angle_mode="metadata",
    manual_start=0.0,
    manual_stop=360.0,
    manual_endpoint=False,
) -> DatasetManifest:
    files = inputs.resolve_files()
    expected_shape = None
    expected_dtype = None
    estimated_bytes = 0
    for group in files.values():
        for filename in group:
            with tifffile.TiffFile(filename) as tif:
                page = tif.pages[0]
                shape = tuple(int(value) for value in page.shape)
                dtype = np.dtype(page.dtype)
            if len(shape) != 2:
                raise ValueError(f"Expected 2D TIFF images; {filename} has shape {shape}.")
            if expected_shape is None:
                expected_shape, expected_dtype = shape, dtype
            elif shape != expected_shape:
                raise ValueError(f"Inconsistent TIFF shape: expected {expected_shape}, got {shape} for {filename}.")
            estimated_bytes += int(np.prod(shape)) * dtype.itemsize
    if angle_mode == "manual":
        angles = manual_angles(len(files["projections"]), manual_start, manual_stop, endpoint=manual_endpoint)
        source = "manual"
    else:
        angles = np.asarray([_angle_from_file(path) for path in files["projections"]], dtype=np.float32)
        if not np.all(np.isfinite(angles)):
            raise ValueError("Projection angle metadata is incomplete. Select manual angle range.")
        source = "metadata"
    return DatasetManifest(
        files=files,
        shape=expected_shape,
        dtype=str(expected_dtype),
        estimated_bytes=estimated_bytes,
        angles=angles,
        angle_source=source,
        angle_summary=summarize_angles(angles),
    )


def native_crop_to_slices(crop, shape):
    if crop is None:
        return (slice(None), slice(None))
    y0, x0, y1, x1 = (int(value) for value in crop)
    height, width = shape
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError(f"Crop {(y0, x0, y1, x1)} is outside image shape {shape}.")
    return (slice(y0, y1), slice(x0, x1))


def scale_native_roi(roi, binning=(1, 1)):
    if roi is None:
        return None
    ybin, xbin = (max(1, int(value)) for value in binning)
    y0, x0, y1, x1 = (int(value) for value in roi)
    return [y0 // ybin, x0 // xbin, math.ceil(y1 / ybin), math.ceil(x1 / xbin)]


def _bin_image(image, binning):
    ybin, xbin = (max(1, int(value)) for value in binning)
    height = image.shape[0] // ybin * ybin
    width = image.shape[1] // xbin * xbin
    trimmed = np.asarray(image[:height, :width], dtype=np.float32)
    if (ybin, xbin) == (1, 1):
        return trimmed
    return trimmed.reshape(height // ybin, ybin, width // xbin, xbin).mean(axis=(1, 3))


def load_preview_stack(manifest, *, crop=None, binning=(1, 1), max_images=6):
    slices = native_crop_to_slices(crop, manifest.shape)
    selected = manifest.files["projections"][: max(1, int(max_images))]
    return np.stack([_bin_image(tifffile.imread(path)[slices], binning) for path in selected]).astype(np.float32)


def _load_group(paths, *, crop, binning, skip=1, maximum=None, dtype="float32", workers=1, progress=None, cancel_event=None, stage="load"):
    chosen = tuple(paths)[:: max(1, int(skip))]
    if maximum is not None:
        chosen = chosen[: max(1, int(maximum))]
    def load_one(path):
        image = np.asarray(tifffile.imread(path))
        image = image[native_crop_to_slices(crop, image.shape)]
        return _bin_image(image, binning).astype(dtype, copy=False)

    arrays = [None] * len(chosen)
    worker_count = max(1, min(int(workers), len(chosen)))
    if worker_count == 1:
        for index, path in enumerate(chosen, 1):
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessingCancelled("Tomography loading cancelled.")
            arrays[index - 1] = load_one(path)
            if progress:
                progress(stage, index, len(chosen), f"Loaded {Path(path).name}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(load_one, path): (index, path) for index, path in enumerate(chosen)}
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                if cancel_event is not None and cancel_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    raise ProcessingCancelled("Tomography loading cancelled.")
                index, path = futures[future]
                arrays[index] = future.result()
                if progress:
                    progress(stage, completed, len(chosen), f"Loaded {Path(path).name}")
    return np.stack(arrays), chosen


def load_tomography_data(manifest, settings, *, progress=None, cancel_event=None):
    from neutron_imaging_tools.reconstruction import TomographyData

    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled("Tomography loading cancelled.")
    crop = settings.get("crop")
    binning = tuple(settings.get("binning", (1, 1)))
    dtype = settings.get("dtype", "float32")
    workers = settings.get("file_workers", 1)
    projections, projection_files = _load_group(
        manifest.files["projections"], crop=crop, binning=binning,
        skip=settings.get("skip_files", 1), maximum=settings.get("max_files"),
        dtype=dtype, workers=workers, progress=progress, cancel_event=cancel_event, stage="projections",
    )
    whites, white_files = _load_group(
        manifest.files["white"], crop=crop, binning=binning,
        skip=settings.get("white_skip_files", 1), maximum=settings.get("white_max_files"),
        dtype=dtype, workers=workers, progress=progress, cancel_event=cancel_event, stage="white",
    )
    darks, dark_files = _load_group(
        manifest.files["dark"], crop=crop, binning=binning,
        skip=settings.get("dark_skip_files", 1), maximum=settings.get("dark_max_files"),
        dtype=dtype, workers=workers, progress=progress, cancel_event=cancel_event, stage="dark",
    )
    indices = np.arange(len(manifest.files["projections"]))[:: max(1, int(settings.get("skip_files", 1)))]
    if settings.get("max_files") is not None:
        indices = indices[: int(settings["max_files"])]
    angles = np.asarray(manifest.angles[indices], dtype=np.float32)
    with tifffile.TiffFile(projection_files[0]) as tif:
        tags = {str(tag.code): tag.value for tag in tif.pages[0].tags.values()}
    return TomographyData(
        projections=projections,
        white_field=whites,
        dark_field=darks,
        angles=angles,
        metadata={"files": list(projection_files)},
        white_metadata={"files": list(white_files)},
        dark_metadata={"files": list(dark_files)},
        tif_tags=tags,
    )


def prepare_tomography(manifest, settings, *, progress=None, cancel_event=None) -> PreparedTomography:
    from neutron_imaging_tools import reconstruction as recon

    _patch_tomopy_numpy2()

    started = time.perf_counter()
    emit = progress or (lambda *_args: None)
    data = load_tomography_data(manifest, settings.get("loading", settings), progress=emit, cancel_event=cancel_event)
    projection_value = settings.get("diagnostic_projection_index")
    sinogram_value = settings.get("diagnostic_sinogram_row")
    projection_index = min(
        len(data.projections) // 2 if projection_value is None else int(projection_value),
        len(data.projections) - 1,
    )
    sinogram_row = min(
        data.projections.shape[1] // 2 if sinogram_value is None else int(sinogram_value),
        data.projections.shape[1] - 1,
    )
    previews = {"Raw projection": data.projections[projection_index].copy()}

    def check(stage, current, message):
        if cancel_event is not None and cancel_event.is_set():
            raise ProcessingCancelled("Tomography preparation cancelled.")
        emit(stage, current, 7, message)

    check("prepare", 1, "Merging white and dark references")
    data = recon.combine_tomography_references(data, method=settings.get("reference_method", "mad_adaptive"))
    previews["Merged white"] = data.white_combined.copy()
    previews["Merged dark"] = data.dark_combined.copy()

    check("prepare", 2, "Filtering detector and gamma outliers")
    outlier = dict(settings.get("outlier", {}))
    outlier.setdefault("size", 5)
    outlier.setdefault("dif", "auto")
    outlier.setdefault("sigma_multiplier", 8.0)
    outlier.setdefault("backend", "auto")
    outlier.setdefault("threshold_mode", "shared")
    outlier.setdefault("calibration_frames", 8)
    outlier.setdefault("ncore", settings.get("ncore"))
    data, outlier_diagnostics = recon.filter_tomography_outliers(data, return_diagnostics=True, **outlier)
    previews["Filtered projection"] = data.projections[projection_index].copy()

    alignment = settings.get("white_alignment", {})
    check("prepare", 3, "Applying optional white-field alignment")
    if alignment.get("enabled") and alignment.get("points_yx"):
        data = recon.align_whitefield_from_points(
            data,
            alignment["points_yx"],
            projection_index=projection_index,
            coordinates_are_binned=False,
            binning=tuple(settings.get("loading", {}).get("binning", (1, 1))),
        )
        previews["Aligned white"] = data.white_shifted.copy()

    check("prepare", 4, "Normalizing white and dark fields")
    data = recon.normalize_white_dark(data)
    previews["White/dark normalized"] = data.projections[projection_index].copy()

    dose = settings.get("dose", {})
    check("prepare", 5, "Applying optional beam-dose normalization")
    dose_before = dose_after = None
    if dose.get("enabled") and dose.get("roi"):
        y0, x0, y1, x1 = scale_native_roi(
            dose["roi"], tuple(settings.get("loading", {}).get("binning", (1, 1)))
        )
        dose_before = np.nanmean(data.projections[:, y0:y1, x0:x1], axis=(1, 2))
        data = recon.normalize_beam_roi(
            data, dose["roi"], binning=tuple(settings.get("loading", {}).get("binning", (1, 1))),
            roi_is_binned=False, ncore=settings.get("ncore"),
        )
        dose_after = np.nanmean(data.projections[:, y0:y1, x0:x1], axis=(1, 2))
        previews["Dose normalized"] = data.projections[projection_index].copy()

    check("prepare", 6, "Removing stripes")
    stripe = dict(settings.get("stripe", {}))
    if stripe.pop("enabled", True):
        projections = recon.remove_tomography_stripes(
            data.projections, binning=tuple(settings.get("loading", {}).get("binning", (1, 1))),
            ncore=settings.get("ncore"), **stripe,
        )
        from dataclasses import replace
        data = replace(data, projections=projections)
    previews["Stripe corrected projection"] = data.projections[projection_index].copy()
    previews["Stripe corrected sinogram"] = data.projections[:, sinogram_row, :].copy()

    check("prepare", 7, "Converting transmission to attenuation")
    from dataclasses import replace
    data = replace(data, projections=recon.linearize_projections(data.projections, ncore=settings.get("ncore")))
    previews["Attenuation projection"] = data.projections[projection_index].copy()
    previews["Attenuation sinogram"] = data.projections[:, sinogram_row, :].copy()
    return PreparedTomography(
        data=data,
        previews=previews,
        diagnostics={
            "outliers": outlier_diagnostics,
            "dose_before": dose_before,
            "dose_after": dose_after,
            "projection_index": projection_index,
            "sinogram_row": sinogram_row,
        },
        elapsed_seconds=time.perf_counter() - started,
    )


def run_tomopy_trial(prepared, *, method="gridrec", slice_index=None, center=None, num_iter=20, ncore=None, name=None):
    try:
        import tomopy
    except ImportError as exc:
        raise ImportError("TomoPy is required for the CPU reconstruction fallback.") from exc
    _patch_tomopy_numpy2()
    projections = np.asarray(prepared.data.projections, dtype=np.float32)
    row = projections.shape[1] // 2 if slice_index is None else int(slice_index)
    if not 0 <= row < projections.shape[1]:
        raise IndexError("Trial slice is outside the detector height.")
    slab = projections[:, row : row + 1, :]
    theta = np.deg2rad(np.asarray(prepared.data.angles, dtype=np.float32))
    kwargs = {}
    if method not in {"gridrec", "fbp"}:
        kwargs["num_iter"] = int(num_iter)
    started = time.perf_counter()
    reconstruction = tomopy.recon(slab, theta, center=center, algorithm=method, ncore=ncore, **kwargs)
    image = np.squeeze(np.asarray(reconstruction, dtype=np.float32))
    elapsed = time.perf_counter() - started
    finite = image[np.isfinite(image)]
    metrics = {
        "mean": float(np.mean(finite)) if finite.size else 0.0,
        "standard_deviation": float(np.std(finite)) if finite.size else 0.0,
        "gradient_energy": float(np.mean(np.hypot(*np.gradient(image)))) if image.size else 0.0,
    }
    return ReconstructionTrial(
        name=name or f"TomoPy {method} · row {row}", backend="tomopy", method=method,
        image=image, parameters={"slice_index": row, "center": center, "num_iter": int(num_iter)},
        elapsed_seconds=elapsed, metrics=metrics,
    )


def create_leap_geometry(prepared, settings):
    from neutron_imaging_tools import reconstruction as recon
    pixel_size = settings.get("pixel_size_mm")
    if pixel_size is not None:
        model = recon.create_leap_model(
            prepared.data.projections,
            prepared.data.angles,
            pixel_height=float(pixel_size),
            pixel_width=float(pixel_size),
            geometry=settings.get("geometry", "parallel"),
            center_col=settings.get("center_col"),
            conebeam_kwargs=settings.get("conebeam_kwargs"),
        )
        estimated_center, estimated_tilt = recon.estimate_leap_center_and_tilt(
            model,
            prepared.data.projections,
            estimate_center=settings.get("estimate_center", True) and settings.get("center_col") is None,
            estimate_tilt=settings.get("estimate_tilt", True) and settings.get("tilt_degrees") is None,
        )
        center = settings.get("center_col") if settings.get("center_col") is not None else estimated_center
        tilt = settings.get("tilt_degrees") if settings.get("tilt_degrees") is not None else estimated_tilt
        recon.apply_leap_center_and_tilt(model, center_col=center, tilt_degrees=tilt)
        info = recon.LeapGeometryInfo(
            model=model,
            center_col=center,
            tilt_degrees=tilt,
            pixel_height=float(pixel_size),
            pixel_width=float(pixel_size),
        )
        _select_all_leap_gpus(info.model)
        return info
    info = recon.prepare_leap_geometry(
        prepared.data,
        binning=tuple(settings.get("binning", (1, 1))),
        geometry=settings.get("geometry", "parallel"),
        estimate_center=settings.get("estimate_center", True),
        estimate_tilt=settings.get("estimate_tilt", True),
        center_col=settings.get("center_col"),
        tilt_degrees=settings.get("tilt_degrees"),
        pixel_size_tag=settings.get("pixel_size_tag", "65040"),
        unit_scale=settings.get("pixel_size_unit_scale", 1 / 1000),
        conebeam_kwargs=settings.get("conebeam_kwargs"),
    )
    _select_all_leap_gpus(info.model)
    return info


def _select_all_leap_gpus(model):
    count = int(model.number_of_gpus())
    if count > 0 and hasattr(model, "set_gpus"):
        model.set_gpus(list(range(count)))


def leap_tilt_comparison(prepared, geometry, *, slice_index=None):
    """Return matched LEAP FBP previews without and with the selected tilt."""
    import leapctype as leap
    from neutron_imaging_tools import reconstruction as recon

    total = int(np.asarray(geometry.model.z_samples()).size)
    index = total // 2 if slice_index is None else int(slice_index)
    untilted = leap.tomographicModels()
    untilted.copy_parameters(geometry.model)
    if geometry.tilt_degrees is not None:
        # Recreate the original parallel geometry because modular rotation is not reversible.
        pixel_height = float(geometry.pixel_height)
        pixel_width = float(geometry.pixel_width)
        untilted = recon.create_leap_model(
            prepared.data.projections,
            prepared.data.angles,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            geometry="parallel",
            center_col=geometry.center_col,
        )
        untilted.set_default_volume()
    before = np.squeeze(recon.reconstruct_leap_preview_slice(untilted, prepared.data.projections, slice_index=index)).copy()
    after = np.squeeze(recon.reconstruct_leap_preview_slice(geometry.model, prepared.data.projections, slice_index=index)).copy()
    return {"Without tilt": before, "With tilt": after, "Difference": after - before, "slice_index": index}


def _leap_tv_factory(regularization):
    if not regularization or not regularization.get("tv_enabled"):
        return None
    import leapctype as leap

    def make_filters(model):
        model.set_numTVneighbors(int(regularization.get("neighbors", 26)))
        sequence = leap.filterSequence(beta=float(regularization.get("beta", 10.0)))
        sequence.append(
            leap.TV(
                model,
                delta=float(regularization.get("delta", 2.5e-6)),
                p=float(regularization.get("p", 1.2)),
                weight=float(regularization.get("weight", 1.0)),
            )
        )
        return sequence
    return make_filters


def run_leap_trial(prepared, geometry, *, method="FBP", slice_index=None, chunk_size=1, pad_each=0, num_iter=20, preconditioner="SQS", regularization=None, name=None):
    from neutron_imaging_tools import reconstruction as recon
    total = int(np.asarray(geometry.model.z_samples()).size)
    start = total // 2 if slice_index is None else int(slice_index)
    started = time.perf_counter()
    result = recon.reconstruct_leap_chunk(
        geometry.model, prepared.data.projections, slice_start=start,
        chunk_size=max(1, int(chunk_size)), pad_each=max(0, int(pad_each)),
        iterative_method=None if method.upper() == "FBP" else method,
        num_iter=0 if method.upper() == "FBP" else int(num_iter), preconditioner=preconditioner,
        filters=_leap_tv_factory(regularization),
    )
    image = np.asarray(result.recon_core[len(result.recon_core) // 2], dtype=np.float32)
    finite = image[np.isfinite(image)]
    return ReconstructionTrial(
        name=name or f"LEAP {method} · z {start}", backend="leap", method=method.upper(), image=image,
        parameters={"slice_index": start, "chunk_size": int(chunk_size), "pad_each": int(pad_each), "num_iter": int(num_iter), "preconditioner": preconditioner, "regularization": regularization or {}},
        elapsed_seconds=time.perf_counter() - started,
        metrics={"mean": float(np.mean(finite)) if finite.size else 0.0, "standard_deviation": float(np.std(finite)) if finite.size else 0.0},
    )


def reconstruction_preflight(prepared, geometry, settings) -> dict[str, Any]:
    total_z = int(np.asarray(geometry.model.z_samples()).size)
    ny, nx = int(geometry.model.get_numY()), int(geometry.model.get_numX())
    z0, z1 = settings.get("z_range") or (0, total_z)
    y0, y1 = settings.get("y_range") or (0, ny)
    x0, x1 = settings.get("x_range") or (0, nx)
    if not (0 <= z0 < z1 <= total_z):
        raise ValueError(f"Z range {(z0, z1)} is outside [0, {total_z}).")
    if not (0 <= y0 < y1 <= ny):
        raise ValueError(f"Y crop {(y0, y1)} is outside [0, {ny}).")
    if not (0 <= x0 < x1 <= nx):
        raise ValueError(f"X crop {(x0, x1)} is outside [0, {nx}).")
    shape = (int(z1 - z0), int(y1 - y0), int(x1 - x0))
    output_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    if not str(settings.get("output_dir", "")).strip():
        raise ValueError("Choose a local reconstruction output directory.")
    output_dir = Path(settings["output_dir"]).expanduser().resolve()
    disk_root = output_dir if output_dir.exists() else next((p for p in output_dir.parents if p.exists()), Path.home())
    free = shutil.disk_usage(disk_root).free
    chunk_size = max(1, int(settings.get("chunk_size", 40)))
    return {
        "backend": "LEAP",
        "gpu_count": int(geometry.model.number_of_gpus()),
        "projection_shape": list(prepared.data.projections.shape),
        "output_shape": list(shape),
        "output_bytes": output_bytes,
        "chunk_count": math.ceil(shape[0] / chunk_size),
        "free_disk_bytes": int(free),
        "enough_disk": free > output_bytes * 1.1,
        "output_dir": str(output_dir),
    }


def run_full_reconstruction(prepared, geometry, settings, *, manifest_metadata=None):
    """Run the restartable NIT/LEAP volume reconstruction described by settings."""
    from neutron_imaging_tools import reconstruction as recon

    method = str(settings.get("method", "FBP")).upper()
    regularization = settings.get("regularization", {})
    filters = _leap_tv_factory(regularization)

    y_range = settings.get("y_range")
    x_range = settings.get("x_range")
    xy_crop = None
    if y_range is not None and x_range is not None:
        xy_crop = (slice(*y_range), slice(*x_range))
    return recon.reconstruct_leap_volume(
        geometry.model,
        prepared.data.projections,
        output_dir=settings["output_dir"],
        base_filename=settings.get("base_filename", "recon"),
        z_range=settings.get("z_range"),
        xy_crop=xy_crop,
        chunk_size=int(settings.get("chunk_size", 40)),
        pad_each=int(settings.get("pad_each", 10)),
        iterative_method=None if method == "FBP" else method,
        num_iter=0 if method == "FBP" else int(settings.get("num_iter", 20)),
        filters=filters,
        preconditioner=settings.get("preconditioner", "SQS"),
        export_tiff=True,
        resume=bool(settings.get("resume", True)),
        overwrite=bool(settings.get("overwrite", False)),
        manifest_metadata=manifest_metadata,
        progress=False,
        diagnostics_dir=(Path(settings["output_dir"]) / "diagnostics") if settings.get("diagnostics", True) else None,
    )


def default_recipe() -> dict[str, Any]:
    return {
        "schema": TOMOGRAPHY_RECIPE_SCHEMA,
        "software": {"neutron_imaging_gui": __version__, "neutron_imaging_tools": _version("neutron-imaging-tools")},
        "paths": {"local": {"data_dir": "", "output_dir": ""}, "cluster": {"data_dir": "", "output_dir": ""}},
        "input": {"mode": "patterns", "projection_pattern": "StackTomo*.tif", "white_pattern": "WhiteField*.tif", "dark_pattern": "DarkField*.tif", "projection_files": [], "white_files": [], "dark_files": [], "angle_source": "metadata", "manual_angles": {"start": 0.0, "stop": 360.0, "endpoint": False}},
        "loading": {"crop": None, "binning": [1, 1], "skip_files": 1, "max_files": None, "white_skip_files": 1, "dark_skip_files": 1, "dtype": "float32", "file_workers": 1},
        "preparation": {"reference_method": "mad_adaptive", "outlier": {"size": 5, "dif": "auto", "sigma_multiplier": 8.0, "backend": "auto", "threshold_mode": "shared", "calibration_frames": 8}, "white_alignment": {"enabled": False, "points_yx": []}, "dose": {"enabled": False, "roi": None}, "stripe": {"enabled": True, "snr": 2.0, "la_size": 163, "sm_size": 31, "sizes_are_binned": False}, "diagnostic_projection_index": None, "diagnostic_sinogram_row": None},
        "geometry": {"geometry": "parallel", "pixel_size_mm": None, "pixel_size_tag": "65040", "pixel_size_unit_scale": 0.001, "conebeam_kwargs": {"sod": 1000.0, "sdd": 1200.0}, "estimate_center": True, "estimate_tilt": True, "center_col": None, "tilt_degrees": None},
        "reconstruction": {"backend": "leap", "method": "FBP", "slice_index": None, "num_iter": 20, "preconditioner": "SQS", "regularization": {"tv_enabled": False, "beta": 10.0, "delta": 2.5e-6, "p": 1.2, "weight": 1.0, "neighbors": 26}},
        "volume": {"output_dir": "", "base_filename": "recon", "z_range": None, "y_range": None, "x_range": None, "chunk_size": 40, "pad_each": 10, "diagnostics": True, "resume": True, "overwrite": False},
        "selected_trial": None,
        "provenance": {"created_utc": None, "updated_utc": None, "source_summary": None},
    }


def validate_recipe(recipe) -> dict[str, Any]:
    if not isinstance(recipe, dict) or recipe.get("schema") != TOMOGRAPHY_RECIPE_SCHEMA:
        raise ValueError(f"Unsupported tomography recipe schema; expected {TOMOGRAPHY_RECIPE_SCHEMA!r}.")
    for key in ("paths", "input", "loading", "preparation", "geometry", "reconstruction", "volume"):
        if not isinstance(recipe.get(key), dict):
            raise ValueError(f"Tomography recipe is missing the {key!r} section.")
    return recipe


def save_recipe(recipe, path):
    payload = json.loads(json.dumps(validate_recipe(recipe), default=_json_default))
    payload.setdefault("provenance", {})["updated_utc"] = datetime.now(timezone.utc).isoformat()
    if payload["provenance"].get("created_utc") is None:
        payload["provenance"]["created_utc"] = payload["provenance"]["updated_utc"]
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_recipe(path):
    return validate_recipe(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))


def trial_summary(trial: ReconstructionTrial | None):
    if trial is None:
        return None
    return {"name": trial.name, "backend": trial.backend, "method": trial.method, "parameters": trial.parameters, "elapsed_seconds": trial.elapsed_seconds, "metrics": trial.metrics, "created_utc": trial.created_utc}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)
