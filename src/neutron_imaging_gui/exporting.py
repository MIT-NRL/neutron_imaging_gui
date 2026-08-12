"""Image export, TIFF provenance, and reduction metadata helpers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import tifffile

from . import __version__


EXPORT_SCHEMA = "neutron-imaging-gui/export-v1"
_EXPOSURE_KEY = re.compile(r"(?:acquire|exposure)[ _.-]*time|^exposure$", re.IGNORECASE)
_EXPOSURE_TEXT = re.compile(
    r"(?:AcquireTime|ExposureTime|exposure(?:[ _.-]*time)?)\s*[:=]\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(ns|us|µs|μs|ms|s|sec|seconds|min)?",
    re.IGNORECASE,
)
_UNIT_SECONDS = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "μs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "seconds": 1.0,
    "min": 60.0,
}
_SKIP_TAGS = {
    "stripoffsets",
    "stripbytecounts",
    "tileoffsets",
    "tilebytecounts",
    "colormap",
}


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))
    return cleaned.strip("_") or "image"


def crop_image(image, bounds=None):
    array = np.asarray(image)
    if bounds is None:
        return array
    if array.ndim != 2:
        raise ValueError("Only 2D images can be exported.")
    x, y, width, height = (int(value) for value in bounds)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(array.shape[1], x + width)
    y1 = min(array.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Crop does not overlap the image.")
    return array[y0:y1, x0:x1]


def _json_safe(value: Any, *, max_items=64):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, max_items=max_items) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) > max_items:
            return {"item_count": len(value), "preview": [_json_safe(v) for v in value[:8]]}
        return [_json_safe(item, max_items=max_items) for item in value]
    if isinstance(value, np.ndarray):
        if value.size > max_items:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        return _json_safe(value.tolist(), max_items=max_items)
    return str(value)


def _positive_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _exposure_from_value(value, *, allow_numeric=False):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator = _positive_float(value[0])
        denominator = _positive_float(value[1])
        if numerator is not None and denominator is not None:
            return numerator / denominator
    if allow_numeric and not isinstance(value, (str, bytes, bytearray)):
        parsed = _positive_float(value)
        if parsed is not None:
            return parsed
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    text = str(value)
    match = _EXPOSURE_TEXT.search(text)
    if match:
        number = _positive_float(match.group(1))
        if number is not None:
            return number * _UNIT_SECONDS.get((match.group(2) or "s").lower(), 1.0)
    if allow_numeric:
        plain = re.fullmatch(
            r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(ns|us|µs|μs|ms|s|sec|seconds|min)?\s*",
            text,
            re.IGNORECASE,
        )
        if plain:
            number = _positive_float(plain.group(1))
            if number is not None:
                return number * _UNIT_SECONDS.get((plain.group(2) or "s").lower(), 1.0)
    return None


def _exposure_from_mapping(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if _EXPOSURE_KEY.search(str(key)):
                exposure = _exposure_from_value(item, allow_numeric=True)
                if exposure is not None:
                    return exposure
        for item in value.values():
            exposure = _exposure_from_mapping(item)
            if exposure is not None:
                return exposure
    elif isinstance(value, (list, tuple)):
        for item in value:
            exposure = _exposure_from_mapping(item)
            if exposure is not None:
                return exposure
    return None


def read_tiff_metadata(path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    metadata: dict[str, Any] = {
        "filename": source.name,
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "exposure_seconds": None,
        "tiff_tags": {},
    }
    try:
        with tifffile.TiffFile(source) as tif:
            page = tif.pages[0]
            metadata["shape"] = list(page.shape)
            metadata["dtype"] = str(page.dtype)
            for tag in page.tags.values():
                name = str(getattr(tag, "name", tag.code))
                normalized = re.sub(r"[^a-z0-9]", "", name.lower())
                if normalized not in _SKIP_TAGS:
                    metadata["tiff_tags"][name] = _json_safe(tag.value)
                exposure = None
                if int(getattr(tag, "code", -1)) == 33434 or _EXPOSURE_KEY.search(name):
                    exposure = _exposure_from_value(tag.value, allow_numeric=True)
                if exposure is None:
                    exposure = _exposure_from_value(tag.value)
                if exposure is not None and metadata["exposure_seconds"] is None:
                    metadata["exposure_seconds"] = exposure
            description = metadata["tiff_tags"].get("ImageDescription")
            if description:
                try:
                    parsed_description = json.loads(str(description))
                except (TypeError, ValueError, json.JSONDecodeError):
                    parsed_description = None
                exposure = _exposure_from_mapping(parsed_description)
                if exposure is not None:
                    metadata["exposure_seconds"] = exposure
    except Exception as exc:
        metadata["metadata_error"] = str(exc)
    return metadata


def _package_version(distribution: str, fallback="unknown") -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def build_export_manifest(result) -> dict[str, Any]:
    config = result.config
    file_groups = {
        "white": tuple(config.white_files),
        "dark": tuple(config.dark_files),
        "samples": tuple(config.sample_files),
    }
    input_metadata = {
        category: [read_tiff_metadata(path) for path in paths]
        for category, paths in file_groups.items()
    }
    known_exposures = [
        item["exposure_seconds"]
        for entries in input_metadata.values()
        for item in entries
        if item.get("exposure_seconds") is not None
    ]
    manifest = {
        "schema": EXPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "neutron_imaging_gui": __version__,
            "neutron_imaging_tools": _package_version("neutron-imaging-tools"),
            "numpy": _package_version("numpy", np.__version__),
            "tifffile": _package_version("tifffile", tifffile.__version__),
        },
        "timing": {
            "processing_started_utc": result.processing_started_utc,
            "processing_finished_utc": result.processing_finished_utc,
            "processing_elapsed_seconds": result.processing_elapsed_seconds,
            "known_exposure_count": len(known_exposures),
            "total_known_exposure_seconds": float(sum(known_exposures)),
            "input_file_count": sum(len(entries) for entries in input_metadata.values()),
        },
        "processing": _json_safe(asdict(config)),
        "inputs": input_metadata,
        "products": {
            name: {
                "source_files": [
                    str(Path(path).expanduser().resolve()) for path in product.files
                ],
                "source_filenames": [Path(path).name for path in product.files],
                "dose_scale": _json_safe(product.dose_scale),
                "shape": list(product.combined.shape),
            }
            for name, product in result.products.items()
        },
    }
    return _json_safe(manifest)


def image_export_metadata(manifest, product_name, image, *, crop_bounds=None):
    input_summary = {
        category: [
            {
                "filename": item.get("filename"),
                "path": item.get("path"),
                "exposure_seconds": item.get("exposure_seconds"),
            }
            for item in entries
        ]
        for category, entries in manifest.get("inputs", {}).items()
    }
    relevant_sources = {
        "white": input_summary.get("white", []),
        "dark": input_summary.get("dark", []),
    }
    product_text = str(product_name)
    if product_text == "reference_white":
        relevant_sources = {"white": input_summary.get("white", [])}
    elif product_text == "reference_dark":
        relevant_sources = {"dark": input_summary.get("dark", [])}
    else:
        for group_name, group in manifest.get("products", {}).items():
            if product_text in {
                f"{group_name}_combined",
                f"{group_name}_transmission",
                f"{group_name}_attenuation",
            }:
                sample_paths = set(group.get("source_files", []))
                relevant_sources["sample"] = [
                    item
                    for item in input_summary.get("samples", [])
                    if item.get("path") in sample_paths
                ]
                break
    return {
        "schema": EXPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "product": str(product_name),
        "shape": list(np.asarray(image).shape),
        "dtype": str(np.asarray(image).dtype),
        "crop": list(crop_bounds) if crop_bounds is not None else None,
        "software": manifest.get("software", {}),
        "timing": manifest.get("timing", {}),
        "processing": manifest.get("processing", {}),
        "products": manifest.get("products", {}),
        "inputs": input_summary,
        "source_inputs": relevant_sources,
    }


def write_json(path, payload) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_tiff(path, image, metadata=None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    description = json.dumps(_json_safe(metadata), ensure_ascii=False) if metadata else None
    software = None
    if metadata:
        versions = metadata.get("software", {})
        software = (
            f"neutron-imaging-gui {versions.get('neutron_imaging_gui', __version__)}; "
            f"neutron-imaging-tools {versions.get('neutron_imaging_tools', 'unknown')}"
        )
    tifffile.imwrite(
        destination,
        np.asarray(image, dtype=np.float32),
        description=description,
        software=software,
        metadata=None,
    )


def write_png(
    path,
    image,
    *,
    cmap="gray",
    levels=None,
    styled=False,
    colorbar=True,
    title="",
    dpi=150,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    vmin, vmax = levels if levels is not None else (None, None)
    if not styled:
        from matplotlib.image import imsave

        imsave(destination, array, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        return

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    height, width = array.shape
    figure_width = max(4.0, min(14.0, width / float(dpi)))
    figure_height = max(3.0, min(12.0, height / float(dpi)))
    figure = Figure(figsize=(figure_width, figure_height), dpi=dpi)
    FigureCanvasAgg(figure)
    populate_styled_figure(
        figure,
        array,
        cmap=cmap,
        levels=levels,
        colorbar=colorbar,
        title=title,
    )
    figure.savefig(destination, dpi=dpi)
    figure.clear()


def populate_styled_figure(
    figure,
    image,
    *,
    cmap="gray",
    levels=None,
    colorbar=True,
    title="",
    extent=None,
):
    """Populate a Matplotlib figure using the same layout as styled exports."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    array = np.asarray(image, dtype=np.float32)
    vmin, vmax = levels if levels is not None else (None, None)
    figure.clear()
    axes = figure.add_subplot(111)
    artist = axes.imshow(
        array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
    )
    axes.set_xlabel("X pixel")
    axes.set_ylabel("Y pixel")
    if title:
        axes.set_title(title)
    colorbar_axes = None
    if colorbar:
        divider = make_axes_locatable(axes)
        colorbar_axes = divider.append_axes("right", size="5%", pad=0.08)
        figure.colorbar(artist, cax=colorbar_axes, label="Intensity")
    figure.tight_layout(pad=1.0)
    return axes, colorbar_axes


def populate_raster_preview(figure, image, *, cmap="gray", levels=None, extent=None):
    """Fill a figure with only the raster, matching an image-only export."""
    array = np.asarray(image, dtype=np.float32)
    vmin, vmax = levels if levels is not None else (None, None)
    figure.clear()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.imshow(
        array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
        aspect="equal",
    )
    axes.set_axis_off()
    return axes


def batch_export_items(result, categories):
    selected = set(categories)
    items = []
    if "background" in selected:
        items.extend(
            (
                ("background", "reference_white.tif", "reference_white", result.white),
                ("background", "reference_dark.tif", "reference_dark", result.dark),
            )
        )
    for name, product in result.products.items():
        filename = safe_name(name)
        if "combined" in selected:
            items.append(
                ("combined", f"{filename}_combined.tif", f"{name}_combined", product.combined)
            )
        if "transmission" in selected:
            items.append(
                (
                    "transmission",
                    f"{filename}_transmission.tif",
                    f"{name}_transmission",
                    product.transmission,
                )
            )
        if "attenuation" in selected and product.attenuation is not None:
            items.append(
                (
                    "attenuation",
                    f"{filename}_attenuation.tif",
                    f"{name}_attenuation",
                    product.attenuation,
                )
            )
    return items


def export_reduction_batch(
    result,
    directory,
    *,
    categories,
    use_subfolders=True,
    embed_tiff_metadata=True,
    companion_json=True,
    overwrite=False,
):
    root = Path(directory)
    items = batch_export_items(result, categories)
    destinations = []
    for category, filename, product_name, image in items:
        target_directory = root / category if use_subfolders else root
        destinations.append((target_directory / filename, product_name, image, category))
    metadata_path = root / "metadata.json"
    conflicts = [path for path, _name, _image, _category in destinations if path.exists()]
    if companion_json and metadata_path.exists():
        conflicts.append(metadata_path)
    if conflicts and not overwrite:
        raise FileExistsError(
            f"{len(conflicts)} destination file(s) already exist; enable overwrite or choose another directory."
        )

    manifest = build_export_manifest(result)
    exported_files = []
    for path, product_name, image, category in destinations:
        metadata = image_export_metadata(manifest, product_name, image)
        metadata["category"] = category
        write_tiff(path, image, metadata if embed_tiff_metadata else None)
        exported_files.append(str(path.relative_to(root)))
    if companion_json:
        manifest["export"] = {
            "root": str(root),
            "subfolders": bool(use_subfolders),
            "categories": list(categories),
            "files": exported_files,
        }
        write_json(metadata_path, manifest)
    return {
        "root": root,
        "files": tuple(root / relative for relative in exported_files),
        "metadata": metadata_path if companion_json else None,
    }
