# Neutron Imaging GUI

A standalone Qt application for basic neutron radiography reduction. The first
workflow supports:

- selecting white-field, dark-field, and sample TIFF images;
- grouping repeated sample exposures by filename;
- optionally combining repeat scans of the same sample across Bluesky UID suffixes;
- robust merging through `NeutronImagingTools`;
- optional gamma filtering;
- optional open-beam ROI dose correction;
- white/dark transmission normalization and attenuation conversion; and
- previewing and exporting processed images.

Selecting a file automatically previews it. The viewer includes previous/next
navigation, percentile sliders for robust auto-levels, manual minimum/maximum
sliders, display gamma, selectable color maps, reset controls, and an always-visible
compact histogram with level markers. Display gamma and color mapping affect only
the viewer, not the processed arrays or exported TIFF values. Dose normalization
can use one shared open-beam ROI or remember a separate draggable ROI for every
sample group. Basic analysis overlays include an editable distance-measurement
line, physical pixel-size calibration from a known length, and a rectangular ROI
for live vertical-to-X or horizontal-to-Y mean/sum profiles.

Current images can be exported as cropped or full-resolution float TIFFs,
image-only PNGs, or Matplotlib-styled PNG figures with an optional colorbar.
Batch export can organize background references, combined samples, transmission,
and attenuation into separate folders. TIFF ImageDescription/Software tags and an
optional `metadata.json` record source files and TIFF tags, known exposure times,
processing settings and elapsed time, crop details, and GUI/NIT dependency versions.

The application keeps reduction jobs in a bounded FIFO queue so Qt remains
responsive. Within the active job, an optional process pool prepares independent
white-field, dark-field, and sample groups concurrently. The top-bar core setting
defaults to four processes; native worker threads are limited to avoid CPU
oversubscription. The progress bar and lower processing log track individual file
loads as well as each merge, gamma-filter, transmission, and attenuation stage.

## Development launch

From this checkout, using the beamline environment:

```bash
conda run -n bluesky-server python run_gui.py
```

Or install it in editable mode and use the console command:

```bash
python -m pip install -e .
neutron-imaging-gui
```

## Filename grouping

Sample files ending in an underscore and an integer are treated as repeated
exposures. For example, `sample_ab12cd34_0000.tif` and
`sample_ab12cd34_0001.tif` become one result named `sample_ab12cd34`. Files
without an integer suffix form their own result.
