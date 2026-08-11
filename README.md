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
sample group.

The application intentionally runs disk-heavy work through a bounded FIFO worker
queue so that Qt remains responsive and large datasets are processed sequentially.
The progress bar and lower processing log track individual file loads as well as
each merge, gamma-filter, transmission, and attenuation stage.

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
