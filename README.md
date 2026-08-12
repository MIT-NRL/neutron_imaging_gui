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

## Tomography workspace

The top-level **Tomography** tab provides a separate preparation and
reconstruction-testing workflow. It accepts either a dataset directory with
projection/reference patterns or explicit TIFF lists, inspects angle metadata,
and supports a manual angle-range fallback. Crop and dose ROIs are stored in
native detector pixels even when the interactive previews are binned.

Tomography preparation follows the NIT sequence: reference merging, adaptive
outlier filtering, optional white-field alignment, white/dark normalization,
optional beam-dose normalization, stripe removal, and attenuation conversion.
The preview selector retains reference, projection, and sinogram checkpoints.

When LEAP is available, the workspace can estimate center/tilt, compare tilt
correction, test FBP or iterative reconstruction methods, and run a guarded,
restartable chunked volume reconstruction. Without LEAP, TomoPy
gridrec/FBP/SIRT trials remain available for screening and are explicitly
labeled as non-equivalent to the LEAP cluster result.

Tomography recipes use the versioned
`neutron-imaging-gui/tomography-recipe-v1` JSON schema. They preserve local and
cluster path mappings, source summaries, loading/preparation settings, native
ROIs and crops, geometry, the promoted reconstruction trial, and full-volume
chunk/export parameters. Importing a recipe restores controls but never loads
data or starts processing automatically.

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

## Conda package

The repository includes a `noarch: python` recipe in `conda.recipe`. The GUI
depends on `neutron-imaging-tools >=0.2.5`, which is not currently available on
conda-forge. NIT is published in the `seanfayfar` Anaconda.org channel.

Build the GUI without automatically uploading:

```bash
conda build conda.recipe \
  --no-anaconda-upload \
  --override-channels \
  -c seanfayfar \
  -c conda-forge
```

The build ends by printing the `.conda` artifact path. Test that artifact in a
clean environment before upload:

```bash
conda create -n neutron-imaging-gui-test \
  --override-channels \
  -c local \
  -c seanfayfar \
  -c conda-forge \
  neutron-imaging-gui

conda run -n neutron-imaging-gui-test neutron-imaging-gui --help
```

Authenticate specifically with Anaconda.org and upload the artifact only after
the clean-environment test succeeds:

```bash
anaconda login --at anaconda.org
anaconda upload --at anaconda.org /path/to/neutron-imaging-gui-*.conda
```

After upload, users can install the GUI and its NIT dependency with:

```bash
conda create -n neutron-imaging-gui \
  --override-channels \
  -c seanfayfar \
  -c conda-forge \
  neutron-imaging-gui
```

TomoPy and LEAP are optional reconstruction backends and are intentionally not
installed by the base GUI package. Install TomoPy separately for CPU tomography
trials; install LEAP in a suitable GPU environment for LEAP reconstruction.

## Filename grouping

Sample files ending in an underscore and an integer are treated as repeated
exposures. For example, `sample_ab12cd34_0000.tif` and
`sample_ab12cd34_0001.tif` become one result named `sample_ab12cd34`. Files
without an integer suffix form their own result.
