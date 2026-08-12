# Changelog

Notable user-facing changes are recorded here. This project is still evolving,
so early releases summarize major capabilities rather than every interface
adjustment.

## Unreleased

- Add persistent Light, Dark, and System appearance modes under **View → Theme**.
- Apply theme colors to the Qt interface and all image, histogram, profile, and
  tomography plot axes, including live mode switching.

## 0.2.0 - 2026-08-12

- Add the staged radiography reduction workflow with reference merging, filtering,
  dose normalization, transmission, and attenuation processing.
- Add queued multiprocessing with configurable core usage and detailed progress.
- Add interactive image levels, gamma and color maps, measurement/profile tools,
  crop-aware single and batch exports, and processing metadata.
- Add the tomography preparation workspace, reconstruction trials, LEAP/TomoPy
  backend detection, and portable JSON recipes for cluster workflows.
- Add the `neutron-imaging-gui` command, automatic package version reporting, and
  Conda packaging for the `seanfayfar` channel.
