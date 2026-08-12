"""Neutron imaging reduction GUI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("neutron-imaging-gui")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
