"""Application entry point."""

from __future__ import annotations

import argparse
import os
import sys

from qtpy import QtCore, QtGui, QtWidgets

from .main_window import MainWindow


def _configure_high_dpi() -> None:
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neutron imaging data reduction GUI")
    parser.add_argument("paths", nargs="*", help="Optional sample TIFF files or directories")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_high_dpi()
    os.environ.setdefault("QT_STYLE_OVERRIDE", "Fusion")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Neutron Imaging GUI")
    app.setOrganizationName("MIT-NRL")
    app.setStyle("Fusion")
    app.setWindowIcon(QtGui.QIcon())

    window = MainWindow(initial_sample_paths=args.paths)
    window.show()
    return int(app.exec_())

