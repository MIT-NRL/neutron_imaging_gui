"""Application-wide light, dark, and desktop-following theme support."""

from __future__ import annotations

import os
import subprocess
import sys

import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets


SETTINGS_ORGANIZATION = "MIT-NRL"
SETTINGS_APPLICATION = "Neutron Imaging GUI"
THEME_MODE_SETTINGS_KEY = "appearance/theme_mode"
THEME_MODES = ("light", "dark", "system")


def settings() -> QtCore.QSettings:
    return QtCore.QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


def saved_theme_mode() -> str:
    mode = str(settings().value(THEME_MODE_SETTINGS_KEY, "system")).strip().lower()
    return mode if mode in THEME_MODES else "system"


def desktop_prefers_dark() -> bool:
    override = os.environ.get("NEUTRON_IMAGING_GUI_THEME", "").strip().lower()
    if override in {"dark", "light"}:
        return override == "dark"

    app = QtWidgets.QApplication.instance()
    hints = app.styleHints() if app is not None else None
    color_scheme = getattr(hints, "colorScheme", None)
    if callable(color_scheme):
        scheme = color_scheme()
        dark = getattr(QtCore.Qt, "ColorScheme", None)
        if dark is not None:
            return scheme == dark.Dark

    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            return result.returncode == 0 and "dark" in result.stdout.lower()
        except Exception:
            return False

    if sys.platform.startswith("linux"):
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            return "dark" in result.stdout.lower()
        except Exception:
            return False
    return False


def build_dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    colors = {
        QtGui.QPalette.Window: QtGui.QColor(45, 45, 45),
        QtGui.QPalette.WindowText: QtGui.QColor(240, 240, 240),
        QtGui.QPalette.Base: QtGui.QColor(30, 30, 30),
        QtGui.QPalette.AlternateBase: QtGui.QColor(45, 45, 45),
        QtGui.QPalette.ToolTipBase: QtGui.QColor(45, 45, 45),
        QtGui.QPalette.ToolTipText: QtGui.QColor(240, 240, 240),
        QtGui.QPalette.Text: QtGui.QColor(240, 240, 240),
        QtGui.QPalette.Button: QtGui.QColor(53, 53, 53),
        QtGui.QPalette.ButtonText: QtGui.QColor(240, 240, 240),
        QtGui.QPalette.BrightText: QtGui.QColor(255, 90, 90),
        QtGui.QPalette.Link: QtGui.QColor(88, 166, 255),
        QtGui.QPalette.Highlight: QtGui.QColor(47, 111, 159),
        QtGui.QPalette.HighlightedText: QtGui.QColor(255, 255, 255),
        QtGui.QPalette.Mid: QtGui.QColor(125, 125, 125),
    }
    for role, color in colors.items():
        palette.setColor(role, color)
    for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.WindowText):
        palette.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(130, 130, 130))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Base, QtGui.QColor(38, 38, 38))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    return palette


def effective_theme(mode: str) -> str:
    return "dark" if mode == "dark" or (mode == "system" and desktop_prefers_dark()) else "light"


def _sync_pyqtgraph(palette: QtGui.QPalette) -> None:
    background = palette.color(QtGui.QPalette.Window)
    foreground = palette.color(QtGui.QPalette.WindowText)
    pg.setConfigOption("background", background.getRgb()[:3])
    pg.setConfigOption("foreground", foreground.getRgb()[:3])


def refresh_pyqtgraph_widgets(root: QtWidgets.QWidget, palette: QtGui.QPalette) -> None:
    background = palette.color(QtGui.QPalette.Window)
    foreground = palette.color(QtGui.QPalette.WindowText)
    foreground_pen = pg.mkPen(foreground)
    for graphics in root.findChildren(pg.GraphicsView):
        try:
            graphics.setBackground(background)
        except Exception:
            pass

    plot_items = [plot.getPlotItem() for plot in root.findChildren(pg.PlotWidget)]
    # ImageView owns a standalone PlotItem rather than a PlotWidget, so its
    # image-axis tick labels need to be collected explicitly.
    plot_items.extend(view.getView() for view in root.findChildren(pg.ImageView))
    seen = set()
    for item in plot_items:
        if id(item) in seen:
            continue
        seen.add(id(item))
        for axis_name in ("left", "right", "top", "bottom"):
            try:
                axis = item.getAxis(axis_name)
                axis.setPen(foreground_pen)
                axis.setTextPen(foreground_pen)
            except Exception:
                pass
        item.update()


def apply_theme(mode: str, *, persist: bool = False, root=None) -> str:
    mode = str(mode).strip().lower()
    if mode not in THEME_MODES:
        raise ValueError(f"Unknown theme mode: {mode!r}")
    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("A QApplication is required before applying a theme.")
    palette = (
        build_dark_palette()
        if effective_theme(mode) == "dark"
        else app.style().standardPalette()
    )
    app.setPalette(palette)
    app.setProperty("neutronImagingThemeMode", mode)
    _sync_pyqtgraph(palette)
    if persist:
        app_settings = settings()
        app_settings.setValue(THEME_MODE_SETTINGS_KEY, mode)
        app_settings.sync()
    if root is not None:
        refresh_pyqtgraph_widgets(root, palette)
        root.update()
    return mode
