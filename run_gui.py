"""Run the application directly from a source checkout."""

from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_DIR))

from neutron_imaging_gui.__main__ import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

