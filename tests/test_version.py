from importlib.metadata import PackageNotFoundError, version
import unittest

import neutron_imaging_gui


class VersionTests(unittest.TestCase):
    def test_runtime_version_comes_from_package_metadata(self):
        try:
            installed_version = version("neutron-imaging-gui")
        except PackageNotFoundError:
            self.assertEqual(neutron_imaging_gui.__version__, "unknown")
        else:
            self.assertEqual(neutron_imaging_gui.__version__, installed_version)


if __name__ == "__main__":
    unittest.main()
