import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from neutron_imaging_gui.exporting import (
    build_export_manifest,
    crop_image,
    export_reduction_batch,
    image_export_metadata,
    read_tiff_metadata,
    write_json,
    write_png,
    write_tiff,
)
from neutron_imaging_gui.processing import (
    ImageProducts,
    ReductionConfig,
    ReductionResult,
)


class ExportingTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    def _source_tiff(self, name, value, exposure):
        path = self.root / name
        tifffile.imwrite(
            path,
            np.full((6, 8), value, dtype=np.float32),
            description=json.dumps({"AcquireTime": exposure, "SampleName": name}),
            metadata=None,
        )
        return str(path)

    def _result(self):
        white_path = self._source_tiff("white.tif", 100.0, "1.5 s")
        dark_path = self._source_tiff("dark.tif", 0.0, "250 ms")
        sample_path = self._source_tiff("sample_0000.tif", 50.0, "2 s")
        config = ReductionConfig(
            white_files=(white_path,),
            dark_files=(dark_path,),
            sample_files=(sample_path,),
            merge_method="median",
            gamma_filter=False,
        )
        combined = np.full((6, 8), 50.0, dtype=np.float32)
        transmission = np.full((6, 8), 0.5, dtype=np.float32)
        attenuation = np.full((6, 8), -np.log(0.5), dtype=np.float32)
        product = ImageProducts(
            name="sample",
            files=(sample_path,),
            combined=combined,
            transmission=transmission,
            attenuation=attenuation,
        )
        return ReductionResult(
            white=np.full((6, 8), 100.0, dtype=np.float32),
            dark=np.zeros((6, 8), dtype=np.float32),
            products={"sample": product},
            config=config,
            processing_elapsed_seconds=3.25,
            processing_started_utc="2026-08-12T12:00:00+00:00",
            processing_finished_utc="2026-08-12T12:00:03.25+00:00",
        )

    def test_reads_exposure_and_builds_provenance_manifest(self):
        result = self._result()
        source = read_tiff_metadata(result.config.dark_files[0])
        self.assertAlmostEqual(source["exposure_seconds"], 0.25)
        self.assertIn("ImageDescription", source["tiff_tags"])

        manifest = build_export_manifest(result)
        self.assertEqual(manifest["schema"], "neutron-imaging-gui/export-v1")
        self.assertEqual(manifest["timing"]["known_exposure_count"], 3)
        self.assertAlmostEqual(manifest["timing"]["total_known_exposure_seconds"], 3.75)
        self.assertEqual(manifest["timing"]["processing_elapsed_seconds"], 3.25)
        self.assertEqual(manifest["products"]["sample"]["source_filenames"], ["sample_0000.tif"])
        self.assertTrue(manifest["software"]["neutron_imaging_tools"])

    def test_crop_tiff_metadata_json_and_png_exports(self):
        result = self._result()
        cropped = crop_image(result.products["sample"].transmission, (2, 1, 4, 3))
        self.assertEqual(cropped.shape, (3, 4))
        manifest = build_export_manifest(result)
        metadata = image_export_metadata(
            manifest,
            "sample_transmission",
            cropped,
            crop_bounds=(2, 1, 4, 3),
        )

        tiff_path = self.root / "transmission.tif"
        write_tiff(tiff_path, cropped, metadata)
        np.testing.assert_allclose(tifffile.imread(tiff_path), cropped)
        with tifffile.TiffFile(tiff_path) as tif:
            embedded = json.loads(tif.pages[0].description)
            self.assertEqual(embedded["product"], "sample_transmission")
            self.assertEqual(embedded["crop"], [2, 1, 4, 3])
            self.assertIn("neutron-imaging-tools", tif.pages[0].tags["Software"].value)

        json_path = self.root / "transmission.json"
        write_json(json_path, manifest)
        self.assertEqual(json.loads(json_path.read_text())["schema"], manifest["schema"])

        image_png = self.root / "image.png"
        styled_png = self.root / "styled.png"
        write_png(image_png, cropped, cmap="viridis", levels=(0.0, 1.0))
        write_png(
            styled_png,
            cropped,
            cmap="magma",
            levels=(0.0, 1.0),
            styled=True,
            colorbar=True,
            title="Transmission",
            dpi=100,
        )
        self.assertGreater(image_png.stat().st_size, 0)
        self.assertGreater(styled_png.stat().st_size, image_png.stat().st_size)

    def test_batch_export_subfolders_manifest_and_overwrite_protection(self):
        result = self._result()
        output = self.root / "batch"
        exported = export_reduction_batch(
            result,
            output,
            categories=("background", "transmission", "attenuation"),
            use_subfolders=True,
            embed_tiff_metadata=True,
            companion_json=True,
        )
        expected = {
            output / "background" / "reference_white.tif",
            output / "background" / "reference_dark.tif",
            output / "transmission" / "sample_transmission.tif",
            output / "attenuation" / "sample_attenuation.tif",
        }
        self.assertEqual(set(exported["files"]), expected)
        self.assertTrue(all(path.exists() for path in expected))
        manifest = json.loads((output / "metadata.json").read_text())
        self.assertEqual(
            manifest["export"]["categories"],
            ["background", "transmission", "attenuation"],
        )
        with tifffile.TiffFile(output / "transmission" / "sample_transmission.tif") as tif:
            embedded = json.loads(tif.pages[0].description)
        self.assertEqual(embedded["category"], "transmission")
        self.assertEqual(
            embedded["source_inputs"]["sample"][0]["filename"],
            "sample_0000.tif",
        )

        with self.assertRaises(FileExistsError):
            export_reduction_batch(
                result,
                output,
                categories=("background",),
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
