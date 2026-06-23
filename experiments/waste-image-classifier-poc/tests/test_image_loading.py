import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from waste_poc.images import load_rgb_image


@unittest.skipIf(Image is None, "Pillow is required for EXIF image tests")
class ImageLoadingTests(unittest.TestCase):
    def test_exif_orientation_is_transposed_to_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotated.jpg"
            image = Image.new("RGB", (20, 10), "red")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            loaded = load_rgb_image(path)
            self.assertEqual(loaded.mode, "RGB")
            self.assertEqual(loaded.size, (10, 20))

    def test_normal_image_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "normal.jpg"
            Image.new("RGB", (13, 7), "blue").save(path)
            loaded = load_rgb_image(path)
            self.assertEqual(loaded.mode, "RGB")
            self.assertEqual(loaded.size, (13, 7))

    def test_missing_or_invalid_file_has_path_in_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.jpg"
            with self.assertRaisesRegex(Exception, "missing.jpg"):
                load_rgb_image(missing)
            invalid = Path(tmp) / "invalid.jpg"
            invalid.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid.jpg"):
                load_rgb_image(invalid)


if __name__ == "__main__":
    unittest.main()
