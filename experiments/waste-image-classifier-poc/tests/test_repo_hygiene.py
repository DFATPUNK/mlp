from __future__ import annotations

import unittest
from pathlib import Path


POC_ROOT = Path(__file__).resolve().parents[1]


class RepoHygieneTests(unittest.TestCase):
    def test_local_taco_label_mapping_is_gitignored(self):
        ignore_lines = {
            line.strip()
            for line in (POC_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

        self.assertIn("data/public_sources/taco_label_mapping.csv", ignore_lines)
        self.assertIn("data/public_sources/taco/review_outputs/", ignore_lines)
        self.assertIn("data/public_sources/taco/taco_license_resolution.csv", ignore_lines)
        self.assertIn("data/public_sources/taco/taco_download_plan.csv", ignore_lines)
        self.assertTrue((POC_ROOT / "data/public_sources/taco_label_mapping.template.csv").is_file())


if __name__ == "__main__":
    unittest.main()
