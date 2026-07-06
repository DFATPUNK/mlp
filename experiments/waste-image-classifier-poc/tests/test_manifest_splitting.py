import unittest
from collections import Counter, defaultdict

from waste_poc.manifests import assign_grouped_stratified_splits
from waste_poc.utils import CLASS_NAMES


class ManifestSplittingTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for label in CLASS_NAMES:
            for idx in range(40):
                digest = f"{label}-sha-{idx // 2}" if idx in (0, 1) else f"{label}-sha-{idx}"
                rows.append({"image_id": f"{label}-{idx}", "label": label, "sha256": digest})
        return rows

    def test_all_classes_preserved_and_ratios_approximate(self):
        rows = assign_grouped_stratified_splits(self.make_rows(), {"train": 0.70, "validation": 0.15, "test": 0.15}, seed=42)
        self.assertEqual(set(row["label"] for row in rows), set(CLASS_NAMES))
        split_counts = Counter(row["split"] for row in rows)
        total = len(rows)
        self.assertAlmostEqual(split_counts["train"] / total, 0.70, delta=0.08)
        self.assertAlmostEqual(split_counts["validation"] / total, 0.15, delta=0.06)
        self.assertAlmostEqual(split_counts["test"] / total, 0.15, delta=0.06)

    def test_no_image_id_or_duplicate_group_crosses_splits(self):
        rows = assign_grouped_stratified_splits(self.make_rows(), {"train": 0.70, "validation": 0.15, "test": 0.15}, seed=7)
        self.assertEqual(len({row["image_id"] for row in rows}), len(rows))
        splits_by_sha = defaultdict(set)
        for row in rows:
            splits_by_sha[row["sha256"]].add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_sha.values()))


if __name__ == "__main__":
    unittest.main()
