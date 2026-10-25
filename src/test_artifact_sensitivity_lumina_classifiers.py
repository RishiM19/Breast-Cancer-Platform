"""Isolated geometry/control safety tests; synthetic fixtures are not research results."""
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from artifact_sensitivity_lumina_classifiers import (
    ROOT, OUTPUT, MANIFEST, PROFILES, ValidationAccessGuard, choose_cases,
    define_regions, perturb, read_pair, transform_mask,
)


class ReliabilityGeometryTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(7).integers(0, 256, (80, 110, 3), dtype=np.uint8)
        self.lesion = np.zeros((80, 110), bool)
        self.lesion[10:25, 0:20] = True  # deliberately crosses top and left ROIs
        self.row = pd.Series(dict(image_id="fixture", image_path="unused"))

    def regions(self, seed=42):
        return define_regions(self.rgb, self.lesion, self.row, {}, seed, 3, .05, .10)[0]

    def test_exact_area_controls_and_lesion_exclusion(self):
        variants = self.regions()
        sizes = {v["region"]: v["mask"].sum() for v in variants if v["region"] != "random_control"}
        for v in variants:
            if v["region"] != "lesion":
                self.assertFalse((v["mask"] & self.lesion).any())
            if v["region"] == "random_control":
                self.assertEqual(v["mask"].sum(), sizes[v["matched_region"]])
                self.assertEqual(v["control_overlap_fraction"], 0)

    def test_random_controls_repeat_and_change_with_seed(self):
        a, b, c = self.regions(), self.regions(), self.regions(43)
        self.assertTrue(all(np.array_equal(x["mask"], y["mask"]) for x,y in zip(a,b)))
        self.assertTrue(any(not np.array_equal(x["mask"], y["mask"]) for x,y in zip(a,c)))

    def test_fills_preserve_geometry_and_unselected_pixels(self):
        for v in self.regions():
            for fill in ("local_median", "blur"):
                changed, _ = perturb(self.rgb, v["mask"], self.lesion, fill)
                self.assertEqual(changed.shape, self.rgb.shape)
                np.testing.assert_array_equal(changed[~v["mask"]], self.rgb[~v["mask"]])
                if v["region"] != "lesion":
                    np.testing.assert_array_equal(changed[self.lesion], self.rgb[self.lesion])

    def test_nearest_projection_never_touches_padding_or_lesion(self):
        for profile in PROFILES.values():
            valid = transform_mask(np.ones_like(self.lesion), profile)
            lesion = transform_mask(self.lesion, profile)
            for v in self.regions():
                mapped = transform_mask(v["mask"], profile)
                self.assertFalse((mapped & ~valid).any())
                if v["region"] != "lesion":
                    self.assertFalse((mapped & lesion).any())

    def test_no_text_invented(self):
        variants, skips = define_regions(self.rgb, self.lesion, self.row, {}, 42, 3, .05, .10)
        self.assertNotIn("text", [v["region"] for v in variants])
        self.assertEqual(skips[0]["region"], "text")

    def test_bad_geometry_and_nonbinary_masks_rejected(self):
        row = pd.Series(dict(image_path="fixture.png", mask_path="fixture_mask.png"))
        fixtures = [(np.zeros((40, 55), np.uint8), "geometry mismatch"),
                    (np.ones((80, 110), np.uint8)*128, "binary"),
                    (np.zeros((80, 110), np.uint8), "Empty")]
        for mask, error in fixtures:
            with patch("artifact_sensitivity_lumina_classifiers.Image.open", side_effect=[
                Image.fromarray(self.rgb), Image.fromarray(mask)
            ]):
                with self.assertRaisesRegex(ValueError, error):
                    read_pair(row)

    def test_data_guard_blocks_unlisted_paths_and_dataset_writes(self):
        guard = ValidationAccessGuard(OUTPUT)
        # Invoke the policy directly; never open a held-out manifest or image.
        with self.assertRaises(PermissionError):
            guard("open", (str(ROOT / "data/unlisted_fixture.png"), "r", 0))
        with self.assertRaises(PermissionError):
            guard("open", (str(MANIFEST), "w", 0))
        guard("open", (str(MANIFEST), "r", 0))

    def test_stratification_is_reproducible_and_case_unique(self):
        df = pd.DataFrame([dict(case_id=str(c), image_id=f"{c}_{i}", birads=2+c%4)
                           for c in range(40) for i in range(2)])
        a, b = choose_cases(df, 24, 42), choose_cases(df, 24, 42)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.case_id.nunique(), 24)
        self.assertEqual(a.birads.value_counts().tolist(), [6]*4)
        self.assertEqual(len(choose_cases(df, 0, 42)), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
