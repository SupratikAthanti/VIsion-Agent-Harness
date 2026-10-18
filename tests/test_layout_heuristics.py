import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

import layout_heuristics as lh  # noqa: E402


def word(text, cx, cy):
    """Synthetic OCR word dict with the same keys ocr_engine produces (only the ones used here)."""
    return {"text": text, "center": (cx, cy), "box": (cx - 0.01, cy - 0.01, cx + 0.01, cy + 0.01),
            "conf": 90.0, "line": ("1", "1", "1"), "bbox_px": (0, 0, 1, 1)}


class HeuristicsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.img = Path(cls.tmpdir.name) / "screen.png"
        Image.new("RGB", (800, 600), "white").save(cls.img)
        cls.missing = Path(cls.tmpdir.name) / "missing.png"

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class Priors(HeuristicsBase):
    def test_search_box_top_centre_prior(self):
        loc = lh.predict_search_box_location(self.img)
        self.assertAlmostEqual(loc["x_pct"], 0.5)
        self.assertLess(loc["y_pct"], 0.15)
        self.assertEqual(loc["source"], "heuristic")

    def test_search_box_uses_placeholder_word(self):
        loc = lh.predict_search_box_location(self.img, words=[word("Search", 0.3, 0.07), word("Home", 0.1, 0.5)])
        self.assertEqual((loc["x_pct"], loc["y_pct"]), (0.3, 0.07))
        self.assertEqual(loc["source"], "ocr")

    def test_close_top_right(self):
        loc = lh.predict_close_button_location(self.img)
        self.assertGreater(loc["x_pct"], 0.9)
        self.assertLess(loc["y_pct"], 0.1)

    def test_menu_top_left(self):
        loc = lh.predict_menu_button_location(self.img)
        self.assertLess(loc["x_pct"], 0.1)
        self.assertLess(loc["y_pct"], 0.1)

    def test_navigation_layouts(self):
        top = lh.predict_navigation_location(self.img, "top_bar")
        side = lh.predict_navigation_location(self.img, "sidebar")
        self.assertLess(top["y_pct"], 0.15)
        self.assertLess(side["x_pct"], 0.2)
        with self.assertRaises(ValueError):
            lh.predict_navigation_location(self.img, "bottom")

    def test_submit_right_of_field(self):
        field = {"x_pct": 0.4, "y_pct": 0.1, "width_pct": 0.2}
        loc = lh.predict_submit_button_location(self.img, field)
        self.assertGreater(loc["x_pct"], field["x_pct"])
        self.assertAlmostEqual(loc["y_pct"], 0.1)

    def test_submit_below_field_at_right_edge(self):
        field = {"x_pct": 0.9, "y_pct": 0.5, "width_pct": 0.2}
        loc = lh.predict_submit_button_location(self.img, field)
        self.assertGreater(loc["y_pct"], 0.5)

    def test_submit_prefers_labelled_word(self):
        field = {"x_pct": 0.4, "y_pct": 0.5}
        loc = lh.predict_submit_button_location(self.img, field, words=[word("Go", 0.45, 0.52)])
        self.assertEqual((loc["x_pct"], loc["y_pct"]), (0.45, 0.52))
        self.assertEqual(loc["source"], "ocr")

    def test_unreadable_image_returns_none(self):
        self.assertIsNone(lh.predict_search_box_location(self.missing))
        self.assertIsNone(lh.predict_close_button_location(self.missing))


class SpatialReasoning(HeuristicsBase):
    def test_find_nearby_picks_nearest_labelled_submit(self):
        field = {"x_pct": 0.5, "y_pct": 0.5}
        words = [word("Submit", 0.9, 0.9), word("Submit", 0.55, 0.52)]
        found = lh.find_nearby_element(self.img, field, "submit", max_distance=0.15, words=words)
        self.assertEqual((found["x_pct"], found["y_pct"]), (0.55, 0.52))

    def test_find_nearby_respects_max_distance(self):
        field = {"x_pct": 0.5, "y_pct": 0.5}
        far = [word("Submit", 0.95, 0.95)]
        self.assertIsNone(lh.find_nearby_element(self.img, field, "submit", max_distance=0.1, words=far))

    def test_find_nearby_falls_back_to_prior(self):
        field = {"x_pct": 0.4, "y_pct": 0.3, "width_pct": 0.2}
        found = lh.find_nearby_element(self.img, field, "submit", max_distance=0.2)
        self.assertIsNotNone(found)
        self.assertEqual(found["source"], "heuristic")

    def test_find_nearby_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            lh.find_nearby_element(self.img, {"x_pct": 0.5, "y_pct": 0.5}, "banana")

    def test_submit_near_field_wrapper(self):
        field = {"x_pct": 0.4, "y_pct": 0.3, "width_pct": 0.2}
        self.assertIsNotNone(lh.find_submit_button_near_field(self.img, field))

    def test_menu_items_below_sorted_and_filtered(self):
        menu = {"x_pct": 0.1, "y_pct": 0.1}
        words = [
            word("Profile", 0.12, 0.25),
            word("Settings", 0.1, 0.18),
            word("Above", 0.1, 0.05),        # above the menu: excluded
            word("Far", 0.1, 0.9),           # too far down: excluded
            word("Elsewhere", 0.7, 0.2),     # outside horizontal tolerance: excluded
        ]
        items = lh.find_menu_items_below(self.img, menu, words=words)
        self.assertEqual([i["text"] for i in items], ["Settings", "Profile"])

    def test_menu_items_need_words(self):
        self.assertEqual(lh.find_menu_items_below(self.img, {"x_pct": 0.1, "y_pct": 0.1}), [])

    def test_tab_content_below_top_tabs(self):
        area = lh.find_content_area_for_tab(self.img, {"x_pct": 0.2, "y_pct": 0.05})
        self.assertLess(area["bbox"]["y0"], 0.15)
        self.assertEqual(area["bbox"]["y1"], 0.95)

    def test_tab_content_right_of_sidebar_tabs(self):
        area = lh.find_content_area_for_tab(self.img, {"x_pct": 0.05, "y_pct": 0.5})
        self.assertGreater(area["bbox"]["x0"], 0.05)

    def test_tab_in_middle_has_no_guess(self):
        self.assertIsNone(lh.find_content_area_for_tab(self.img, {"x_pct": 0.5, "y_pct": 0.5}))


if __name__ == "__main__":
    unittest.main()
