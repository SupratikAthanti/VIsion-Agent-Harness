import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vision_cli as vc  # noqa: E402
from yolo_detector import YOLODetector  # noqa: E402


class DetectionLayerTests(unittest.TestCase):
    def test_template_matches_when_query_and_template_spacing_differ(self):
        self.assertEqual(vc.find_trained_template("click the new tab icon", ["new tab icon"]), "new tab icon")
        self.assertEqual(vc.find_trained_template("click the new tab button", ["new tab icon"]), "new tab icon")
        self.assertEqual(vc.find_trained_template("new tab", ["new tab icon"]), "new tab icon")
        self.assertIsNone(vc.find_trained_template("click tab", ["new tab icon"]))

    def test_new_tab_icon_uses_plus_glyph_not_label(self):
        image = Path(__file__).resolve().parent.parent / "images" / "1.jpeg"
        if not image.exists():
            self.skipTest("project screenshot fixture is unavailable")
        result = vc._ocr_symbol_candidate(image, "new tab", kind="icon")
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "+")
        self.assertAlmostEqual(result["x_pct"] * 100, 25.56, delta=0.5)
        self.assertAlmostEqual(result["y_pct"] * 100, 2.22, delta=0.5)

    def test_close_symbol_targets_top_right_x(self):
        words = [
            {"text": "x", "center": (0.2, 0.03), "conf": 95},
            {"text": "×", "center": (0.96, 0.03), "conf": 90},
        ]
        with patch.object(vc, "extract_text_with_coordinates", return_value=words):
            result = vc._ocr_symbol_candidate(Path("unused.png"), "close", kind="button")
        self.assertEqual(result["label"], "×")
        self.assertEqual(result["x_pct"], 0.96)

    def test_menu_symbol_targets_hamburger_near_top_left(self):
        words = [
            {"text": "≡", "center": (0.04, 0.04), "conf": 90},
            {"text": "≡", "center": (0.5, 0.04), "conf": 90},
        ]
        with patch.object(vc, "extract_text_with_coordinates", return_value=words):
            result = vc._ocr_symbol_candidate(Path("unused.png"), "menu", kind="button")
        self.assertEqual(result["x_pct"], 0.04)

    def test_yolo_position_filter_prefers_confidence_before_size(self):
        weak_small = {"bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
                      "center_pct": {"x": 20, "y": 5}, "confidence": 0.22}
        strong_icon = {"bbox": {"x": 40, "y": 10, "width": 80, "height": 80},
                       "center_pct": {"x": 50, "y": 5}, "confidence": 0.75}
        detector = YOLODetector.__new__(YOLODetector)
        with patch.object(detector, "detect", return_value=[weak_small, strong_icon]), patch("yolo_detector.Image.open") as open_image:
            open_image.return_value.size = (1000, 600)
            result = detector.find_icon_by_position("unused.png", icon_type="top_bar")
        self.assertIs(result, strong_icon)

    def test_low_confidence_yolo_is_rejected(self):
        detection = {"confidence": 0.22, "bbox": {"x": 20, "y": 20, "width": 30, "height": 30}}
        with patch("builtins.print") as output:
            self.assertFalse(vc._reliable_yolo_detection(detection, Path("unused.png")))
        self.assertIn("22%", " ".join(str(call) for call in output.call_args_list))

    def test_oversized_yolo_box_is_rejected(self):
        image = Path(__file__).resolve().parent.parent / "images" / "1.jpeg"
        detection = {"confidence": 0.9, "bbox": {"x": 0, "y": 0, "width": 1200, "height": 400}}
        self.assertFalse(vc._reliable_yolo_detection(detection, image))

    def test_low_confidence_yolo_does_not_fall_through_to_layout_guess(self):
        def reject_detection(*args, **kwargs):
            vc._yolo_rejected_unreliable = True
            return None

        with patch.object(vc, "HAS_ATSPI", False), patch.object(vc, "HAS_TEMPLATES", False), \
             patch.object(vc, "_ocr_symbol_candidate", return_value=None), \
             patch.object(vc, "_ocr_candidates", return_value=[]), \
             patch.object(vc, "_yolo_candidate", side_effect=reject_detection), \
             patch.object(vc, "_heuristic_candidate") as heuristic:
            candidates, source = vc.locate_target("close", Path("unused.png"))
        self.assertEqual((candidates, source), ([], None))
        heuristic.assert_not_called()

    def test_icon_ocr_does_not_return_text_label(self):
        with patch.object(vc, "HAS_ATSPI", False), patch.object(vc, "HAS_TEMPLATES", False), \
             patch.object(vc, "_ocr_symbol_candidate", return_value=None), \
             patch.object(vc, "_ocr_candidates", return_value=[{"label": "New Tab"}]), \
             patch.object(vc, "_yolo_candidate", return_value=None), \
             patch.object(vc, "_heuristic_candidate") as heuristic, \
             patch.object(vc.context_memory, "get_known_location", return_value=(0.2, 0.2)):
            candidates, source = vc.locate_target("new tab", Path("unused.png"), kind="icon")
        self.assertEqual((candidates, source), ([], None))
        heuristic.assert_not_called()

    def test_optional_atspi_failure_falls_back(self):
        with patch.object(vc, "HAS_ATSPI", True), patch.object(
            vc.atspi_bridge, "find_element_by_name", side_effect=RuntimeError("no bus")
        ):
            self.assertIsNone(vc._atspi_location("Gmail", Path("unused.png")))

    def test_only_known_static_targets_use_templates(self):
        self.assertTrue(vc._is_static_template_target("Where is the AI mode button?"))
        self.assertTrue(vc._is_static_template_target("click Gmail button"))
        self.assertTrue(vc._is_static_template_target("YouTube logo"))
        self.assertFalse(vc._is_static_template_target("new tab"))
        self.assertFalse(vc._is_static_template_target("random toolbar button"))

    def test_dynamic_targets_skip_template_matching(self):
        for query in ("new tab", "close tab", "search bar", "menu item", "unknown logo"):
            with self.subTest(query=query), patch.object(vc, "list_templates") as templates:
                self.assertFalse(vc.run_template_query(query, Path("unused.png"), Path(".")))
                templates.assert_not_called()

    def test_cleanup_removes_dynamic_and_unapproved_templates(self):
        with (
            patch.object(vc, "HAS_TEMPLATES", True),
            patch.object(vc, "list_templates", return_value=["x", "new tab", "gmail button"]),
            patch.object(vc, "delete_template") as delete,
        ):
            vc._clean_dynamic_templates()
        self.assertEqual(delete.call_args_list, [call("x"), call("new tab")])

    def test_hybrid_pipeline_checks_accessibility_ocr_before_static_template(self):
        calls = []
        with (
            patch.object(vc, "_clean_dynamic_templates"),
            patch.object(vc, "_atspi_location", side_effect=lambda *args: calls.append("atspi") or None),
            patch.object(vc, "_ocr_symbol_candidate", side_effect=lambda *args: calls.append("spatial OCR") or None),
            patch.object(vc, "find_text_element", side_effect=lambda *args: calls.append("OCR") or None),
            patch.object(vc, "run_template_query", side_effect=lambda *args: calls.append("template") or True),
            patch.object(vc, "auto_mark_target"),
        ):
            vc.run_hybrid_query("click Gmail button", Path("unused.png"), Path("."))
        self.assertEqual(calls, ["atspi", "OCR", "spatial OCR", "template"])

    def test_hybrid_dynamic_icon_uses_spatial_ocr_before_router(self):
        with (
            patch.object(vc, "_clean_dynamic_templates"),
            patch.object(vc, "_atspi_location") as atspi,
            patch.object(vc, "find_text_element", return_value=None) as text_ocr,
            patch.object(vc, "_ocr_symbol_candidate", return_value={
                "x_pct": 0.25, "y_pct": 0.03, "label": "+"}) as spatial,
            patch.object(vc, "run_template_query") as template,
            patch.object(vc, "route_task") as router,
            patch.object(vc, "auto_mark_target"),
        ):
            vc.run_hybrid_query("click the new tab icon", Path("unused.png"), Path("."))
        atspi.assert_not_called()
        text_ocr.assert_called_once()
        spatial.assert_called_once()
        template.assert_not_called()
        router.assert_not_called()

    def test_dynamic_target_is_not_learned_as_template(self):
        with patch.object(vc, "save_template") as save:
            vc._maybe_learn("search bar", {"x_pct": 0.5, "y_pct": 0.1},
                            Path("unused.png"), "ocr", learn=True, interactive=False)
        save.assert_not_called()

    def test_unknown_target_is_not_learned_as_template(self):
        with patch.object(vc, "save_template") as save:
            vc._maybe_learn("moving toolbar item", {"x_pct": 0.5, "y_pct": 0.1},
                            Path("unused.png"), "ocr", learn=True, interactive=False)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
