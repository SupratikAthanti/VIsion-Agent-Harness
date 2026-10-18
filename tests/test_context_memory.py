import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

import context_memory as cm  # noqa: E402


class ContextTestCase(unittest.TestCase):
    def setUp(self):
        cm.clear_context()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self):
        cm.clear_context()
        self.tmpdir.cleanup()

    def solid(self, name, colour):
        path = self.dir / name
        Image.new("RGB", (400, 300), colour).save(path)
        return path


class History(ContextTestCase):
    def test_recent_actions_capped_and_ordered(self):
        for i in range(15):
            cm.update_context({"action": "CLICK", "target": f"t{i}"})
        recent = cm.get_recent_actions(5)
        self.assertEqual([a["target"] for a in recent], ["t10", "t11", "t12", "t13", "t14"])
        self.assertEqual(len(cm.get_context()["last_actions"]), 10)

    def test_zero_count_returns_empty(self):
        cm.update_context({"action": "CLICK", "target": "x"})
        self.assertEqual(cm.get_recent_actions(0), [])

    def test_type_records_form_state(self):
        cm.update_context({"action": "TYPE", "target": "search box", "params": {"text": "cats"}})
        cm.update_context({"action": "TYPE", "target": "search box", "params": {"text": "dogs"}})
        self.assertEqual(cm.get_form_state(), {"search box": "dogs"})

    def test_location_recorded_as_tuple(self):
        cm.update_context({"action": "CLICK", "target": "ok", "location": {"x_pct": 0.5, "y_pct": 0.9}})
        self.assertEqual(cm.get_recent_actions(1)[0]["location"], (0.5, 0.9))
        self.assertEqual(cm.get_known_location("ok"), (0.5, 0.9))

    def test_returned_copies_do_not_alias_state(self):
        cm.update_context({"action": "TYPE", "target": "a", "params": {"text": "1"}})
        cm.get_form_state()["a"] = "tampered"
        self.assertEqual(cm.get_form_state(), {"a": "1"})


class Navigation(ContextTestCase):
    def test_navigate_sets_page_and_clears_form(self):
        cm.update_context({"action": "TYPE", "target": "search box", "params": {"text": "cats"}})
        cm.update_context({"action": "CLICK", "target": "go", "location": (0.5, 0.5)})
        cm.update_context({"action": "NAVIGATE", "params": {"url": "https://youtube.com/"}})
        ctx = cm.get_context()
        self.assertEqual(ctx["current_page"], "youtube.com")
        self.assertEqual(ctx["form_state"], {})
        self.assertEqual(ctx["known_locations"], {})

    def test_window_title_change_clears_page_state(self):
        cm.set_current_page("Inbox")
        cm.update_context({"action": "TYPE", "target": "to", "params": {"text": "bob"}})
        cm.set_current_page("Inbox")  # same name: nothing cleared
        self.assertEqual(cm.get_form_state(), {"to": "bob"})
        cm.set_current_page("Compose")
        self.assertEqual(cm.get_form_state(), {})

    def test_clear_context_resets_everything(self):
        cm.set_current_page("Page")
        cm.update_context({"action": "TYPE", "target": "a", "params": {"text": "1"}})
        cm.clear_context()
        ctx = cm.get_context()
        self.assertIsNone(ctx["current_page"])
        self.assertEqual(ctx["last_actions"], [])
        self.assertEqual(ctx["form_state"], {})


class PageChange(ContextTestCase):
    def test_first_frame_is_not_a_change(self):
        self.assertFalse(cm.detect_page_change(self.solid("a.png", (255, 255, 255))))

    def test_identical_frame_is_not_a_change(self):
        cm.detect_page_change(self.solid("a.png", (255, 255, 255)))
        self.assertFalse(cm.detect_page_change(self.solid("b.png", (255, 255, 255))))

    def test_different_frame_is_a_change_and_clears_form(self):
        cm.detect_page_change(self.solid("a.png", (255, 255, 255)))
        cm.update_context({"action": "TYPE", "target": "box", "params": {"text": "x"}})
        self.assertTrue(cm.detect_page_change(self.solid("b.png", (0, 0, 0))))
        self.assertEqual(cm.get_form_state(), {})

    def test_small_local_change_is_not_a_page_change(self):
        base = Image.new("RGB", (400, 300), (255, 255, 255))
        first = self.dir / "first.png"
        base.save(first)
        draw = ImageDraw.Draw(base)
        draw.rectangle((10, 10, 40, 25), fill=(0, 0, 0))  # a few typed characters
        second = self.dir / "second.png"
        base.save(second)
        cm.detect_page_change(first)
        self.assertFalse(cm.detect_page_change(second))

    def test_unreadable_image_is_ignored(self):
        self.assertFalse(cm.detect_page_change(self.dir / "nope.png"))


if __name__ == "__main__":
    unittest.main()
