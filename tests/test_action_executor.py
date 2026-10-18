import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import action_executor as ax  # noqa: E402

SCREEN = (1920, 1080)


def fake_run(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class ExecutorBase(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"DISPLAY": ":0"})
        self.env.start()
        self.which = mock.patch("action_executor.shutil.which", return_value="/usr/bin/xdotool")
        self.which.start()
        self.sleep = mock.patch("action_executor.time.sleep")
        self.mock_sleep = self.sleep.start()
        self.run = mock.patch("action_executor.subprocess.run", side_effect=self._dispatch)
        self.mock_run = self.run.start()
        self.calls = []
        self.screen_result = fake_run(stdout="1920 1080\n")

    def tearDown(self):
        for p in (self.env, self.which, self.sleep, self.run):
            p.stop()

    def _dispatch(self, args, **kwargs):
        self.calls.append(args)
        if args[1] == "getdisplaygeometry":
            return self.screen_result
        return fake_run()

    def xdotool_calls(self):
        return [c[1:] for c in self.calls if c[1] != "getdisplaygeometry"]


class Coordinates(ExecutorBase):
    def test_to_absolute_scales_fractions(self):
        self.assertEqual(ax.to_absolute(0.5, 0.5, SCREEN), (960, 540))

    def test_to_absolute_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            ax.to_absolute(1.5, 0.2, SCREEN)

    def test_screen_size_read_from_xdotool(self):
        self.assertEqual(ax.get_screen_size(), SCREEN)


class Clicks(ExecutorBase):
    def test_click_at_sends_absolute_move_and_click(self):
        self.assertTrue(ax.click_at(0.25, 0.5))
        self.assertIn(["mousemove", "--sync", "480", "540", "click", "1"], self.xdotool_calls())

    def test_click_with_offset(self):
        ax.click_at(0.5, 0.5, offset_px=(6, 0))
        self.assertIn(["mousemove", "--sync", "966", "540", "click", "1"], self.xdotool_calls())

    def test_no_display_fails_without_calling_xdotool(self):
        with mock.patch.dict(os.environ, {"DISPLAY": ""}):
            self.assertFalse(ax.click_at(0.5, 0.5))
        self.assertEqual(self.xdotool_calls(), [])

    def test_missing_xdotool_fails_gracefully(self):
        with mock.patch("action_executor.shutil.which", return_value=None):
            self.assertFalse(ax.click_at(0.5, 0.5))

    def test_xdotool_error_returns_false(self):
        self.mock_run.side_effect = lambda args, **kw: fake_run(returncode=1, stderr="boom")
        self.assertFalse(ax.click_at(0.5, 0.5))

    def test_retry_without_verify_uses_offsets_on_failure(self):
        results = iter([False, True])

        def side_effect(args, **kwargs):
            if args[1] == "getdisplaygeometry":
                return self.screen_result
            return fake_run(returncode=0 if next(results) else 1)

        self.mock_run.side_effect = side_effect
        self.assertTrue(ax.click_with_retry(0.5, 0.5))

    def test_retry_with_verify_repeats_until_effect(self):
        verdicts = iter([False, False, True])  # first click, first retry, second retry
        self.assertTrue(ax.click_with_retry(0.5, 0.5, verify=lambda: next(verdicts), retries=2))
        clicks = [c for c in self.xdotool_calls() if c[0] == "mousemove"]
        self.assertEqual(len(clicks), 3)
        self.assertNotEqual(clicks[0], clicks[1])  # retries use a pixel offset

    def test_retry_gives_up(self):
        self.assertFalse(ax.click_with_retry(0.5, 0.5, verify=lambda: False, retries=2))


class Typing(ExecutorBase):
    def test_type_text_clicks_field_then_types(self):
        self.assertTrue(ax.type_text("hello", 0.5, 0.1))
        calls = self.xdotool_calls()
        self.assertEqual(calls[0][0], "mousemove")
        self.assertEqual(calls[-1][-2:], ["--", "hello"])

    def test_type_text_without_coordinates_types_only(self):
        ax.type_text("hi")
        self.assertEqual([c[0] for c in self.xdotool_calls()], ["type"])

    def test_text_starting_with_dash_is_safe(self):
        ax.type_text("-rf")
        self.assertEqual(self.xdotool_calls()[-1][-2:], ["--", "-rf"])

    def test_press_key(self):
        self.assertTrue(ax.press_key("Return"))
        self.assertEqual(self.xdotool_calls(), [["key", "Return"]])


class Scrolling(ExecutorBase):
    def test_scroll_down_three(self):
        self.assertTrue(ax.scroll("down", 3))
        self.assertEqual(self.xdotool_calls()[0][:4], ["click", "--repeat", "3", "--delay"])
        self.assertEqual(self.xdotool_calls()[0][-1], "5")

    def test_scroll_up_button(self):
        ax.scroll("up")
        self.assertEqual(self.xdotool_calls()[0][-1], "4")

    def test_bad_direction_raises(self):
        with self.assertRaises(ValueError):
            ax.scroll("sideways")

    def test_bad_amount_raises(self):
        with self.assertRaises(ValueError):
            ax.scroll("down", 0)


class Drag(ExecutorBase):
    def test_drag_press_move_release(self):
        self.assertTrue(ax.drag((0.1, 0.1), (0.9, 0.9)))
        flat = self.xdotool_calls()[0]
        self.assertIn("mousedown", flat)
        self.assertIn("mouseup", flat)
        self.assertLess(flat.index("mousedown"), flat.index("mouseup"))


class Navigation(ExecutorBase):
    def test_navigate_adds_scheme_and_opens_browser(self):
        with mock.patch("action_executor.webbrowser.open", return_value=True) as opener:
            self.assertTrue(ax.navigate_to("youtube.com"))
        opener.assert_called_once_with("https://youtube.com")

    def test_navigate_keeps_existing_scheme(self):
        with mock.patch("action_executor.webbrowser.open", return_value=True) as opener:
            ax.navigate_to("http://localhost:8080")
        opener.assert_called_once_with("http://localhost:8080")

    def test_navigate_reports_failure(self):
        with mock.patch("action_executor.webbrowser.open", return_value=False):
            self.assertFalse(ax.navigate_to("example.com"))

    def test_navigate_empty_url(self):
        self.assertFalse(ax.navigate_to("  "))


class Misc(ExecutorBase):
    def test_wait_sleeps(self):
        ax.wait(1.5)
        self.mock_sleep.assert_called_with(1.5)

    def test_active_window_title(self):
        self.mock_run.side_effect = lambda args, **kw: fake_run(stdout="Inbox - Mail\n")
        self.assertEqual(ax.get_active_window_title(), "Inbox - Mail")


if __name__ == "__main__":
    unittest.main()
