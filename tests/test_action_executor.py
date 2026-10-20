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


class AtspiActions(unittest.TestCase):
    def test_click_element_invokes_click_action_by_name(self):
        action = mock.Mock(nActions=2)
        action.getName.side_effect = ["show", "click"]
        action.doAction.return_value = True
        accessible = mock.Mock()
        accessible.queryAction.return_value = action
        backend = (mock.sentinel.pyatspi, "pyatspi")
        with (
            mock.patch.object(ax.atspi_bridge, "_backend", return_value=backend),
            mock.patch.object(ax.atspi_bridge, "_resolve_ref", return_value=accessible),
        ):
            self.assertTrue(ax.atspi_bridge.click_element("/0/1"))
        action.doAction.assert_called_once_with(1)

    def test_click_element_returns_false_when_no_supported_action(self):
        action = mock.Mock(nActions=1)
        action.getName.return_value = "focus"
        accessible = mock.Mock()
        accessible.queryAction.return_value = action
        backend = (mock.sentinel.pyatspi, "pyatspi")
        with (
            mock.patch.object(ax.atspi_bridge, "_backend", return_value=backend),
            mock.patch.object(ax.atspi_bridge, "_resolve_ref", return_value=accessible),
        ):
            self.assertFalse(ax.atspi_bridge.click_element("/0/1"))
        action.doAction.assert_not_called()


class Clicks(ExecutorBase):
    def test_click_at_sends_absolute_move_and_click(self):
        with mock.patch("action_executor._focus_window_at", return_value=True):
            self.assertTrue(ax.click_at(0.25, 0.5))
        self.assertIn(["mousemove", "--sync", "480", "540", "click", "1"], self.xdotool_calls())

    def test_click_with_offset(self):
        with mock.patch("action_executor._focus_window_at", return_value=True):
            ax.click_at(0.5, 0.5, offset_px=(6, 0))
        self.assertIn(["mousemove", "--sync", "966", "540", "click", "1"], self.xdotool_calls())

    def test_atspi_action_click_does_not_focus_or_use_xdotool(self):
        with (
            mock.patch("action_executor.atspi_bridge.is_at_spi_available", return_value=True),
            mock.patch("action_executor.atspi_bridge.click_element", return_value=True) as click,
            mock.patch("action_executor._focus_window_at") as focus,
        ):
            self.assertTrue(ax.click_at(0.25, 0.5, accessible_ref="/0/1"))
        click.assert_called_once_with("/0/1")
        focus.assert_not_called()
        self.assertEqual(self.xdotool_calls(), [])

    def test_atspi_unavailable_falls_back_to_focus_then_xdotool(self):
        events = []

        def run_click(args):
            events.append(("xdotool", args))
            return True

        with (
            mock.patch("action_executor.atspi_bridge.is_at_spi_available", return_value=False),
            mock.patch("action_executor._focus_window_at", side_effect=lambda x, y: events.append(("focus", x, y)) or True),
            mock.patch("action_executor._run", side_effect=run_click),
        ):
            self.assertTrue(ax.click_at(0.25, 0.5))
        self.assertEqual(events[0], ("focus", 480, 540))
        self.assertEqual(events[1][0], "xdotool")

    def test_failed_atspi_action_falls_back_to_wmctrl_and_xdotool(self):
        with (
            mock.patch("action_executor.atspi_bridge.is_at_spi_available", return_value=True),
            mock.patch("action_executor.atspi_bridge.click_element", return_value=False),
            mock.patch("action_executor._focus_window_at", return_value=True) as focus,
            mock.patch("action_executor._run", return_value=True) as run,
        ):
            self.assertTrue(ax.click_at(0.25, 0.5, accessible_ref="/0/1"))
        focus.assert_called_once_with(480, 540)
        run.assert_called_once()

    def test_atspi_action_can_be_disabled(self):
        with (
            mock.patch("action_executor.atspi_bridge.is_at_spi_available") as available,
            mock.patch("action_executor._focus_window_at", return_value=True),
            mock.patch("action_executor._run", return_value=True),
        ):
            self.assertTrue(ax.click_at(0.25, 0.5, use_atspi_action=False))
        available.assert_not_called()

    def test_wmctrl_focuses_smallest_window_containing_target(self):
        outputs = [
            fake_run(stdout="0x001 0 0 0 1920 1080 desktop\n"
                            "0x002 0 0 0 900 1080 terminal\n"
                            "0x003 0 900 0 1020 1080 chrome\n"),
            fake_run(),
        ]
        with (
            mock.patch("action_executor.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
            mock.patch("action_executor.subprocess.run", side_effect=outputs) as run,
        ):
            self.assertTrue(ax._focus_window_at(1200, 500))
        self.assertEqual(run.call_args_list[1].args[0], ["/usr/bin/wmctrl", "-ia", "0x003"])

    def test_no_window_at_target_refuses_to_click(self):
        with (
            mock.patch("action_executor._focus_window_at", return_value=False),
            mock.patch("action_executor._run") as run,
        ):
            self.assertFalse(ax.click_at(0.25, 0.5))
        run.assert_not_called()

    def test_wmctrl_missing_fails_without_running_xdotool(self):
        with (
            mock.patch("action_executor.shutil.which", side_effect=lambda name: "/usr/bin/xdotool" if name == "xdotool" else None),
            mock.patch("action_executor._run") as run,
        ):
            self.assertFalse(ax.click_at(0.25, 0.5))
        run.assert_not_called()

    def test_click_previews_target_and_hides_overlay(self):
        events = []
        overlay = mock.Mock()
        overlay.show_cursor.side_effect = lambda x, y: events.append(("show", x, y)) or True
        overlay.hide_cursor.side_effect = lambda: events.append(("hide",))

        def run_click(args):
            events.append(("click", args))
            return True

        with (
            mock.patch("action_executor._get_cursor_overlay", return_value=overlay),
            mock.patch("action_executor._focus_window_at", return_value=True),
            mock.patch("action_executor._run", side_effect=run_click),
            mock.patch("action_executor.time.sleep") as sleep,
        ):
            self.assertTrue(ax.click_at(0.25, 0.5))

        self.assertEqual(events[0], ("show", 480, 540))
        self.assertEqual(events[1][0], "click")
        self.assertEqual(events[2], ("hide",))
        sleep.assert_any_call(ax.CURSOR_PREVIEW_SECONDS)

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
        with mock.patch("action_executor._focus_window_at", return_value=True):
            self.assertTrue(ax.click_with_retry(0.5, 0.5))

    def test_retry_with_verify_repeats_until_effect(self):
        verdicts = iter([False, False, True])  # first click, first retry, second retry
        with mock.patch("action_executor._focus_window_at", return_value=True):
            self.assertTrue(ax.click_with_retry(0.5, 0.5, verify=lambda: next(verdicts), retries=2))
        clicks = [c for c in self.xdotool_calls() if c[0] == "mousemove"]
        self.assertEqual(len(clicks), 3)
        self.assertNotEqual(clicks[0], clicks[1])  # retries use a pixel offset

    def test_retry_gives_up(self):
        self.assertFalse(ax.click_with_retry(0.5, 0.5, verify=lambda: False, retries=2))


class Typing(ExecutorBase):
    def test_type_text_clicks_field_then_types(self):
        with mock.patch("action_executor._focus_window_at", return_value=True):
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
