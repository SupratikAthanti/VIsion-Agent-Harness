"""
Action executor: performs clicks, typing, scrolling, drags, and waits on the desktop.

Input goes through xdotool (already installed on most Linux desktops). Coordinates
are fractions of the screen (0.0-1.0), the same convention used everywhere else.
Every public call returns True/False (or None for wait) and never raises for
environment problems such as a missing DISPLAY or a missing binary; it prints a
short warning instead, so the pipeline can report the failure and carry on.
"""

import os
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from cursor_overlay import CursorOverlay

DEFAULT_DELAY = 0.25      # settle time after each input, seconds
CURSOR_PREVIEW_SECONDS = 0.6
XDOTOOL_TIMEOUT = 5       # seconds per xdotool call
TYPE_DELAY_MS = 30        # delay between typed characters
SCROLL_BUTTONS = {"up": "4", "down": "5", "left": "6", "right": "7"}
RETRY_OFFSETS_PX = ((6, 0), (-6, 0), (0, 6), (0, -6))
_cursor_overlay: Optional[CursorOverlay] = None


def _get_cursor_overlay() -> Optional[CursorOverlay]:
    """Create the cursor overlay on first use, if a desktop display is available."""
    global _cursor_overlay
    if not os.environ.get("DISPLAY"):
        return None
    if _cursor_overlay is None:
        try:
            _cursor_overlay = CursorOverlay()
        except Exception:
            return None
    return _cursor_overlay


def _warn(message: str) -> None:
    print(f"   [action] {message}")


def _run(args: List[str]) -> bool:
    """Run xdotool with args. Returns True on success, False (with a warning) otherwise."""
    xdotool = shutil.which("xdotool")
    if not xdotool:
        _warn("xdotool is not installed (sudo apt install xdotool)")
        return False
    if not os.environ.get("DISPLAY"):
        _warn("no X display (DISPLAY is not set); cannot send input")
        return False
    try:
        proc = subprocess.run([xdotool, *args], capture_output=True, text=True, timeout=XDOTOOL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        _warn(f"xdotool failed: {e}")
        return False
    if proc.returncode != 0:
        _warn(f"xdotool {args[0]} exited {proc.returncode}: {proc.stderr.strip()[:120]}")
        return False
    return True


def get_screen_size() -> Optional[Tuple[int, int]]:
    """Screen (width, height) in pixels, or None when it cannot be read."""
    xdotool = shutil.which("xdotool")
    if not xdotool or not os.environ.get("DISPLAY"):
        return None
    try:
        proc = subprocess.run([xdotool, "getdisplaygeometry"], capture_output=True,
                              text=True, timeout=XDOTOOL_TIMEOUT)
        w, h = (int(v) for v in proc.stdout.split()[:2])
        return (w, h) if proc.returncode == 0 and w > 0 and h > 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def get_active_window_title() -> Optional[str]:
    """Title of the focused window, used as the current page name. None if unavailable."""
    xdotool = shutil.which("xdotool")
    if not xdotool or not os.environ.get("DISPLAY"):
        return None
    try:
        proc = subprocess.run([xdotool, "getactivewindowname"], capture_output=True,
                              text=True, timeout=XDOTOOL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    title = proc.stdout.strip()
    return title if proc.returncode == 0 and title else None


def to_absolute(x_pct: float, y_pct: float, screen: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    """Convert fractional coordinates to pixels. Raises ValueError outside 0.0-1.0."""
    if not (0.0 <= x_pct <= 1.0 and 0.0 <= y_pct <= 1.0):
        raise ValueError(f"Coordinates must be fractions between 0.0 and 1.0 (got {x_pct}, {y_pct})")
    screen = screen or get_screen_size()
    if screen is None:
        return None
    w, h = screen
    return min(w - 1, round(x_pct * w)), min(h - 1, round(y_pct * h))


def click_at(x_pct: float, y_pct: float, offset_px: Tuple[int, int] = (0, 0)) -> bool:
    """Move the pointer to a fractional screen position and left-click."""
    point = to_absolute(x_pct, y_pct)
    if point is None:
        _warn("could not read the screen size; click skipped")
        return False
    x, y = point[0] + offset_px[0], point[1] + offset_px[1]
    overlay = _get_cursor_overlay()
    preview_shown = False
    if overlay is not None:
        try:
            preview_shown = overlay.show_cursor(x, y)
            if preview_shown:
                time.sleep(CURSOR_PREVIEW_SECONDS)
        except Exception as e:
            _warn(f"cursor preview unavailable: {e}")

    try:
        ok = _run(["mousemove", "--sync", str(x), str(y), "click", "1"])
    finally:
        if overlay is not None:
            try:
                overlay.hide_cursor()
            except Exception as e:
                _warn(f"could not hide cursor preview: {e}")
    time.sleep(DEFAULT_DELAY)
    return ok


def click_with_retry(x_pct: float, y_pct: float,
                     verify: Optional[Callable[[], bool]] = None, retries: int = 2) -> bool:
    """
    Click, then check the result with `verify` (returns True when the click had an effect).
    If it did not, retry with a small pixel offset around the target. Without `verify`,
    only a failed xdotool call is retried.
    """
    ok = click_at(x_pct, y_pct)
    if verify is None:
        if ok:
            return True
        for offset in RETRY_OFFSETS_PX[:retries]:
            if click_at(x_pct, y_pct, offset):
                return True
        return False

    if ok and verify():
        return True
    for offset in RETRY_OFFSETS_PX[:retries]:
        _warn(f"click had no visible effect; retrying with {offset} px offset")
        if click_at(x_pct, y_pct, offset) and verify():
            return True
    return False


def type_text(text: str, x_pct: Optional[float] = None, y_pct: Optional[float] = None) -> bool:
    """Focus the field at (x_pct, y_pct) by clicking it, then type `text`. Without coordinates, types into the focused window."""
    if x_pct is not None and y_pct is not None:
        if not click_at(x_pct, y_pct):
            return False
    ok = _run(["type", "--delay", str(TYPE_DELAY_MS), "--", text])
    time.sleep(DEFAULT_DELAY)
    return ok


def press_key(key: str) -> bool:
    """Press a named key such as 'Return', 'Tab' or 'Escape'."""
    ok = _run(["key", key])
    time.sleep(DEFAULT_DELAY)
    return ok


def navigate_to(url: str) -> bool:
    """
    Open `url` in the default browser. A new URL is opened with webbrowser rather than
    typed into the address bar, because the focused window may not be a browser.
    """
    target = url.strip()
    if not target:
        return False
    if "://" not in target:
        target = "https://" + target
    try:
        opened = webbrowser.open(target)
    except webbrowser.Error as e:
        _warn(f"could not open a browser: {e}")
        return False
    if not opened:
        _warn("no browser could be started")
        return False
    time.sleep(1.0)  # give the browser time to load before the next step
    return True


def scroll(direction: str, amount: int = 1) -> bool:
    """Scroll 'up', 'down', 'left' or 'right' by `amount` wheel notches."""
    key = direction.lower()
    if key not in SCROLL_BUTTONS:
        raise ValueError(f"Unknown scroll direction: {direction!r}")
    if amount < 1:
        raise ValueError("amount must be at least 1")
    ok = _run(["click", "--repeat", str(amount), "--delay", "50", SCROLL_BUTTONS[key]])
    time.sleep(DEFAULT_DELAY)
    return ok


def drag(from_xy: Tuple[float, float], to_xy: Tuple[float, float]) -> bool:
    """Press at one fractional position, move to another, and release."""
    start = to_absolute(*from_xy)
    end = to_absolute(*to_xy)
    if start is None or end is None:
        _warn("could not read the screen size; drag skipped")
        return False
    ok = _run(["mousemove", "--sync", str(start[0]), str(start[1]),
               "mousedown", "1",
               "mousemove", "--sync", str(end[0]), str(end[1]),
               "mouseup", "1"])
    time.sleep(DEFAULT_DELAY)
    return ok


def wait(seconds: float) -> None:
    """Pause for `seconds`."""
    time.sleep(max(0.0, seconds))


def capture_screen(out_path: Union[str, Path]) -> bool:
    """Save a screenshot of the whole display to out_path. Tries Pillow first, then scrot."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import ImageGrab
        ImageGrab.grab(xdisplay=os.environ.get("DISPLAY")).save(out_path, format="PNG")
        return True
    except Exception:
        pass
    scrot = shutil.which("scrot")
    if scrot and os.environ.get("DISPLAY"):
        try:
            proc = subprocess.run([scrot, "-o", str(out_path)], capture_output=True, timeout=XDOTOOL_TIMEOUT)
            return proc.returncode == 0 and out_path.exists()
        except (OSError, subprocess.TimeoutExpired):
            pass
    _warn("could not capture the screen (needs an X display, Pillow ImageGrab or scrot)")
    return False
