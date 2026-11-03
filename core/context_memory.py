"""
In-memory context for the action pipeline: what page we think we are on, the last
few actions, what was typed where, and where targets were last found.

Nothing is persisted; the state lives for the length of the CLI session.

    {
      "current_page": "youtube.com",
      "last_actions": [{"action": "CLICK", "target": "search box", "time": ..., "location": (0.5, 0.1)}, ...],
      "form_state": {"search box": "cats"},
    }
"""

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

MAX_ACTIONS = 10
THUMB_SIZE = (64, 64)
# Mean absolute grey-level difference (0-255) between two 64x64 thumbnails that counts as a new page.
PAGE_CHANGE_THRESHOLD = 25.0

_lock = threading.Lock()


def _empty_state() -> Dict:
    return {
        "current_page": None,
        "last_actions": [],
        "form_state": {},
        "known_locations": {},   # target name -> (x_pct, y_pct) from the last successful locate
        "reference_thumb": None,  # last seen screen, used by detect_page_change
    }


_state: Dict = _empty_state()


def _thumbnail(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    try:
        with Image.open(image_path) as img:
            small = img.convert("L").resize(THUMB_SIZE, Image.Resampling.BILINEAR)
            return np.asarray(small, dtype=np.float32)
    except (OSError, ValueError):
        return None


def _clear_page_state() -> None:
    """Called when the page changes: values typed or located on the old page no longer apply."""
    _state["form_state"] = {}
    _state["known_locations"] = {}


def update_context(action: Dict, image_path: Optional[Union[str, Path]] = None) -> None:
    """
    Record a successful action. `action` is an intent dict, optionally with a "location"
    entry ({"x_pct": ..., "y_pct": ...} or an (x, y) tuple) where the action landed.
    Pass `image_path` to store a reference screenshot for later page-change detection.
    """
    kind = action.get("action")
    target = action.get("target")
    params = action.get("params") or {}

    location = None
    loc = action.get("location")
    if isinstance(loc, dict) and "x_pct" in loc and "y_pct" in loc:
        location = (float(loc["x_pct"]), float(loc["y_pct"]))
    elif isinstance(loc, (tuple, list)) and len(loc) == 2:
        location = (float(loc[0]), float(loc[1]))

    record = {"action": kind, "time": time.time()}
    if target:
        record["target"] = target
    if "text" in params and params["text"] is not None:
        record["text"] = params["text"]
    if params.get("url"):
        record["url"] = params["url"]
    if location is not None:
        record["location"] = location

    thumb = _thumbnail(image_path) if image_path is not None else None

    with _lock:
        _state["last_actions"].append(record)
        del _state["last_actions"][:-MAX_ACTIONS]

        if kind == "TYPE" and target and "text" in params and params["text"] is not None:
            _state["form_state"][target] = params["text"]
        if kind == "NAVIGATE" and params.get("url"):
            _state["current_page"] = _page_name(params["url"])
            _clear_page_state()
        if target and location is not None and kind in ("CLICK", "TYPE"):
            _state["known_locations"][target] = location
        if thumb is not None:
            _state["reference_thumb"] = thumb


def _page_name(url: str) -> str:
    name = url.split("://", 1)[-1]
    return name.rstrip("/")


def get_recent_actions(count: int = 5) -> List[Dict]:
    """The last `count` recorded actions, oldest first."""
    if count <= 0:
        return []
    with _lock:
        return [dict(a) for a in _state["last_actions"][-count:]]


def get_form_state() -> Dict:
    """Map of field name -> text typed into it on the current page."""
    with _lock:
        return dict(_state["form_state"])


def get_known_location(target: str) -> Optional[Tuple[float, float]]:
    """Where `target` was last successfully located on the current page, if known."""
    with _lock:
        return _state["known_locations"].get(target)


def set_current_page(name: Optional[str]) -> None:
    """Set the page name, e.g. the active window title. A changed name clears page-scoped state."""
    with _lock:
        if name and name != _state["current_page"]:
            _state["current_page"] = name
            _clear_page_state()


def get_context() -> Dict:
    """Copy of the whole context, for display (/context)."""
    with _lock:
        return {
            "current_page": _state["current_page"],
            "last_actions": [dict(a) for a in _state["last_actions"]],
            "form_state": dict(_state["form_state"]),
            "known_locations": dict(_state["known_locations"]),
        }


def clear_context() -> None:
    """Reset all context (/clear-context)."""
    global _state
    with _lock:
        _state = _empty_state()


def image_difference(path_a: Union[str, Path], path_b: Union[str, Path]) -> Optional[float]:
    """Mean grey-level difference (0-255) between two screenshots at thumbnail size. None if unreadable."""
    a, b = _thumbnail(path_a), _thumbnail(path_b)
    if a is None or b is None:
        return None
    return float(np.mean(np.abs(a - b)))


def detect_page_change(new_image_path: Union[str, Path]) -> bool:
    """
    Compare a new screenshot with the last reference screenshot.

    Returns True when the page has changed. A change clears the form state and
    known locations, because they belong to the old page. The new screenshot
    becomes the reference for the next comparison. The first call only sets the
    reference and returns False.
    """
    new_thumb = _thumbnail(new_image_path)
    if new_thumb is None:
        return False
    with _lock:
        previous = _state["reference_thumb"]
        _state["reference_thumb"] = new_thumb
        if previous is None:
            return False
        changed = float(np.mean(np.abs(new_thumb - previous))) > PAGE_CHANGE_THRESHOLD
        if changed:
            _clear_page_state()
        return changed
