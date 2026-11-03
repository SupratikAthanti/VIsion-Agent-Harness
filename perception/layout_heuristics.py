"""
Layout heuristics for common UI patterns, plus spatial reasoning between elements.

All locations are dicts with fractional coordinates (0.0-1.0 of the image), the same
convention as template_matcher and the OCR engine:

    {"x_pct": 0.5, "y_pct": 0.08, "confidence": 0.35, "source": "heuristic", "label": "search box"}

`words` is an optional list of OCR word dicts (as returned by
ocr_engine.extract_text_with_coordinates). When it is given, the heuristics look for
real text labels first (e.g. a "Search" placeholder) and only fall back to fixed
layout priors when no label is found. No model is loaded here, so calls are fast.
"""

from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

# Confidence levels: text found on screen beats a layout prior.
CONF_TEXT = 0.85
CONF_PRIOR = 0.35

SEARCH_LABELS = {"search", "find", "query"}
SUBMIT_LABELS = {"submit", "search", "go", "send", "login", "signin", "ok", "continue", "next", "save", "post"}
NAV_LABELS = {"home", "back", "forward", "menu", "settings", "file", "edit", "view", "help", "tools", "tabs"}
CLOSE_LABELS = {"x", "×", "✕", "close", "✖"}
MENU_LABELS = {"menu", "≡", "☰", "⋮"}

ELEMENT_TYPES = ("submit", "search", "navigation", "close", "menu")
TOP_BAND = 0.15     # fraction of image height treated as the top bar
LEFT_BAND = 0.2     # fraction of image width treated as a left sidebar
RIGHT_BAND = 0.85   # fraction of image width treated as the right edge


def _location(x: float, y: float, confidence: float, source: str, label: str = "") -> Dict:
    return {
        "x_pct": max(0.0, min(1.0, x)),
        "y_pct": max(0.0, min(1.0, y)),
        "confidence": confidence,
        "source": source,
        "label": label,
    }


def _image_readable(image_path: Path) -> bool:
    """True if the image can be opened. Heuristics return None for unreadable images."""
    try:
        with Image.open(image_path) as img:
            return img.size[0] > 0 and img.size[1] > 0
    except (OSError, ValueError):
        return False


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch in "×✕✖≡☰⋮")


def _words_in(words: Optional[List[Dict]], labels: set, region=None) -> List[Dict]:
    """OCR words whose normalized text is in `labels` and whose center is inside `region`."""
    if not words:
        return []
    hits = []
    for w in words:
        if _norm(w["text"]) not in labels:
            continue
        cx, cy = w["center"]
        if region is not None and not region(cx, cy):
            continue
        hits.append(w)
    return hits


def _word_location(word: Dict, label: str) -> Dict:
    cx, cy = word["center"]
    return _location(cx, cy, CONF_TEXT, "ocr", label or word["text"])


def _distance(a: Dict, b: Dict) -> float:
    return ((a["x_pct"] - b["x_pct"]) ** 2 + (a["y_pct"] - b["y_pct"]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Single-element predictions
# ---------------------------------------------------------------------------

def predict_search_box_location(image_path: Path, words: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Search boxes sit in the top band, usually centred or top-left, and often show a 'Search' placeholder."""
    if not _image_readable(image_path):
        return None
    hits = _words_in(words, SEARCH_LABELS, lambda x, y: y <= 0.3)
    if hits:
        best = min(hits, key=lambda w: w["center"][1])
        return _word_location(best, "search box")
    return _location(0.5, 0.08, CONF_PRIOR, "heuristic", "search box")


def predict_submit_button_location(image_path: Path, form_location: Dict,
                                   words: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Submit buttons sit right of, or below, the form field they belong to."""
    if not _image_readable(image_path) or not form_location:
        return None
    hits = _words_in(words, SUBMIT_LABELS)
    if hits:
        best = min(hits, key=lambda w: _distance(
            _location(w["center"][0], w["center"][1], 0, ""), form_location))
        return _word_location(best, "submit button")

    half_w = form_location.get("width_pct", 0.2) / 2
    fx, fy = form_location["x_pct"], form_location["y_pct"]
    if fx + half_w + 0.06 <= RIGHT_BAND:
        return _location(fx + half_w + 0.06, fy, CONF_PRIOR, "heuristic", "submit button")
    return _location(fx, fy + 0.06, CONF_PRIOR, "heuristic", "submit button")


def predict_navigation_location(image_path: Path, layout: str = "top_bar",
                                words: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Navigation is a top bar (default) or a left sidebar."""
    if not _image_readable(image_path):
        return None
    if layout == "sidebar":
        region = lambda x, y: x <= LEFT_BAND
        prior = (0.06, 0.5)
    elif layout == "top_bar":
        region = lambda x, y: y <= TOP_BAND
        prior = (0.5, 0.04)
    else:
        raise ValueError(f"Unknown layout: {layout!r} (use 'top_bar' or 'sidebar')")
    hits = _words_in(words, NAV_LABELS, region)
    if hits:
        return _word_location(min(hits, key=lambda w: (w["center"][1], w["center"][0])), "navigation")
    return _location(prior[0], prior[1], CONF_PRIOR, "heuristic", "navigation")


def predict_close_button_location(image_path: Path, words: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Close buttons sit in the top-right corner of windows and tabs."""
    if not _image_readable(image_path):
        return None
    hits = _words_in(words, CLOSE_LABELS, lambda x, y: y <= TOP_BAND and x >= RIGHT_BAND)
    if hits:
        return _word_location(hits[-1], "close button")
    return _location(0.97, 0.03, CONF_PRIOR, "heuristic", "close button")


def predict_menu_button_location(image_path: Path, words: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Menu (hamburger) buttons sit in the top-left corner."""
    if not _image_readable(image_path):
        return None
    hits = _words_in(words, MENU_LABELS, lambda x, y: y <= TOP_BAND and x <= LEFT_BAND)
    if hits:
        return _word_location(hits[0], "menu button")
    return _location(0.03, 0.04, CONF_PRIOR, "heuristic", "menu button")


# ---------------------------------------------------------------------------
# Spatial reasoning
# ---------------------------------------------------------------------------

def _prior_for(element_type: str, reference: Dict, image_path: Path, words) -> Optional[Dict]:
    if element_type == "submit":
        return predict_submit_button_location(image_path, reference, words)
    if element_type == "search":
        return predict_search_box_location(image_path, words)
    if element_type == "navigation":
        return predict_navigation_location(image_path, "top_bar", words)
    if element_type == "close":
        return predict_close_button_location(image_path, words)
    if element_type == "menu":
        return predict_menu_button_location(image_path, words)
    raise ValueError(f"Unknown element type: {element_type!r} (expected one of {ELEMENT_TYPES})")


_TYPE_LABELS = {
    "submit": SUBMIT_LABELS,
    "search": SEARCH_LABELS,
    "navigation": NAV_LABELS,
    "close": CLOSE_LABELS,
    "menu": MENU_LABELS,
}


def find_nearby_element(image_path: Path, reference_location: Dict, element_type: str,
                        max_distance: float = 0.15, words: Optional[List[Dict]] = None) -> Optional[Dict]:
    """
    Find an element of `element_type` within `max_distance` (fraction of image size) of a reference.

    Text labels from `words` win over layout priors. When several candidates are in range,
    the nearest one is returned. Distance is measured in fraction space, not pixels.
    """
    if element_type not in ELEMENT_TYPES:
        raise ValueError(f"Unknown element type: {element_type!r} (expected one of {ELEMENT_TYPES})")
    if not reference_location or not _image_readable(image_path):
        return None

    near = [
        w for w in _words_in(words, _TYPE_LABELS[element_type])
        if _distance(_location(w["center"][0], w["center"][1], 0, ""), reference_location) <= max_distance
    ]
    if near:
        best = min(near, key=lambda w: _distance(
            _location(w["center"][0], w["center"][1], 0, ""), reference_location))
        return _word_location(best, element_type)

    candidate = _prior_for(element_type, reference_location, image_path, None)
    if candidate and _distance(candidate, reference_location) <= max_distance:
        return candidate
    return None


def find_submit_button_near_field(image_path: Path, field_location: Dict,
                                  words: Optional[List[Dict]] = None) -> Optional[Dict]:
    """After typing in a field, the submit button is usually right of it or just below it."""
    return find_nearby_element(image_path, field_location, "submit", max_distance=0.2, words=words)


def find_menu_items_below(image_path: Path, menu_location: Dict,
                          words: Optional[List[Dict]] = None,
                          max_depth: float = 0.35, horizontal_tolerance: float = 0.12) -> List[Dict]:
    """
    Text items in the column below a clicked menu, nearest first.
    Needs OCR words: without them there is nothing to report, so this returns [].
    """
    if not words or not menu_location or not _image_readable(image_path):
        return []
    items = []
    mx, my = menu_location["x_pct"], menu_location["y_pct"]
    for w in words:
        cx, cy = w["center"]
        dy = cy - my
        if 0.005 < dy <= max_depth and abs(cx - mx) <= horizontal_tolerance:
            items.append({
                "text": w["text"],
                "x_pct": cx,
                "y_pct": cy,
                "confidence": CONF_TEXT,
                "source": "ocr",
            })
    return sorted(items, key=lambda item: item["y_pct"])


def find_content_area_for_tab(image_path: Path, tab_location: Dict) -> Optional[Dict]:
    """
    The content shown by a tab: the area below a top tab bar, or to the right of a vertical tab list.
    Returns the centre of that area plus its bounding box in fractions.
    """
    if not tab_location or not _image_readable(image_path):
        return None
    if tab_location["y_pct"] <= TOP_BAND:
        bbox = {"x0": 0.0, "y0": min(tab_location["y_pct"] + 0.05, 0.9), "x1": 1.0, "y1": 0.95}
    elif tab_location["x_pct"] <= LEFT_BAND:
        bbox = {"x0": min(tab_location["x_pct"] + 0.15, 0.9), "y0": 0.0, "x1": 1.0, "y1": 1.0}
    else:
        return None
    return {
        "x_pct": (bbox["x0"] + bbox["x1"]) / 2,
        "y_pct": (bbox["y0"] + bbox["y1"]) / 2,
        "confidence": CONF_PRIOR,
        "source": "heuristic",
        "label": "tab content",
        "bbox": bbox,
    }
