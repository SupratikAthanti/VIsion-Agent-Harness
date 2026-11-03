"""
OpenCV template matching for trained UI buttons.

Coordinates are returned as 0.0-1.0 fractions of the target image so they are
independent of window size, matching the rest of the Vision CLI.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from template_manager import load_template

# Normalized correlation is unreliable on near-flat patches (it reports ~1.0 at
# arbitrary spots), so templates or matched windows with less contrast are rejected.
MIN_STD = 5.0
# Coarse pass runs on a downscaled image for speed; the decision is made at full
# resolution inside a small window around the coarse hit.
COARSE_SCALE = 0.25
# The coarse pass keeps this many candidate peaks; repeated UI elements can make the
# single best coarse peak the wrong one, so each candidate is refined at full resolution.
COARSE_PEAKS = 10


def _match_full(target: np.ndarray, template: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def _coarse_peaks(small_i: np.ndarray, small_t: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """Top-k peak locations of the coarse correlation map, suppressing each peak's neighborhood."""
    work = cv2.matchTemplate(small_i, small_t, cv2.TM_CCOEFF_NORMED)
    work = np.nan_to_num(work, nan=-2.0, posinf=-2.0, neginf=-2.0)
    rh, rw = small_t.shape[:2]
    peaks: List[Tuple[int, int]] = []
    for _ in range(k):
        _, max_val, _, (px, py) = cv2.minMaxLoc(work)
        if max_val <= -1.5:
            break
        peaks.append((int(px), int(py)))
        work[max(0, py - rh // 2):py + rh // 2 + 1, max(0, px - rw // 2):px + rw // 2 + 1] = -2.0
    return peaks


def _locate(target: np.ndarray, template: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """Return (score, top-left) of the best full-resolution match, using a coarse-then-refine search."""
    th, tw = template.shape[:2]
    ih, iw = target.shape[:2]
    s = COARSE_SCALE
    small_t = cv2.resize(template, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small_i = cv2.resize(target, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if small_t.shape[0] < 4 or small_t.shape[1] < 4 or small_t.shape[0] > small_i.shape[0] or small_t.shape[1] > small_i.shape[1]:
        return _match_full(target, template)

    margin = int(math.ceil(2 / s))  # covers the coarse rounding error at full resolution
    best: Tuple[float, Tuple[int, int]] = (-2.0, (0, 0))
    for cx, cy in _coarse_peaks(small_i, small_t, COARSE_PEAKS):
        fx, fy = int(cx / s), int(cy / s)
        x0, y0 = max(0, fx - margin), max(0, fy - margin)
        x1, y1 = min(iw, fx + tw + margin), min(ih, fy + th + margin)
        roi = target[y0:y1, x0:x1]
        if roi.shape[0] < th or roi.shape[1] < tw:
            continue
        score, (rx, ry) = _match_full(roi, template)
        if score > best[0]:
            best = (score, (x0 + rx, y0 + ry))
    if best[0] <= -2.0:
        return _match_full(target, template)
    return best


def find_template(target_image_path: Path, template_path: Path,
                  threshold: float = 0.8) -> Optional[Dict]:
    """
    Use OpenCV template matching to find template in target image.
    Returns: {"x_pct": float, "y_pct": float, "confidence": float} or None
    The point is the center of the best match.
    """
    target = cv2.imread(str(target_image_path), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if target is None or template is None:
        return None

    th, tw = template.shape[:2]
    ih, iw = target.shape[:2]
    if th > ih or tw > iw:
        return None
    if float(template.std()) < MIN_STD:
        return None  # flat template: no reliable match

    max_val, (left, top) = _locate(target, template)
    if max_val < threshold:
        return None
    if float(target[top:top + th, left:left + tw].std()) < MIN_STD:
        return None  # matched a flat patch, not the button

    return {
        "x_pct": (left + tw / 2) / iw,
        "y_pct": (top + th / 2) / ih,
        "confidence": max_val,
    }


def find_best_match(target_image_path: Path, template_names: List[str],
                    threshold: float = 0.7) -> Optional[Dict]:
    """
    Try multiple templates, return best match above threshold.
    Returns: {"name": str, "x_pct": float, "y_pct": float, "confidence": float} or None
    """
    best: Optional[Dict] = None
    for name in template_names:
        template_path = load_template(name)
        if template_path is None:
            continue
        match = find_template(target_image_path, template_path, threshold)
        if match and (best is None or match["confidence"] > best["confidence"]):
            best = {"name": name, **match}
    return best
