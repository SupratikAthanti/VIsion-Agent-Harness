"""
OCR text extraction with coordinate mapping for the Vision Agent Harness.

Uses the system `tesseract` CLI (TSV output), so no extra Python dependency
is required. Coordinates are returned as 0.0-1.0 fractions of the original
image, the same convention used by the /mark command.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

TESSERACT_TIMEOUT = 60  # seconds
TESSERACT_PSM = "11"    # sparse text: suits UI screenshots with scattered labels


class OCRError(RuntimeError):
    """Raised when OCR cannot run (missing binary, bad image, timeout)."""


def _normalize(token: str) -> str:
    return re.sub(r"\W", "", token).lower()


def extract_text_with_coordinates(image_path: Path, resize_max: int = 800) -> List[Dict]:
    """
    OCR an image and return one dict per detected word:
        text      - recognized word
        conf      - Tesseract confidence (0-100)
        line      - (block, paragraph, line) key used to group words
        box       - (x0, y0, x1, y1) as 0.0-1.0 fractions of the image
        center    - (x, y) as 0.0-1.0 fractions of the image
        bbox_px   - (left, top, width, height) in ORIGINAL pixels

    The image is resized to `resize_max` pixels wide before OCR.
    """
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise OCRError("tesseract is not installed (sudo apt install tesseract-ocr)")

    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        scale = min(1.0, resize_max / orig_w)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))
        resized = img.convert("L").resize((new_w, new_h), Image.Resampling.LANCZOS)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_png = Path(tmp) / "ocr_input.png"
        resized.save(tmp_png, "PNG")
        try:
            proc = subprocess.run(
                [tesseract, str(tmp_png), "stdout", "--psm", TESSERACT_PSM, "tsv"],
                capture_output=True,
                text=True,
                timeout=TESSERACT_TIMEOUT,
            )
        except subprocess.TimeoutExpired as e:
            raise OCRError(f"tesseract timed out after {TESSERACT_TIMEOUT}s") from e

    if proc.returncode != 0:
        raise OCRError(f"tesseract failed: {proc.stderr.strip()[:200]}")

    words: List[Dict] = []
    rows = proc.stdout.splitlines()[1:]  # skip TSV header
    for row in rows:
        cols = row.split("\t")
        # TSV: level, page, block, par, line, word, left, top, width, height, conf, text
        if len(cols) < 12 or cols[0] != "5":
            continue
        text = cols[11].strip()
        conf = float(cols[10])
        if not text or conf < 0:
            continue
        left, top, w, h = (int(c) for c in cols[6:10])
        x0, y0 = left / new_w, top / new_h
        x1, y1 = (left + w) / new_w, (top + h) / new_h
        words.append({
            "text": text,
            "conf": conf,
            "line": (cols[2], cols[3], cols[4]),
            "box": (x0, y0, x1, y1),
            "center": ((x0 + x1) / 2, (y0 + y1) / 2),
            "bbox_px": (
                round(x0 * orig_w),
                round(y0 * orig_h),
                round((x1 - x0) * orig_w),
                round((y1 - y0) * orig_h),
            ),
        })
    return words


# If the phrase is not found at the first width, retry at these widths.
# 800px alone misses small UI labels on large screenshots (tested: a 2932px-wide
# screenshot only yields 18 words at 800px, but 71 words at 1600px).
RETRY_WIDTHS = (1600,)


def _match_phrase(words: List[Dict], tokens: List[str]) -> Optional[Dict]:
    n = len(tokens)
    for i in range(len(words) - n + 1):
        window = words[i:i + n]
        if any(w["line"] != window[0]["line"] for w in window):
            continue
        if all(_normalize(w["text"]) == tok for w, tok in zip(window, tokens)):
            x0 = min(w["box"][0] for w in window)
            y0 = min(w["box"][1] for w in window)
            x1 = max(w["box"][2] for w in window)
            y1 = max(w["box"][3] for w in window)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            left = window[0]["bbox_px"][0]
            top = min(w["bbox_px"][1] for w in window)
            right = max(w["bbox_px"][0] + w["bbox_px"][2] for w in window)
            bottom = max(w["bbox_px"][1] + w["bbox_px"][3] for w in window)
            return {
                "text": " ".join(w["text"] for w in window),
                "center_x": cx,
                "center_y": cy,
                "center_pct": (cx * 100, cy * 100),
                "bbox_px": (left, top, right - left, bottom - top),
                "conf": min(w["conf"] for w in window),
            }
    return None


def find_text_element(image_path: Path, search_text: str, resize_max: int = 800) -> Optional[Dict]:
    """
    Find a (possibly multi-word) phrase on screen.

    Words must appear consecutively on the same text line, compared
    case-insensitively and ignoring punctuation. Tries `resize_max` first, then
    each wider width in RETRY_WIDTHS. Returns None if not found, otherwise a dict
    with `text`, `center_x`, `center_y` (0.0-1.0), `center_pct` (0-100),
    `bbox_px` and `conf` (lowest word confidence in the match).
    
    Also applies heuristic rules for common UI patterns.
    """
    tokens = [t for t in (_normalize(p) for p in search_text.split()) if t]
    if not tokens:
        return None

    widths = [resize_max] + [w for w in RETRY_WIDTHS if w > resize_max]
    for width in widths:
        words = extract_text_with_coordinates(image_path, resize_max=width)
        match = _match_phrase(words, tokens)
        if match:
            # Get image size for heuristics
            with Image.open(image_path) as img:
                img_size = img.size
            
            # Apply heuristics to potentially adjust the result
            search_phrase = extract_search_phrase(search_text)
            heuristic_result = apply_heuristics(search_phrase, match, words, img_size)
            if heuristic_result:
                return heuristic_result
            return match
    return None


_LEADING = re.compile(
    r"^\s*(?:where\s+(?:is|are)|where's|find|locate|click(?:\s+on)?|press|tap|open|select|go\s+to)\s+",
    re.IGNORECASE,
)
_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_TRAILING = re.compile(r"\s+(?:button|link|icon|tab|menu|option|field)\s*$", re.IGNORECASE)


def extract_search_phrase(query: str) -> str:
    """
    Pull the on-screen label out of a request, e.g.
    "Where is the Ask Gemini button?" -> "Ask Gemini".
    """
    phrase = query.strip().rstrip("?.!").strip()
    for _ in range(3):
        new = _ARTICLE.sub("", _LEADING.sub("", phrase, count=1), count=1).strip()
        if new == phrase:
            break
        phrase = new
    return _TRAILING.sub("", phrase).strip()


def apply_heuristics(
    search_phrase: str,
    ocr_result: Optional[Dict],
    all_words: List[Dict],
    image_size: tuple
) -> Optional[Dict]:
    """
    Apply heuristic rules to adjust OCR results for common UI patterns.
    
    Returns a modified result dict if a heuristic applies, None otherwise.
    """
    if not ocr_result:
        return None
    
    img_w, img_h = image_size
    phrase_lower = search_phrase.lower()
    
    # Rule: New tab - look for + icon to the right of "New Tab" text
    if "new tab" in phrase_lower:
        # Find words to the right of the New Tab text
        text_x = ocr_result["center_x"]
        text_y = ocr_result["center_y"]
        
        # Look for symbols or small text nearby (within 10% horizontal)
        for word in all_words:
            word_x, word_y = word["center"]
            # Check if word is to the right and on same line
            if (word_x > text_x and 
                word_x - text_x < 0.15 and  # Within 15% to the right
                abs(word_y - text_y) < 0.05):  # Same line (within 5%)
                # Check if it looks like a plus sign or small element
                word_text = word["text"].strip()
                if word_text in ["+", "＋", "add", "new"] or len(word_text) <= 2:
                    # Return the icon coordinates instead
                    return {
                        "text": f"+ (new tab icon)",
                        "center_x": word_x,
                        "center_y": word_y,
                        "center_pct": (word_x * 100, word_y * 100),
                        "bbox_px": word["bbox_px"],
                        "conf": ocr_result["conf"],
                    }
    
    # Rule: Close tab - look for X icon near tab text
    if "close" in phrase_lower and "tab" in phrase_lower:
        text_x = ocr_result["center_x"]
        text_y = ocr_result["center_y"]
        
        # Look for X to the right of the tab label
        for word in all_words:
            word_x, word_y = word["center"]
            if (word_x > text_x and 
                word_x - text_x < 0.10 and
                abs(word_y - text_y) < 0.05):
                word_text = word["text"].strip()
                if word_text.lower() in ["x", "×", "close", "✕"]:
                    return {
                        "text": f"X (close tab icon)",
                        "center_x": word_x,
                        "center_y": word_y,
                        "center_pct": (word_x * 100, word_y * 100),
                        "bbox_px": word["bbox_px"],
                        "conf": ocr_result["conf"],
                    }
    
    # Rule: Menu button - look for hamburger icon or menu text
    if "menu" in phrase_lower:
        text_x = ocr_result["center_x"]
        text_y = ocr_result["center_y"]
        
        # Look for hamburger icon (three lines symbol often OCR'd as ||| or similar)
        for word in all_words:
            word_x, word_y = word["center"]
            if (abs(word_x - text_x) < 0.10 and
                abs(word_y - text_y) < 0.05):
                word_text = word["text"].strip()
                if word_text in ["≡", "|||", "☰", "menu", "⋮"]:
                    return {
                        "text": f"Menu icon",
                        "center_x": word_x,
                        "center_y": word_y,
                        "center_pct": (word_x * 100, word_y * 100),
                        "bbox_px": word["bbox_px"],
                        "conf": ocr_result["conf"],
                    }
    
    # No heuristic applied, return original result
    return None
