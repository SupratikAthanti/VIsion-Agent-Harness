#!/usr/bin/env python3
"""
Vision Agent Harness - CLI MVP (v0.2)
Supports both Moondream2 (via local Ollama, 4-bit) and llama-cpp-python (SmolVLM).
Includes Coordinate Grid Overlays and Visual Target Box verification.
"""

import os
import sys
import time
import threading
import shutil
import json
import math
import re
import base64
import argparse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Try to import Pillow for grid operations
try:
    from PIL import Image
    from grid_overlay import create_coordinate_grid, draw_target_marker, GRID_COLS, GRID_ROWS, column_label
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Hybrid backend: Tesseract OCR + TEV1 routing
try:
    from ocr_engine import find_text_element, extract_search_phrase, extract_text_with_coordinates, OCRError
    from tev1_router import route_task, check_tev1_available, TEV1Error
    from yolo_detector import YOLODetector
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

# AT-SPI is optional; the adapter handles missing bindings and inactive buses.
try:
    import atspi_bridge
    HAS_ATSPI = True
except ImportError:
    HAS_ATSPI = False

# Trained UI templates (/annotate, /teach, template-first queries)
try:
    from template_manager import (
        save_template, load_template, list_templates, delete_template,
        get_template_info, normalize_name, validate_name,
    )
    from template_matcher import find_template
    HAS_TEMPLATES = True
except ImportError:
    HAS_TEMPLATES = False

# Browser annotation tool (needs fastapi + uvicorn)
try:
    from annotation_server import AnnotationServer
    HAS_ANNOTATE = True
except ImportError:
    HAS_ANNOTATE = False

# Intent -> locate -> act modules (optional; the rest of the CLI works without them)
try:
    from intent_classifier import classify_intent
    from layout_heuristics import (
        predict_search_box_location, predict_submit_button_location,
        predict_navigation_location, predict_close_button_location, predict_menu_button_location,
    )
    from ocr_engine import find_all_text_elements
    import context_memory
    import action_executor
    HAS_ACTIONS = True
except ImportError:
    HAS_ACTIONS = False

TEMPLATE_THRESHOLD = 0.8   # minimum OpenCV match confidence for a trained template
YOLO_CONFIDENCE_THRESHOLD = 0.5  # reject weak icon detections before reporting or acting
ANNOTATE_TIMEOUT = 600    # seconds to wait for the browser before giving up


def _normalized_match_text(value: str) -> str:
    """Normalize a label for matching regardless of spaces or punctuation."""
    return "".join(character for character in (value or "").casefold() if character.isalnum())


def _atspi_location(name: str, image_path: Path) -> Optional[Dict]:
    """Resolve an accessible named element to image-relative coordinates, when possible."""
    if not HAS_ATSPI or not name:
        return None
    try:
        element = atspi_bridge.find_element_by_name(name)
        role = _normalized_match_text(element.get("role", "")) if element else ""
        actionable_roles = ("button", "menuitem", "link", "tab", "pushbutton", "togglebutton")
        if not element or not any(actionable in role for actionable in actionable_roles):
            return None
        bounds = element.get("bounds")
        if not bounds:
            return None
        with Image.open(image_path) as image:
            width, height = image.size
        x, y, element_width, element_height = bounds
        if width <= 0 or height <= 0 or element_width <= 0 or element_height <= 0:
            return None
        center_x, center_y = x + element_width / 2, y + element_height / 2
        if not (0 <= center_x < width and 0 <= center_y < height):
            return None
        return {
            "x_pct": center_x / width,
            "y_pct": center_y / height,
            "confidence": 0.99,
            "source": "atspi",
            "label": element.get("name") or name,
            "ref": element.get("ref"),
        }
    except Exception as e:
        print(f"   [AT-SPI] unavailable for this query: {e}")
        return None


def _reliable_yolo_detection(detection: Optional[Dict], image_path: Path) -> bool:
    """Reject low-confidence or screen-container YOLO boxes before using their centres."""
    if not detection:
        return False
    confidence = float(detection.get("confidence", 0.0))
    if confidence < YOLO_CONFIDENCE_THRESHOLD:
        print(f"YOLO detection confidence too low ({confidence * 100:.0f}%), skipping")
        return False
    bbox = detection.get("bbox") or detection
    box_width = bbox.get("width")
    box_height = bbox.get("height")
    if box_width is not None and box_height is not None:
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            if box_width > 0.3 * image_width and box_height > 0.1 * image_height:
                print("YOLO detection is too large to be a reliable icon, skipping")
                return False
        except (OSError, ValueError):
            pass
    return True


def _ocr_symbol_candidate(image_path: Path, target: str, kind: Optional[str] = None) -> Optional[Dict]:
    """Use nearby OCR-recognized glyphs instead of clicking a label for an icon request."""
    text = (target or "").casefold()
    is_icon = kind == "icon" or "icon" in text
    if "new tab" in text and (is_icon or text.strip() == "new tab"):
        reference_text, symbols, horizontal_limit, region = "new tab", {"+", "＋"}, 0.15, None
    elif "close" in text:
        reference_text, symbols, horizontal_limit, region = "close", {"x", "×", "✕", "✖"}, 0.1, lambda x, y: y <= 0.15 and x >= 0.5
    elif "menu" in text:
        reference_text, symbols, horizontal_limit, region = "menu", {"≡", "☰", "|||", "⋮"}, 0.1, lambda x, y: y <= 0.15 and x <= 0.3
    elif "search" in text and is_icon:
        reference_text, symbols, horizontal_limit, region = "search", {"🔍", "⌕"}, 0.15, None
    else:
        return None

    try:
        words = extract_text_with_coordinates(image_path, resize_max=1600)
    except OCRError:
        return None
    normalized_ref = _normalized_match_text(reference_text)
    ref_words = [word for word in words if normalized_ref and normalized_ref in _normalized_match_text(word["text"])]
    if reference_text == "new tab" and not ref_words:
        new_words = [word for word in words if _normalized_match_text(word["text"]) in {"new", "tab"}]
        ref_words = new_words if len(new_words) >= 2 else []
        if not ref_words:
            return None
    if ref_words:
        ref_x = sum(word["center"][0] for word in ref_words) / len(ref_words)
        ref_y = sum(word["center"][1] for word in ref_words) / len(ref_words)
    else:
        ref_x = ref_y = None

    candidates = []
    for word in words:
        symbol = word["text"].strip().casefold()
        x, y = word["center"]
        if symbol not in symbols:
            continue
        if region and not region(x, y):
            continue
        if ref_x is not None and (abs(x - ref_x) >= horizontal_limit or abs(y - ref_y) >= 0.05):
            continue
        if reference_text == "new tab" and x <= ref_x:
            continue
        candidates.append(word)
    if not candidates:
        return None
    if ref_x is not None:
        candidate = min(candidates, key=lambda word: (
            abs(word["center"][0] - ref_x) + abs(word["center"][1] - ref_y)
        ))
    elif reference_text == "close":
        candidate = max(candidates, key=lambda word: word["center"][0])
    else:
        anchor_x, anchor_y = (0.03, 0.04) if reference_text == "menu" else (0.5, 0.08)
        candidate = min(candidates, key=lambda word: (
            abs(word["center"][0] - anchor_x) + abs(word["center"][1] - anchor_y)
        ))
    x, y = candidate["center"]
    return {
        "x_pct": x, "y_pct": y, "confidence": candidate["conf"] / 100,
        "source": "ocr", "label": candidate["text"],
    }


def _suggest_annotation() -> None:
    print("   Use /annotate or /teach to train unknown elements.")


def detect_image_mime(file_path: Path) -> Optional[str]:
    """Detect image MIME type from file header (magic bytes) or extension."""
    if not file_path.is_file() or file_path.name.startswith("."):
        return None
    try:
        with open(file_path, "rb") as f:
            header = f.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif header.startswith(b"RIFF") and b"WEBP" in header[:16]:
            return "image/webp"
        elif header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return "image/gif"
        elif header.startswith(b"BM"):
            return "image/bmp"
    except Exception:
        pass

    ext = file_path.suffix.lower()
    if ext == ".png":
        return "image/png"
    elif ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    elif ext == ".webp":
        return "image/webp"
    elif ext == ".bmp":
        return "image/bmp"
    return None


def is_image_file(file_path: Path) -> bool:
    """Check if a file is a valid image by inspecting its header and extension."""
    return detect_image_mime(file_path) is not None


def get_image_list(images_dir: Path) -> List[Path]:
    """Find and return all base image files (excluding generated grid/target previews)."""
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        return []

    images = [
        p for p in images_dir.iterdir()
        if is_image_file(p) and not p.name.startswith("grid_") and not p.name.startswith("target_")
    ]

    # Natural sort: 1, 2, 10
    def sort_key(p: Path):
        name = p.name.lower()
        if name.isdigit():
            return (0, int(name), name)
        return (1, 0, name)

    images.sort(key=sort_key)
    return images


def get_first_image(images_dir: Path) -> Optional[Path]:
    """Return the first image alphabetically/numerically from images_dir."""
    images = get_image_list(images_dir)
    return images[0] if images else None


def image_to_base64(image_path: Path, max_dimension: int = 1920) -> Tuple[str, str]:
    """
    Read image, optionally resize if excessively large to keep memory low,
    and return (mime_type, base64_str).
    """
    mime = detect_image_mime(image_path) or "image/png"

    # If Pillow is available, ensure we don't blow up memory on 4K+ retina images
    if HAS_PIL:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if max(w, h) > max_dimension:
                    scale = max_dimension / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    from io import BytesIO
                    buf = BytesIO()
                    fmt = "JPEG" if "jpeg" in mime else "PNG"
                    resized.convert("RGB").save(buf, format=fmt, quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    return mime, b64
        except Exception:
            pass

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return mime, b64


class Spinner:
    """Animated status line shown while a blocking call runs (only in a real terminal)."""

    FRAMES = "|/-\\"

    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._t0 = 0.0

    def _spin(self) -> None:
        i = 0
        while not self._stop.wait(0.1):
            elapsed = time.time() - self._t0
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r{frame} {self.message} {elapsed:.1f}s")
            sys.stdout.flush()
            i += 1

    def __enter__(self) -> "Spinner":
        self._t0 = time.time()
        if sys.stdout.isatty():
            self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        if self._thread.is_alive():
            self._stop.set()
            self._thread.join()
            sys.stdout.write("\r" + " " * 79 + "\r")
            sys.stdout.flush()
        return False


def query_ollama_moondream(image_path: Path, prompt: str, timeout: int = 120) -> Tuple[str, Dict[str, float]]:
    """Query local Moondream model via Ollama REST API. Returns (answer, timing stats)."""
    _, b64_img = image_to_base64(image_path)
    url = "http://127.0.0.1:11434/api/generate"
    payload = json.dumps({
        "model": "moondream",
        "prompt": prompt,
        "images": [b64_img],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 256,
        }
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    ns = 1e9  # Ollama reports durations in nanoseconds
    stats = {
        "total": elapsed,
        "load": data.get("load_duration", 0) / ns,  # >0 only when the model had to be loaded
        "prompt": data.get("prompt_eval_duration", 0) / ns,
        "generate": data.get("eval_duration", 0) / ns,
        "tokens": data.get("eval_count", 0),
    }
    return data.get("response", "").strip(), stats


def format_bytes(size: int) -> str:
    """Format file size in human-readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def auto_mark_target(image_path: Path, images_dir: Path, x_pct: float, y_pct: float, label: str = "TARGET") -> None:
    """Automatically mark a target on the image without requiring /mark command."""
    if not HAS_PIL:
        print("   [Warning] Pillow not available, skipping auto-mark")
        return
    
    if not image_path or not image_path.exists():
        print(f"   [Warning] Image path invalid or does not exist: {image_path}")
        return
    
    try:
        out_target = images_dir / f"target_{image_path.name}.png"
        draw_target_marker(image_path, x_pct, y_pct, out_target, label=label)
        print(f"   Target automatically marked at ({x_pct*100:.1f}%, {y_pct*100:.1f}%)!")
        print(f"   Preview saved to: {out_target}\n")
    except Exception as e:
        print(f"   [Error] Failed to mark target: {e}\n")


def confirm_overwrite(name: str) -> bool:
    """Return True if the template may be written: it is new, or the user confirmed the overwrite."""
    if load_template(name) is None:
        return True
    try:
        answer = input(f"Template '{name}' already exists. Overwrite? (y/n) ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def print_template_saved(name: str, rel_path: str, x_pct: float, y_pct: float) -> None:
    info = get_template_info(name) or {}
    size = info.get("crop_size", "?")
    print(f"✓ Saved template: {rel_path}")
    print("✓ Saved metadata: trained_ui/button_locations.json")
    print(f"   Template saved from ({x_pct*100:.1f}%, {y_pct*100:.1f}%) - crop size: {size}x{size}px")


def run_annotate_session(image_path: Path) -> None:
    """Serve the annotation UI, wait for the browser to save or cancel, and handle the result here."""
    server = AnnotationServer(image_path)
    try:
        server.start()
    except RuntimeError as e:
        print(f"[Error] {e}")
        return

    url = server.url
    print(f"Opening annotation tool at {url}...")
    print(f"   Image: {image_path.name}  (Ctrl+C to cancel)")
    try:
        if not webbrowser.open(url):
            print(f"   Could not open a browser automatically. Open this URL manually: {url}")
    except Exception as e:
        print(f"   Browser preview failed ({e}). Open this URL manually: {url}")

    keep_running = False  # True once a template is saved; the server then shuts itself down
    try:
        deadline = time.time() + ANNOTATE_TIMEOUT
        while time.time() < deadline:
            sub = server.next_submission(timeout=0.5)
            if sub is None:
                continue
            if sub.kind == "cancel":
                print("Annotation cancelled.\n")
                break
            if not _is_static_template_target(sub.name):
                message = "Only known static logos or fixed buttons can be trained; use OCR/AT-SPI for dynamic elements."
                sub.respond({"status": "declined", "message": message})
                print(f"Not saved: {message}")
                continue
            if not confirm_overwrite(sub.name):
                sub.respond({
                    "status": "declined",
                    "message": f"Not saved: '{sub.name}' already exists. Choose another name or click Cancel.",
                })
                print(f"Overwrite of '{sub.name}' declined. Waiting for another submission...")
                continue
            try:
                rel = save_template(image_path, sub.name, sub.x_pct, sub.y_pct, sub.crop_size)
            except (ValueError, OSError) as e:
                sub.respond({"status": "error", "message": f"Save failed: {e}"})
                print(f"[Error] Failed to save template: {e}")
                continue
            sub.respond({"status": "success", "template_path": rel, "message": "Template saved successfully"})
            print_template_saved(sub.name, rel, sub.x_pct, sub.y_pct)
            print()
            keep_running = True
            server.schedule_stop(5.0)
            break
        else:
            print("Timed out waiting for the browser. Annotation cancelled.\n")
    except KeyboardInterrupt:
        print("\nAnnotation cancelled.\n")
    finally:
        if not keep_running:
            server.stop()


def find_trained_template(query: str, names: List[str]) -> Optional[str]:
    """Return the longest template label contained in a query, ignoring spaces and punctuation."""
    phrase = _normalized_match_text(extract_search_phrase(query))
    compact_query = _normalized_match_text(query)
    hits = []
    for name in names:
        normalized_name = _normalized_match_text(name)
        if not normalized_name:
            continue
        # The extracted target may omit a generic kind ("new tab" for "new tab icon").
        query_tokens = [token for token in re.findall(r"[a-z0-9]+", query.casefold())
                        if token not in {"please", "click", "tap", "press", "hit", "select", "find", "locate",
                                         "where", "is", "are", "the", "a", "an", "button", "icon", "link", "tab", "menu"}]
        name_tokens = [token for token in re.findall(r"[a-z0-9]+", name.casefold())
                       if token not in {"button", "icon", "link", "tab", "menu"}]
        if normalized_name in compact_query or (name_tokens and all(token in query_tokens for token in name_tokens)) or (len(phrase) >= 4 and phrase in normalized_name):
            hits.append(name)
    return max(hits, key=lambda name: len(_normalized_match_text(name))) if hits else None


_DYNAMIC_TEMPLATE_TERMS = re.compile(
    r"\b(?:tabs?|new|close|search|fields?|boxes?|bars?|text|menus?|"
    r"items?|options?|x)\b",
    re.IGNORECASE,
)
_STATIC_TEMPLATE_NAMES = {"ai mode button", "gmail button", "youtube"}
_STATIC_TEMPLATE_TOKENS = {
    "please", "click", "tap", "press", "hit", "select", "find", "locate", "where",
    "is", "are", "the", "a", "an", "button", "icon", "link", "logo", "fixed", "static",
}


def _is_static_template_target(target: str) -> bool:
    """Only known fixed visual controls are eligible for matching or training."""
    target_tokens = [token for token in re.findall(r"[a-z0-9]+", (target or "").casefold())
                     if token not in _STATIC_TEMPLATE_TOKENS]
    static_names = [name.casefold().split() for name in _STATIC_TEMPLATE_NAMES]
    return bool(target_tokens) and any(
        target_tokens == [token for token in name if token not in {"button", "icon", "link"}]
        or target_tokens == [name[0]]
        for name in static_names
    )


def _clean_dynamic_templates() -> None:
    """Remove legacy templates for known dynamic items from the prior strategy."""
    if not HAS_TEMPLATES:
        return
    for name in list_templates():
        if _is_dynamic_template_target(name):
            delete_template(name)


def _is_dynamic_template_target(target: str) -> bool:
    """Dynamic layout targets must use accessible/OCR location, never trained templates."""
    return bool(_DYNAMIC_TEMPLATE_TERMS.search(target or ""))


def run_template_query(query: str, image_path: Path, images_dir: Path) -> bool:
    """
    Try a matching static template. Returns True if a template matched and was printed;
    False means the caller should fall through to the next detection layer.
    """
    if not HAS_TEMPLATES or _is_dynamic_template_target(query):
        return False
    target = extract_search_phrase(query)
    if not _is_static_template_target(target):
        return False
    names = list_templates()
    name = find_trained_template(query, names) if names else None
    if name is None or not _is_static_template_target(name):
        return False

    # OCR is preferred, so this is only reached after text/spatial OCR missed.
    # Keep template matching for static visual elements whose appearance is stable.
    print(f"Checking trained templates... Found \"{name}\"")
    template_path = load_template(name)
    if template_path is None:
        print(f"   Template file for '{name}' is missing; skipping.")
        return False

    with Spinner(f"Matching '{name}' template..."):
        match = find_template(image_path, template_path, threshold=TEMPLATE_THRESHOLD)
    if match is None:
        print(f"   No confident match for '{name}' (threshold {TEMPLATE_THRESHOLD*100:.0f}%).")
        print("   Falling back to OCR / YOLO...")
        return False

    x_pct, y_pct = match["x_pct"] * 100, match["y_pct"] * 100
    print(f"\nFound '{name}' with {match['confidence']*100:.0f}% confidence (template match)")
    print(f"   Would click at ({x_pct:.1f}%, {y_pct:.1f}%)")
    auto_mark_target(image_path, images_dir, match["x_pct"], match["y_pct"], name)
    return True


def run_hybrid_query(query: str, image_path: Path, images_dir: Path, first_run: bool = False) -> None:
    """Locate in order: AT-SPI, OCR/spatial heuristics, static templates, then YOLO/vision."""
    _clean_dynamic_templates()
    target_name = extract_search_phrase(query)
    # Icon requests should not resolve to a nearby text label through accessibility.
    icon_request = bool(re.search(r"\bicon\b", query, re.IGNORECASE))
    accessible = None if icon_request else _atspi_location(target_name, image_path)
    if accessible:
        x_pct, y_pct = accessible["x_pct"], accessible["y_pct"]
        print(f"Found '{accessible['label']}' via atspi")
        print(f"   Would click at ({x_pct * 100:.1f}%, {y_pct * 100:.1f}%)")
        auto_mark_target(image_path, images_dir, x_pct, y_pct, accessible["label"])
        return

    # Search the requested text first; refine it to a nearby icon only for
    # icon-oriented queries where clicking a text label would be wrong.
    phrase = target_name
    hit = None
    if phrase:
        try:
            with Spinner("Running OCR..."):
                hit = find_text_element(image_path, phrase)
        except OCRError as e:
            print(f"[Error] OCR failed: {e}")
    else:
        print("Warning: Could not find a label to search for in that request.")

    symbol = _ocr_symbol_candidate(image_path, query)
    if symbol:
        x_pct, y_pct = symbol["x_pct"], symbol["y_pct"]
        print(f"\nFound '{symbol['label']}' via spatial OCR")
        print(f"   Would click at ({x_pct * 100:.1f}%, {y_pct * 100:.1f}%)")
        auto_mark_target(image_path, images_dir, x_pct, y_pct, symbol["label"])
        return

    if hit is not None and not icon_request:
        x_pct, y_pct = hit["center_pct"]
        print(f"\nFound '{hit['text']}' (OCR confidence {hit['conf']:.0f}%)")
        print(f"   Would click at ({x_pct:.1f}%, {y_pct:.1f}%)")
        auto_mark_target(image_path, images_dir, x_pct, y_pct, hit["text"])
        return
    if phrase and hit is None:
        print(f"OCR could not find '{phrase}' on screen.")

    # Tabs, search controls, menu items and similar targets move with the layout.
    # Never use a crop trained at one position for these queries.
    if run_template_query(query, image_path, images_dir):
        return

    try:
        with Spinner("Loading TEV1 router..." if first_run else "Routing request..."):
            route = route_task(query)
        method = route["method"]
        print(f"TEV1 route: {method} (confidence {route['confidence'] * 100:.1f}%)")
    except TEV1Error as e:
        print(f"Warning: TEV1 routing failed ({e}); defaulting to YOLO fallback.")
        method = "yolo"

    if method == "vision":
        print("Note: Vision answers are not wired into hybrid mode yet.")
        print("   Ask about a visible label instead, e.g. 'Where is the <label> button?'")
        return

    # OCR already ran above; route icon-only and otherwise-unresolved targets to YOLO.
    try:
        with Spinner("Running YOLO icon detection..." if method == "yolo" else "Running YOLO fallback..."):
            detector = YOLODetector()
            lowered_query = query.casefold()
            icon_type = "top_left" if "menu" in lowered_query else "top_bar"
            detection = detector.find_icon_by_position(str(image_path), icon_type=icon_type)

        if detection is not None:
            if not _reliable_yolo_detection(detection, image_path):
                print("   No reliable match found.")
                _suggest_annotation()
                return
            x_pct, y_pct = detection["center_pct"]["x"], detection["center_pct"]["y"]
            print(f"\nFound icon (YOLO confidence {detection['confidence'] * 100:.0f}%)")
            print(f"   Would click at ({x_pct:.1f}%, {y_pct:.1f}%)")
            auto_mark_target(image_path, images_dir, x_pct / 100, y_pct / 100, "icon")
            return
        print("YOLO found no matching icons in the image.")
        if icon_request:
            print("   No reliable match found.")
            _suggest_annotation()
        else:
            print("   Use /grid to inspect the screen manually.")
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        if icon_request:
            print("   No reliable match found.")
            _suggest_annotation()
        else:
            print("   Use /grid to inspect the screen manually.")


# ---------------------------------------------------------------------------
# Action pipeline: classify -> locate (template, OCR, YOLO, heuristics) -> act
# ---------------------------------------------------------------------------

LIVE_DIR = Path(__file__).resolve().parent / ".cache"  # gitignored scratch space for live screenshots
CLICK_EFFECT_THRESHOLD = 1.0  # mean grey-level change that counts as "the click did something"
PIPELINE_BUDGET = 2.0         # seconds; target response time for common actions
_yolo_instance = None
_yolo_rejected_unreliable = False


def _safe(fn, label: str):
    """Run one detection layer. A failing layer is reported and skipped, never fatal."""
    try:
        return fn()
    except Exception as e:
        print(f"   [{label}] failed: {e}")
        return None


def _capture_live(name: str) -> Optional[Path]:
    path = LIVE_DIR / name
    return path if action_executor.capture_screen(path) else None


def _ocr_candidates(image: Path, target: str) -> List[Dict]:
    variants = [target]
    stripped = re.sub(r"\s+(menu|box|field|bar|button|icon|tab|link)$", "", target, flags=re.IGNORECASE).strip()
    if stripped and stripped != target:
        variants.append(stripped)
    for phrase in variants:
        hits = find_all_text_elements(image, phrase)
        if hits:
            return [{
                "x_pct": h["center_x"], "y_pct": h["center_y"],
                "confidence": h["conf"] / 100, "source": "ocr", "label": h["text"],
            } for h in hits]
    return []


def _yolo_candidate(image: Path, target: str) -> Optional[Dict]:
    global _yolo_instance, _yolo_rejected_unreliable
    model = Path(__file__).resolve().parent / "models" / "onnx" / "model.onnx"
    if not HAS_HYBRID or not model.exists():
        return None
    t = target.lower()
    icon_type = "top_left" if "menu" in t else "top_right" if "close" in t else "top_bar"
    if _yolo_instance is None:
        _yolo_instance = YOLODetector(str(model))
    detection = _yolo_instance.find_icon_by_position(str(image), icon_type=icon_type)
    if detection and not _reliable_yolo_detection(detection, image):
        _yolo_rejected_unreliable = True
        return None
    if not detection:
        return None
    return {
        "x_pct": detection["center_pct"]["x"] / 100, "y_pct": detection["center_pct"]["y"] / 100,
        "confidence": detection["confidence"], "source": "yolo", "label": target,
    }


def _heuristic_candidate(image: Path, target: str, ref: Optional[Dict]) -> Optional[Dict]:
    t = target.lower()
    anchor = ref or {"x_pct": 0.5, "y_pct": 0.5}
    if "close" in t:
        return predict_close_button_location(image)
    if "menu" in t:
        return predict_menu_button_location(image)
    if re.search(r"submit|send|login|log in|sign in|\bok\b|continue|save|\bgo\b", t):
        return predict_submit_button_location(image, anchor)
    if "search" in t:
        return predict_search_box_location(image)
    if re.search(r"\b(home|back|forward|settings|file|edit|view|help|tools)\b", t):
        return predict_navigation_location(image, "top_bar")
    return None


def locate_target(target: str, image: Path, ref: Optional[Dict] = None,
                  kind: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
    """
    Find a target on screen. Layers, in order: trained template, OCR, YOLO, layout heuristics,
    then the last known location from context memory. Returns (candidates, source).
    """
    accessible = _atspi_location(target, image)
    if accessible and not (kind == "icon" or "icon" in target.casefold()):
        return [accessible], "atspi"

    symbol = _safe(lambda: _ocr_symbol_candidate(image, target, kind), "spatial OCR")
    if symbol:
        return [symbol], "ocr"
    # Icon-only requests must not degrade to clicking a neighboring text label.
    if kind != "icon":
        cands = _safe(lambda: _ocr_candidates(image, target), "OCR")
        if cands:
            return cands, "ocr"
    if HAS_TEMPLATES and _is_static_template_target(target):
        names = list_templates()
        name = find_trained_template(target, names) if names else None
        if name and _is_static_template_target(name):
            template = load_template(name)
            match = find_template(image, template, threshold=TEMPLATE_THRESHOLD) if template else None
            if match:
                return [{"x_pct": match["x_pct"], "y_pct": match["y_pct"],
                         "confidence": match["confidence"], "source": "template", "label": name}], "template"
    global _yolo_rejected_unreliable
    _yolo_rejected_unreliable = False
    cand = _safe(lambda: _yolo_candidate(image, target), "YOLO")
    if cand:
        return [cand], "yolo"
    if _yolo_rejected_unreliable or kind == "icon":
        return [], None
    cand = _safe(lambda: _heuristic_candidate(image, target, ref), "heuristic")
    if cand:
        return [cand], "heuristic"
    known = context_memory.get_known_location(target)
    if known:
        return [{"x_pct": known[0], "y_pct": known[1], "confidence": 0.5,
                 "source": "context", "label": target}], "context"
    return [], None


def _last_location() -> Optional[Dict]:
    """Position of the most recent successful action; used as the reference for spatial reasoning."""
    for action in reversed(context_memory.get_recent_actions(10)):
        if action.get("location"):
            x, y = action["location"]
            return {"x_pct": x, "y_pct": y}
    return None


def _choose_candidate(cands: List[Dict], ref: Optional[Dict], execute: bool, interactive: bool) -> Optional[Dict]:
    """Pick one of several matches: nearest to the last action, else ask the user, else the first."""
    if len(cands) == 1:
        return cands[0]
    if ref is not None:
        best = min(cands, key=lambda c: math.hypot(c["x_pct"] - ref["x_pct"], c["y_pct"] - ref["y_pct"]))
        print(f"   {len(cands)} matches; using the one nearest the last action.")
        return best
    print(f"   {len(cands)} matches found:")
    for i, c in enumerate(cands, 1):
        print(f"    {i}. '{c['label']}' at ({c['x_pct'] * 100:.1f}%, {c['y_pct'] * 100:.1f}%)")
    if not (execute and interactive):
        print("   (dry run: showing the first match)")
        return cands[0]
    try:
        answer = input("   Which one? [1] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not answer:
        return cands[0]
    if answer.isdigit() and 1 <= int(answer) <= len(cands):
        return cands[int(answer) - 1]
    print("   Not a valid choice; cancelled.")
    return None


def _maybe_learn(target: str, loc: Dict, image: Path, source: str, learn: bool, interactive: bool) -> None:
    """Save only non-dynamic targets as templates (auto in /learn mode, else offered)."""
    if source in ("template", "context") or not HAS_TEMPLATES:
        return
    try:
        name = validate_name(target)
    except ValueError:
        return
    if not _is_static_template_target(name):
        return
    if load_template(name) is not None:
        return  # never overwrite a template silently
    if learn:
        save = True
    elif interactive:
        try:
            save = input(f"   Save '{name}' as a template for next time? (y/n) ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            print()
            save = False
    else:
        save = False
    if save:
        try:
            rel = save_template(image, name, loc["x_pct"], loc["y_pct"])
            print_template_saved(name, rel, loc["x_pct"], loc["y_pct"])
        except (ValueError, OSError) as e:
            print(f"   [Error] Failed to save template: {e}")


def _after_changed(before: Path) -> bool:
    """Verify a click: did the screen change? Unverifiable (no capture) counts as success."""
    after = _capture_live("after_action.png")
    if after is None:
        return True
    diff = context_memory.image_difference(before, after)
    return diff is not None and diff > CLICK_EFFECT_THRESHOLD


def _note_page_change() -> None:
    shot = _capture_live("after_action.png")
    if shot is not None and context_memory.detect_page_change(shot):
        print("   Page changed: cleared form state and known locations.")


def _run_step(intent: Dict, image: Optional[Path], images_dir: Path, execute: bool, learn: bool, interactive: bool) -> bool:
    """Run one classified action. Dry run (execute=False) only reports what would happen."""
    if intent.get("needs_clarification"):
        print(f"   {intent['needs_clarification']}")
        return False
    kind = intent["action"]
    target = intent.get("target")
    params = intent.get("params") or {}

    if kind == "UNKNOWN":
        print("   I don't recognise that action.")
        return False

    if kind == "WAIT":
        seconds = params["seconds"]
        if execute:
            action_executor.wait(seconds)
        print(f"   {'Waited' if execute else 'Would wait'} {seconds:g}s")
        return True

    if kind == "SCROLL":
        direction, amount = params["direction"], params["amount"]
        if not execute:
            print(f"   Would scroll {direction} ({amount})")
            return True
        ok = action_executor.scroll(direction, amount)
        print(f"   {'Scrolled' if ok else 'Scroll failed'} {direction}.")
        return ok

    if kind == "NAVIGATE":
        url = params.get("url")
        if not url:
            print("   No URL given.")
            return False
        if not execute:
            print(f"   Would open {url}")
            return True
        ok = action_executor.navigate_to(url)
        if ok:
            context_memory.update_context(intent)
        print(f"   {'Opened' if ok else 'Could not open'} {url}")
        return ok

    if kind == "TYPE" and params.get("key"):
        key = params["key"]
        if not execute:
            print(f"   Would press {key}")
            return True
        return action_executor.press_key(key)

    if kind == "TYPE" and not params.get("text"):
        print("   Nothing to type.")
        return False

    if kind == "TYPE" and not target:
        text = params["text"]
        if not execute:
            print(f"   Would type {text!r} into the focused field")
            return True
        ok = action_executor.type_text(text)
        if ok:
            context_memory.update_context(intent)
        return ok

    if kind not in ("CLICK", "TYPE", "DRAG"):
        print(f"   Unsupported action: {kind}")
        return False
    if image is None:
        print("   No screen image to search.")
        return False

    ref = _last_location()
    cands, source = locate_target(target, image, ref, kind=params.get("kind"))
    if not cands:
        if _yolo_rejected_unreliable:
            print("   No reliable match found.")
            _suggest_annotation()
            return False
        print(f"   Could not find '{target}' on screen (OCR, YOLO, heuristics all failed).")
        print("   Use /annotate to train it, or /grid to inspect the screen.")
        return False
    loc = _choose_candidate(cands, ref, execute, interactive)
    if loc is None:
        return False
    x, y = loc["x_pct"], loc["y_pct"]
    print(f"   Found '{target}' via {source} ({loc['confidence'] * 100:.0f}% confidence) at ({x * 100:.1f}%, {y * 100:.1f}%)")

    # Update visual marker for dry run mode
    if image and not execute:
        try:
            auto_mark_target(image, images_dir, x, y, loc.get("label", target))
        except Exception as e:
            print(f"   [Warning] Could not update visual marker: {e}")

    if params.get("locate_only"):
        print("   Located only (question form); nothing clicked.")
        return True
    if not execute:
        verb = {"CLICK": "click", "TYPE": "type into", "DRAG": "drag from"}[kind]
        print(f"   Would {verb} ({x * 100:.1f}%, {y * 100:.1f}%)")
        return True
    if source == "heuristic" and interactive:
        try:
            answer = input("   Only a layout guess. Act on it? (y/n) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer not in ("y", "yes"):
            print("   Cancelled.")
            return False

    if kind == "CLICK":
        ok = action_executor.click_with_retry(
            x, y, verify=lambda: _after_changed(image), accessible_ref=loc.get("ref"))
    elif kind == "TYPE":
        ok = action_executor.type_text(params["text"], x, y)
    else:
        destination = params.get("destination")
        dest_cands, _ = locate_target(destination, image, loc)
        dest = _choose_candidate(dest_cands, loc, execute, interactive) if dest_cands else None
        if dest is None:
            print(f"   Could not find drop target '{destination}'.")
            return False
        ok = action_executor.drag((x, y), (dest["x_pct"], dest["y_pct"]))
    if not ok:
        print("   Action failed.")
        return False

    context_memory.update_context({"action": kind, "target": target, "params": params,
                                   "location": {"x_pct": x, "y_pct": y}})
    _note_page_change()
    _maybe_learn(target, loc, image, source, learn, interactive)
    return True


def run_action_pipeline(intent: Dict, base_image: Optional[Path], images_dir: Path, execute: bool, learn: bool) -> bool:
    """
    Run a classified request. With execute=False this is a dry run: it locates targets in
    `base_image` and reports them without moving the mouse or typing.
    With execute=True each locating step captures a fresh screenshot of the live display.
    """
    t0 = time.time()
    interactive = sys.stdin.isatty()
    steps = intent["params"]["steps"] if intent["action"] == "MULTISTEP" else [intent]
    ok = True
    for i, step in enumerate(steps, 1):
        if len(steps) > 1:
            print(f"Step {i}/{len(steps)}: {step['action']} {step.get('target') or ''}".rstrip())
        image = base_image
        if execute and step["action"] in ("CLICK", "TYPE", "DRAG"):
            image = _capture_live("live_screen.png")
            if image is None:
                print("[Error] Could not capture the screen; nothing was done.")
                return False
        ok = _run_step(step, image, images_dir, execute, learn, interactive)
        if not ok:
            print("Stopped: that step failed.")
            break
    elapsed = time.time() - t0
    note = "" if elapsed < PIPELINE_BUDGET else f"  (over the {PIPELINE_BUDGET:.0f}s budget)"
    print(f"Pipeline took {elapsed:.2f}s{note}")
    return ok


def print_context() -> None:
    ctx = context_memory.get_context()
    print(f"Current page: {ctx['current_page'] or '(unknown)'}")
    actions = ctx["last_actions"]
    print(f"Recent actions ({len(actions)}):")
    for a in actions:
        detail = a.get("target") or a.get("url") or ""
        if a.get("text") is not None:
            detail += f" text={a['text']!r}"
        print(f"  - {a['action']} {detail}".rstrip())
    if not actions:
        print("  (none)")
    print(f"Form state: {ctx['form_state'] or '(empty)'}")
    print(f"Known locations: {len(ctx['known_locations'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Vision Agent Harness CLI v0.2 - CPU-Optimized for Raspberry Pi 5"
    )
    parser.add_argument(
        "--backend", "-b", type=str, choices=["moondream", "llamacpp", "hybrid"], default="moondream",
        help="Inference engine: 'moondream' (Ollama 4-bit), 'llamacpp' (SmolVLM), or 'hybrid' (Tesseract OCR + TEV1 routing)"
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Path to specific image file"
    )
    parser.add_argument(
        "--images-dir", type=str, default="images",
        help="Directory containing images (default: 'images')"
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=4,
        help="Number of CPU threads to use (default: 4 for RPi 5)"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually move the mouse and type for hybrid action requests (default: dry run)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    images_dir = (project_root / args.images_dir).resolve()
    models_dir = (project_root / "models").resolve()

    # 1. Resolve image
    selected_image: Optional[Path] = None
    if args.image:
        custom_path = Path(args.image).resolve()
        if custom_path.exists() and custom_path.is_file():
            selected_image = custom_path
        else:
            print(f"[Error] Specified image not found: {args.image}")
            sys.exit(1)
    else:
        selected_image = get_first_image(images_dir)

    base_image = selected_image  # Keep reference to original clean image
    current_mark: Optional[Dict] = None  # Last /mark target, used by /teach
    grid_active = False
    model_used = False  # Becomes True after the first inference query
    learn_mode = False  # /learn: save located targets as templates automatically after success
    execute_mode = bool(args.execute and HAS_ACTIONS)  # False = dry run
    if args.execute and not HAS_ACTIONS:
        print("[Error] Action modules unavailable; staying in dry-run mode.")

    # 2. Setup backend
    print("=" * 65)
    print("  Vision Agent Harness CLI (v0.2)")
    print("=" * 65)
    print(f" • Backend:     {args.backend.upper()}")

    llm = None
    if args.backend == "llamacpp":
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MTMDChatHandler
        except ImportError as e:
            print(f"[Error] Failed to import llama_cpp: {e}")
            sys.exit(1)

        model_path = models_dir / "SmolVLM-256M-Instruct-Q8_0.gguf"
        mmproj_path = models_dir / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

        if not model_path.exists() or not mmproj_path.exists():
            print(f"[Error] GGUF model files not found in {models_dir}.")
            sys.exit(1)

        print("\nInitializing llama-cpp-python vision engine...")
        t0 = time.time()
        chat_handler = MTMDChatHandler(clip_model_path=str(mmproj_path), verbose=False, use_gpu=False)
        llm = Llama(model_path=str(model_path), chat_handler=chat_handler, n_ctx=2048, n_threads=args.threads, verbose=False)
        print(f"Model loaded in {time.time() - t0:.2f}s!")
    elif args.backend == "hybrid":
        if not HAS_HYBRID:
            print("[Error] ocr_engine / tev1_router could not be imported.")
            sys.exit(1)
        print(" • Model:       Hybrid (Tesseract OCR + TEV1 router - Ollama localhost:11434, TEV1 loads on first message)")
        if not shutil.which("tesseract"):
            print("[Error] tesseract not found. Install with: sudo apt install tesseract-ocr")
            sys.exit(1)
        try:
            tev1_name = check_tev1_available()
        except TEV1Error as e:
            print(f"[Error] TEV1 healthcheck failed: {e}")
            sys.exit(1)
        print(f"TEV1 router ready ({tev1_name}); OCR ready.")
    else:
        print(" • Model:       Moondream2 (1.8B INT4 - Ollama localhost:11434, loads on first message)")
        # Quick healthcheck
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as r:
                pass
        except Exception as e:
            print(f"[Error] Ollama service not reachable on 127.0.0.1:11434: {e}")
            sys.exit(1)
        print("Moondream backend connected and ready!")

    def print_image_status():
        nonlocal selected_image
        if selected_image and selected_image.exists():
            size_str = format_bytes(selected_image.stat().st_size)
            dim_str = ""
            if HAS_PIL:
                try:
                    with Image.open(selected_image) as im:
                        dim_str = f" [{im.width}x{im.height}]"
                except Exception:
                    pass
            grid_indicator = " (GRID OVERLAY ACTIVE)" if grid_active else ""
            print(f"\nActive Image: [{selected_image.name}]{dim_str} ({size_str}){grid_indicator}")
        else:
            print("\nWarning: No image currently found in 'images/' folder.")
            print("   Drop an image into 'images/' and run '/reload'.")

    print_image_status()

    print("\nCommands:")
    print(f"  /grid            - Overlay a {GRID_COLS}x{GRID_ROWS} coordinate grid onto image for coordinate queries")
    print("  /orig            - Revert to original clean image (turn off grid)")
    print("  /mark <x> <y> [l] - Draw target crosshair at coords (0.0-1.0) & save preview")
    print("  /reload          - Rescan images folder and pick the top image")
    print("  /list            - List all images in images/ folder")
    print("  /select <name>   - Select a specific image by name")
    print("  /info            - Show active image info")
    print("  /annotate        - Open browser tool: click a button and name it (saves a template)")
    print("  /teach <name>    - Save the last /mark target as a named template")
    print("  /list-templates  - List trained button templates")
    print("  /delete-template <name> - Delete a trained button template")
    print("  /context         - Show current page, recent actions and form state")
    print("  /clear-context   - Reset the context memory")
    print("  /test-action <q> - Classify a request without executing it")
    print("  /learn           - Toggle auto-learning (save successful targets as templates)")
    print("  /execute on|off  - Enable or disable real mouse/keyboard actions (hybrid mode)")
    print("  /exit, quit      - Exit the CLI\n")

    # Interactive REPL Loop
    while True:
        try:
            label_name = selected_image.name if selected_image else "No Image"
            user_input = input(f"[{label_name}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["/exit", "exit", "/quit", "quit"]:
            print("Exiting. Goodbye!")
            break

        if cmd in ["/reload", "/refresh"]:
            selected_image = get_first_image(images_dir)
            base_image = selected_image
            grid_active = False
            print_image_status()
            continue

        if cmd == "/list":
            imgs = get_image_list(images_dir)
            if not imgs:
                print("No images found in 'images/' folder.")
            else:
                print(f"Base images in '{images_dir.name}/':")
                for idx, img in enumerate(imgs, 1):
                    prefix = ">" if selected_image and img.name == selected_image.name else " "
                    print(f" {prefix} {idx}. {img.name} ({format_bytes(img.stat().st_size)})")
            continue

        if cmd.startswith("/select"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /select <filename>")
                continue
            target_name = parts[1].strip()
            target_path = images_dir / target_name
            if target_path.exists() and target_path.is_file():
                selected_image = target_path
                base_image = selected_image
                grid_active = False
                print_image_status()
            else:
                print(f"File not found in {images_dir}: {target_name}")
            continue

        if cmd == "/grid":
            if not HAS_PIL:
                print("[Error] Pillow is required for grid generation.")
                continue
            if not base_image or not base_image.exists():
                print("[Error] No active image to generate grid on.")
                continue
            grid_file = images_dir / f"grid_{base_image.name}.png"
            create_coordinate_grid(base_image, grid_file)
            selected_image = grid_file
            grid_active = True
            print(f"Grid overlay generated and active: {grid_file.name}")
            print(f"   You can open '{grid_file}' to see the labeled cells (Cols {column_label(0)}-{column_label(GRID_COLS - 1)}, Rows R1-R{GRID_ROWS})!")
            print_image_status()
            continue

        if cmd == "/orig":
            if base_image and base_image.exists():
                selected_image = base_image
                grid_active = False
                print("Reverted to original image.")
                print_image_status()
            continue

        if cmd.startswith("/mark"):
            # Usage: /mark 0.95 0.03 Gemini
            if not HAS_PIL:
                print("[Error] Pillow is required for target marking.")
                continue
            parts = user_input.split()
            if len(parts) < 3:
                print("Usage: /mark <x_percent> <y_percent> [optional_label]")
                print("Example: /mark 0.95 0.03 Gemini")
                continue
            try:
                x_val = float(parts[1])
                y_val = float(parts[2])
                # If entered as whole percentage like 95 3 instead of 0.95 0.03:
                if x_val > 1.0:
                    x_val /= 100.0
                if y_val > 1.0:
                    y_val /= 100.0
                marker_label = " ".join(parts[3:]) if len(parts) > 3 else "TARGET"
                src = base_image if base_image else selected_image
                out_target = images_dir / f"target_{src.name}.png"
                draw_target_marker(src, x_val, y_val, out_target, label=marker_label)
                current_mark = {"x": x_val, "y": y_val, "image": src}
                print(f"\nTarget marked at ({x_val*100:.1f}%, {y_val*100:.1f}%)!")
                print(f"   Saved preview to: {out_target}")
                print("   Open this file in your IDE to verify if the box hits the button!\n")
            except ValueError as ve:
                print(f"Invalid coordinate numbers: {ve}")
            continue

        if cmd == "/context":
            if not HAS_ACTIONS:
                print("[Error] Action modules unavailable.")
            else:
                print_context()
            continue

        if cmd == "/clear-context":
            if HAS_ACTIONS:
                context_memory.clear_context()
                print("Context cleared.")
            continue

        if cmd.startswith("/test-action"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /test-action <query>")
            elif not HAS_ACTIONS:
                print("[Error] Action modules unavailable.")
            else:
                t0 = time.perf_counter()
                intent = classify_intent(parts[1])
                ms = (time.perf_counter() - t0) * 1000
                print(json.dumps(intent, indent=2))
                print(f"Classified in {ms:.2f} ms (nothing executed)")
            continue

        if cmd == "/learn":
            learn_mode = not learn_mode
            print(f"Auto-learn {'ON' if learn_mode else 'OFF'}: successful targets are "
                  f"{'saved as templates automatically' if learn_mode else 'offered for saving'}.")
            continue

        if cmd in ("/execute on", "/execute off"):
            if not HAS_ACTIONS:
                print("[Error] Action modules unavailable.")
            else:
                execute_mode = cmd.endswith("on")
                print(f"Execute mode {'ON: actions move the real mouse and keyboard' if execute_mode else 'OFF: dry run only'}.")
            continue

        if cmd == "/info":
            print_image_status()
            continue

        if cmd == "/annotate":
            if not HAS_ANNOTATE:
                print("[Error] Annotation tool unavailable (fastapi/uvicorn not importable).")
                continue
            if not base_image or not base_image.exists():
                print("No image selected. Use /select or drop an image in images/")
                continue
            run_annotate_session(base_image)
            continue

        if cmd.startswith("/teach"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /teach <name>")
                print("Example: /mark 0.972 0.021 then /teach gemini")
                continue
            if not HAS_TEMPLATES:
                print("[Error] Template manager unavailable.")
                continue
            if current_mark is None:
                print("No target marked. Use /mark <x> <y> first, then /teach <name>.")
                continue
            try:
                name = validate_name(parts[1])
            except ValueError as e:
                print(f"[Error] {e}")
                continue
            if not current_mark["image"].exists():
                print("[Error] The image used for /mark no longer exists.")
                continue
            if not _is_static_template_target(name):
                print("[Error] Only known static logos or fixed buttons can be trained; use OCR/AT-SPI for dynamic elements.")
                continue
            if not confirm_overwrite(name):
                print("Not saved.")
                continue
            try:
                rel = save_template(current_mark["image"], name, current_mark["x"], current_mark["y"])
            except (ValueError, OSError) as e:
                print(f"[Error] Failed to save template: {e}")
                continue
            print_template_saved(name, rel, current_mark["x"], current_mark["y"])
            continue

        if cmd == "/list-templates":
            names = list_templates() if HAS_TEMPLATES else []
            if not names:
                print("No trained templates yet. Use /annotate, or /mark then /teach <name>.")
            else:
                print("Trained templates:")
                for n in names:
                    info = get_template_info(n) or {}
                    print(f"  - {n} ({info.get('crop_size', '?')}px, from {info.get('image_source', '?')})")
            continue

        if cmd.startswith("/delete-template"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /delete-template <name>")
                continue
            if HAS_TEMPLATES and delete_template(parts[1]):
                print(f"Deleted template '{normalize_name(parts[1])}'.")
            else:
                print(f"No template named '{parts[1]}'.")
            continue

        # Inference query
        if not selected_image or not selected_image.exists():
            print("[Error] Cannot query: No image loaded. Drop an image into 'images/' and type '/reload'.")
            continue

        first_run = not model_used
        model_used = True

        if args.backend == "moondream":
            try:
                # If grid is active, add helpful instruction to prompt
                effective_prompt = user_input
                if grid_active:
                    effective_prompt = (
                        f"This image has an overlaid coordinate grid with Columns {column_label(0)}-{column_label(GRID_COLS - 1)} and Rows R1-R{GRID_ROWS}. "
                        f"{user_input} State the grid cell (e.g. J1) or percentage coordinate if applicable."
                    )
                with Spinner("Loading model..." if first_run else "Thinking..."):
                    answer, stats = query_ollama_moondream(selected_image, effective_prompt)
                if stats["load"] >= 1.0:  # Ollama reports tiny nonzero loads for an already-loaded model
                    print(f"\nModel loaded (took {stats['load']:.2f}s)")
                print(f"\nMoondream:\n{answer}\n")
                tokens, gen = stats["tokens"], stats["generate"]
                rate = f", {tokens / gen:.1f} tok/s" if tokens > 1 and gen >= 0.1 else ""
                parts = []
                if stats["load"] >= 1.0:
                    parts.append(f"load {stats['load']:.2f}s")
                parts.append(f"prompt {stats['prompt']:.2f}s")
                parts.append(f"generated {tokens} tokens in {gen:.2f}s{rate}")
                print(f"Response took {stats['total']:.2f}s ({', '.join(parts)})\n")
            except Exception as e:
                print(f"\n[Error during Moondream inference]: {e}\n")

        elif args.backend == "hybrid":
            try:
                t0 = time.time()
                # Always use base_image for detection and marking (not grid overlay)
                target_image = base_image if base_image else selected_image
                intent = classify_intent(user_input) if HAS_ACTIONS else None
                if intent is not None and intent["action"] != "UNKNOWN":
                    run_action_pipeline(intent, target_image, images_dir, execute_mode, learn_mode)
                else:
                    run_hybrid_query(user_input, target_image, images_dir, first_run=first_run)
                print(f"Response took {time.time() - t0:.2f}s\n")
            except Exception as e:
                print(f"\n[Error during hybrid query]: {e}\n")

        else:
            # llama_cpp backend
            try:
                mime, b64_img = image_to_base64(selected_image)
                data_uri = f"data:{mime};base64,{b64_img}"
                t0 = time.time()
                with Spinner("Loading model..." if first_run else "Thinking..."):
                    response = llm.create_chat_completion(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": data_uri}},
                                    {"type": "text", "text": user_input},
                                ],
                            }
                        ],
                        max_tokens=256,
                        temperature=0.2,
                    )
                elapsed = time.time() - t0
                answer = response["choices"][0]["message"]["content"]
                tokens = response.get("usage", {}).get("completion_tokens", 0)
                print(f"\nSmolVLM:\n{answer.strip()}\n")
                print(f"Response took {elapsed:.2f}s (generated {tokens} tokens)\n")
            except Exception as e:
                print(f"\n[Error during SmolVLM inference]: {e}\n")


if __name__ == "__main__":
    main()
