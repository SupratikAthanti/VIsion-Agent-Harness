"""
Template manager for trained UI buttons.

Crops square templates around a click point (stored as 0.0-1.0 fractions of the
image) and keeps a small metadata file describing them:

    trained_ui/
        button_locations.json   {"gemini": {"crop_size": 50, "created": ..., "image_source": ...}}
        templates/gemini.png
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINED_DIR = PROJECT_ROOT / "trained_ui"
TEMPLATES_DIR = TRAINED_DIR / "templates"
METADATA_FILE = TRAINED_DIR / "button_locations.json"

DEFAULT_CROP_SIZE = 50
MAX_NAME_LENGTH = 50
_NAME_RE = re.compile(r"^[A-Za-z0-9 ]+$")


def normalize_name(name: str) -> str:
    """Collapse whitespace and lowercase: ' New   Tab ' -> 'new tab'."""
    return " ".join(name.split()).lower()


def validate_name(name: str) -> str:
    """
    Validate a button name and return its normalized form.
    Raises ValueError with a user-facing message if the name is not allowed.
    """
    normalized = normalize_name(name or "")
    if not normalized:
        raise ValueError("Button name cannot be empty")
    if len(normalized) > MAX_NAME_LENGTH:
        raise ValueError(f"Button name must be at most {MAX_NAME_LENGTH} characters")
    if not _NAME_RE.match(normalized):
        raise ValueError("Button name may only contain letters, numbers and spaces")
    return normalized


def _template_file(name: str) -> Path:
    return TEMPLATES_DIR / f"{normalize_name(name).replace(' ', '_')}.png"


def _read_metadata() -> Dict[str, dict]:
    if not METADATA_FILE.exists():
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_metadata(data: Dict[str, dict]) -> None:
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = METADATA_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, METADATA_FILE)


def crop_bounds(img_w: int, img_h: int, x_pct: float, y_pct: float, crop_size: int) -> tuple:
    """
    Return (left, top, right, bottom) of a square crop centered on the point,
    shifted to stay inside the image. Shrinks the square only if the image is
    smaller than crop_size.
    """
    size = max(1, min(crop_size, img_w, img_h))
    cx = round(x_pct * img_w)
    cy = round(y_pct * img_h)
    left = min(max(cx - size // 2, 0), img_w - size)
    top = min(max(cy - size // 2, 0), img_h - size)
    return left, top, left + size, top + size


def save_template(image_path: Path, name: str, x_pct: float, y_pct: float,
                  crop_size: int = DEFAULT_CROP_SIZE) -> str:
    """
    Crop a template from image at (x_pct, y_pct) and save to trained_ui/templates/{name}.png.
    Also update metadata in trained_ui/button_locations.json.
    Overwrites an existing template with the same name (callers confirm first).
    Returns: path to saved template, relative to the project root.
    """
    key = validate_name(name)
    if not (0.0 <= x_pct <= 1.0 and 0.0 <= y_pct <= 1.0):
        raise ValueError("Coordinates must be fractions between 0.0 and 1.0")

    with Image.open(image_path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
        left, top, right, bottom = crop_bounds(img.width, img.height, x_pct, y_pct, crop_size)
        crop = img.crop((left, top, right, bottom))

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _template_file(key)
    crop.save(out_path, format="PNG")

    metadata = _read_metadata()
    metadata[key] = {
        "crop_size": crop.width,
        "created": datetime.now().isoformat(timespec="seconds"),
        "image_source": Path(image_path).name,
    }
    _write_metadata(metadata)

    try:
        return str(out_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(out_path)


def load_template(name: str) -> Optional[Path]:
    """Return path to template if exists, else None."""
    try:
        path = _template_file(validate_name(name))
    except ValueError:
        return None
    return path if path.is_file() else None


def list_templates() -> List[str]:
    """Return sorted list of all trained button names."""
    return sorted(_read_metadata().keys())


def get_template_info(name: str) -> Optional[dict]:
    """Return the metadata entry for a template, or None."""
    try:
        key = validate_name(name)
    except ValueError:
        return None
    return _read_metadata().get(key)


def delete_template(name: str) -> bool:
    """Delete template and metadata, return True if anything was removed."""
    try:
        key = validate_name(name)
    except ValueError:
        return False

    removed = False
    path = _template_file(key)
    if path.is_file():
        path.unlink()
        removed = True

    metadata = _read_metadata()
    if key in metadata:
        del metadata[key]
        _write_metadata(metadata)
        removed = True
    return removed
