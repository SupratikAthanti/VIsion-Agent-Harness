"""
Grid Overlay and Visual Grounding Utilities for Vision Agent Harness.
Allows drawing labeled coordinate grids and target markers on screenshots.
"""

from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


GRID_COLS = 20
GRID_ROWS = 20


def column_label(index: int) -> str:
    """Spreadsheet-style column label: 0->A, 25->Z, 26->AA, ..."""
    name = ""
    n = index
    while True:
        name = chr(ord("A") + n % 26) + name
        n = n // 26 - 1
        if n < 0:
            break
    return name


def create_coordinate_grid(
    image_path: Path,
    output_path: Optional[Path] = None,
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
    max_dimension: int = 1920,
) -> Path:
    """
    Draws a labeled grid overlay (default 20x20) onto an image.
    Columns are labeled A-T, rows are labeled R1-R20.
    Also displays percentage coordinate ticks along the borders.
    """
    if output_path is None:
        output_path = image_path.parent / f"grid_{image_path.name}.png"

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # Scale down if ridiculously large to save memory/tokens
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = new_w, new_h

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    col_width = w / cols
    row_height = h / rows

    # Try to load default font, or fallback
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw vertical grid lines & column headers
    for c in range(cols + 1):
        x = int(c * col_width)
        # Line
        draw.line([(x, 0), (x, h)], fill=(0, 200, 255, 90), width=1)
        if c < cols:
            label = f"{column_label(c)} ({(c/cols)*100:.0f}%)"
            # Label box at top
            draw.rectangle([(x + 2, 2), (x + 60, 18)], fill=(0, 0, 0, 180))
            draw.text((x + 5, 4), label, fill=(0, 230, 255, 255), font=font)

    # Draw horizontal grid lines & row headers
    for r in range(rows + 1):
        y = int(r * row_height)
        # Line
        draw.line([(0, y), (w, y)], fill=(0, 200, 255, 90), width=1)
        if r < rows:
            label = f"R{r+1} ({(r/rows)*100:.0f}%)"
            # Label box at left
            draw.rectangle([(2, y + 2), (55, y + 18)], fill=(0, 0, 0, 180))
            draw.text((5, y + 4), label, fill=(255, 220, 0, 255), font=font)

    # Composite overlay onto base image
    combined = Image.alpha_composite(img, overlay).convert("RGB")
    combined.save(output_path, "PNG")
    return output_path


def draw_target_marker(
    image_path: Path,
    x_ratio: float,
    y_ratio: float,
    output_path: Optional[Path] = None,
    label: str = "CLICK TARGET",
    box_radius: int = 35,
) -> Path:
    """
    Draws a high-visibility target marker (reticle circle & crosshair)
    at the given normalized coordinates (0.0 to 1.0).
    Saves the image for visual verification by user.
    """
    if output_path is None:
        output_path = image_path.parent / f"target_{image_path.name}.png"

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    px = int(x_ratio * w)
    py = int(y_ratio * h)

    # Draw neon outer circle
    draw.ellipse(
        [(px - box_radius, py - box_radius), (px + box_radius, py + box_radius)],
        outline=(255, 100, 0, 255),
        width=4,
    )
    # Inner center dot
    draw.ellipse(
        [(px - 6, py - 6), (px + 6, py + 6)],
        fill=(255, 200, 0, 255),
    )
    # Crosshair lines
    draw.line([(px - box_radius - 15, py), (px + box_radius + 15, py)], fill=(255, 100, 0, 220), width=3)
    draw.line([(px, py - box_radius - 15), (px, py + box_radius + 15)], fill=(255, 100, 0, 220), width=3)

    # Text banner
    banner_text = f"{label} ({px}px, {py}px | {x_ratio*100:.1f}%, {y_ratio*100:.1f}%)"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    text_x = min(px + box_radius + 10, w - 240)
    text_y = max(py - 10, 10)
    draw.rectangle([(text_x - 4, text_y - 2), (text_x + 230, text_y + 18)], fill=(0, 0, 0, 200))
    draw.text((text_x, text_y), banner_text, fill=(255, 255, 255, 255), font=font)

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    combined.save(output_path, "PNG")
    return output_path
