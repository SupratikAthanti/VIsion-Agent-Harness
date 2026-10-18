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
import base64
import argparse
import urllib.request
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
    from ocr_engine import find_text_element, extract_search_phrase, OCRError
    from tev1_router import route_task, check_tev1_available, TEV1Error
    from yolo_detector import YOLODetector
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False


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


def run_hybrid_query(query: str, image_path: Path, images_dir: Path, first_run: bool = False) -> None:
    """Route a request via TEV1, then run OCR with graceful fallbacks, and print the result."""
    try:
        with Spinner("Loading TEV1 router..." if first_run else "Routing request..."):
            route = route_task(query)
        method = route["method"]
        print(f"TEV1 route: {method} (confidence {route['confidence'] * 100:.1f}%)")
    except TEV1Error as e:
        print(f"Warning: TEV1 routing failed ({e}); defaulting to OCR.")
        method = "ocr"

    if method == "vision":
        print("Note: Vision answers are not wired into hybrid mode yet.")
        print("   Ask about a visible label instead, e.g. 'Where is the <label> button?'")
        return

    if method == "yolo":
        try:
            with Spinner("Running YOLO icon detection..."):
                detector = YOLODetector()
                
                # For new tab icon, try to find text first, then look for nearby icon
                if 'new tab' in query.lower():
                    # First try OCR to find "New Tab" text
                    phrase = extract_search_phrase(query) or "New Tab"
                    text_hit = find_text_element(image_path, phrase)
                    
                    if text_hit:
                        # Look for small icon near the text
                        icon_detection = detector.find_icon_near_text(
                            str(image_path), 
                            text_hit, 
                            icon_type='plus'
                        )
                        if icon_detection and icon_detection['confidence'] > 0.05:
                            x_pct, y_pct = icon_detection['center_pct']['x'], icon_detection['center_pct']['y']
                            print(f"\nFound icon near '{text_hit['text']}' (YOLO confidence {icon_detection['confidence']*100:.0f}%)")
                            print(f"   Would click at ({x_pct:.1f}%, {y_pct:.1f}%)")
                            auto_mark_target(image_path, images_dir, x_pct/100, y_pct/100, f"{text_hit['text']} icon")
                            return
                        else:
                            # Text-based icon detection failed, try position-based
                            print(f"Text-based icon detection failed, trying position-based...")
                            detection = detector.find_icon_by_position(str(image_path), icon_type='top_bar')
                            if detection and detection['confidence'] > 0.05:
                                x_pct, y_pct = detection['center_pct']['x'], detection['center_pct']['y']
                                print(f"\nFound icon (YOLO confidence {detection['confidence']*100:.0f}%)")
                                print(f"   Would click at ({x_pct:.1f}%, {y_pct:.1f}%)")
                                auto_mark_target(image_path, images_dir, x_pct/100, y_pct/100, "icon")
                                return
                
                # Fallback to position-based detection
                icon_type = 'top_bar'
                if 'close' in query.lower():
                    icon_type = 'top_bar'
                elif 'menu' in query.lower():
                    icon_type = 'top_left'
                
                detection = detector.find_icon_by_position(str(image_path), icon_type=icon_type)
            
            if detection is None:
                print("YOLO found no matching icons in the image.")
                print("   Falling back to OCR...")
                method = "ocr"
            else:
                x_pct, y_pct = detection['center_pct']['x'], detection['center_pct']['y']
                print(f"\nFound icon (YOLO confidence {detection['confidence']*100:.0f}%)")
                print(f"   Would click at ({x_pct:.1f}%, {y_pct:.1f}%)")
                auto_mark_target(image_path, images_dir, x_pct/100, y_pct/100, "icon")
                return
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            print("   Falling back to OCR...")
            method = "ocr"

    # OCR path (either routed to OCR or YOLO fallback)
    phrase = extract_search_phrase(query)
    if not phrase:
        print("Warning: Could not find a label to search for in that request.")
        return

    try:
        with Spinner("Running OCR..."):
            hit = find_text_element(image_path, phrase)
    except OCRError as e:
        print(f"[Error] OCR failed: {e}")
        return

    if hit is None:
        print(f"OCR could not find '{phrase}' on screen.")
        print("   Trying YOLO icon detection as fallback...")
        try:
            with Spinner("Running YOLO fallback..."):
                detector = YOLODetector()
                
                # Determine icon type based on query
                icon_type = 'top_bar'
                if 'new tab' in query.lower():
                    icon_type = 'top_bar'
                elif 'close' in query.lower():
                    icon_type = 'top_bar'
                elif 'menu' in query.lower():
                    icon_type = 'top_left'
                
                detection = detector.find_icon_by_position(str(image_path), icon_type=icon_type)
            
            if detection is None:
                print("   YOLO also found no matching icons.")
                print("   Use /grid to inspect the screen manually.")
                return
            else:
                x_pct, y_pct = detection['center_pct']['x'], detection['center_pct']['y']
                print(f"\nFound icon via YOLO fallback (confidence {detection['confidence']*100:.0f}%)")
                print(f"   Would click at ({x_pct:.1f}%, {y_pct:.1f}%)")
                auto_mark_target(image_path, images_dir, x_pct/100, y_pct/100, "icon")
                return
        except Exception as e:
            print(f"   YOLO fallback failed: {e}")
            print("   Use /grid to inspect the screen manually.")
            return

    x_pct, y_pct = hit["center_pct"]
    print(f"\nFound '{hit['text']}' (OCR confidence {hit['conf']:.0f}%)")
    print(f"   Would click at ({x_pct:.1f}%, {y_pct:.1f}%)")
    auto_mark_target(image_path, images_dir, hit['center_x'], hit['center_y'], hit['text'])


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
    grid_active = False
    model_used = False  # Becomes True after the first inference query

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
                print(f"\nTarget marked at ({x_val*100:.1f}%, {y_val*100:.1f}%)!")
                print(f"   Saved preview to: {out_target}")
                print("   Open this file in your IDE to verify if the box hits the button!\n")
            except ValueError as ve:
                print(f"Invalid coordinate numbers: {ve}")
            continue

        if cmd == "/info":
            print_image_status()
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
