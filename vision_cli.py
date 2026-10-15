#!/usr/bin/env python3
"""
Vision Agent Harness - CLI MVP (v0.1)
Allows querying local vision-language models on Raspberry Pi 5 using llama-cpp-python.
"""

import os
import sys
import time
import base64
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def get_image_list(images_dir: Path) -> List[Path]:
    """Find and return all image files in images_dir, sorted alphabetically."""
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        return []

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    # Sort alphabetically by filename
    images.sort(key=lambda p: p.name.lower())
    return images


def get_first_image(images_dir: Path) -> Optional[Path]:
    """Return the first image alphabetically from images_dir."""
    images = get_image_list(images_dir)
    return images[0] if images else None


def image_to_base64_data_uri(image_path: Path) -> str:
    """Read an image file and encode it as a base64 data URI."""
    ext = image_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{encoded}"


def find_default_models(models_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Auto-detect main model and mmproj model in models_dir."""
    if not models_dir.exists():
        return None, None

    model_path = None
    mmproj_path = None

    for p in models_dir.glob("*.gguf"):
        name = p.name.lower()
        if "mmproj" in name:
            if mmproj_path is None or "256m" in name:
                mmproj_path = p
        else:
            if model_path is None or "256m" in name:
                model_path = p

    return model_path, mmproj_path


def format_bytes(size: int) -> str:
    """Format file size in human-readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Vision Agent Harness CLI v0.1 - CPU-Optimized for Raspberry Pi 5"
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Path to specific image file (default: top alphabetical image in images/)"
    )
    parser.add_argument(
        "--images-dir", type=str, default="images",
        help="Directory containing images (default: 'images')"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--mmproj", type=str, default=None,
        help="Path to mmproj GGUF vision projector"
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=4,
        help="Number of CPU threads to use (default: 4 for RPi 5)"
    )
    parser.add_argument(
        "--ctx", "-c", type=int, default=2048,
        help="Context window size (default: 2048)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose llama.cpp logging"
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

    # 2. Resolve model files
    model_path = Path(args.model).resolve() if args.model else None
    mmproj_path = Path(args.mmproj).resolve() if args.mmproj else None

    if not model_path or not mmproj_path:
        detected_model, detected_mmproj = find_default_models(models_dir)
        model_path = model_path or detected_model
        mmproj_path = mmproj_path or detected_mmproj

    if not model_path or not model_path.exists():
        print(f"[Error] Main model not found in {models_dir}.")
        print("Please ensure a .gguf model file is placed in 'models/'.")
        sys.exit(1)

    if not mmproj_path or not mmproj_path.exists():
        print(f"[Error] Multimodal vision projector (mmproj) not found in {models_dir}.")
        print("Please ensure an mmproj-*.gguf file is placed in 'models/'.")
        sys.exit(1)

    # 3. Load llama_cpp with MTMDChatHandler
    print("=" * 60)
    print("  👁️  Vision Agent Harness CLI (v0.1 - llama-cpp-python)")
    print("=" * 60)
    print(f" • Model:       {model_path.name} ({format_bytes(model_path.stat().st_size)})")
    print(f" • Projector:   {mmproj_path.name} ({format_bytes(mmproj_path.stat().st_size)})")
    print(f" • Threads:     {args.threads} (RPi 5 Cortex-A76)")
    print(f" • Context:     {args.ctx} tokens")

    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler
    except ImportError as e:
        print(f"\n[Error] Failed to import llama_cpp: {e}")
        print("Make sure you are running in the venv: source venv/bin/activate")
        sys.exit(1)

    print("\n⏳ Initializing vision engine and loading weights into RAM...")
    start_load = time.time()
    try:
        chat_handler = MTMDChatHandler(
            clip_model_path=str(mmproj_path),
            verbose=args.verbose,
            use_gpu=False
        )
        llm = Llama(
            model_path=str(model_path),
            chat_handler=chat_handler,
            n_ctx=args.ctx,
            n_threads=args.threads,
            n_batch=512,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        sys.exit(1)

    load_time = time.time() - start_load
    print(f"✅ Model loaded successfully in {load_time:.2f}s!")

    # Helper function to print current image status
    def print_current_image(img: Optional[Path]):
        if img and img.exists():
            size_str = format_bytes(img.stat().st_size)
            print(f"\n🖼️  Active Image: [{img.name}] ({size_str}) -> {img}")
        else:
            print("\n⚠️  No image currently found in 'images/' folder.")
            print("   Drop an image (e.g. screenshot.png) into 'images/' and run '/reload'.")

    print_current_image(selected_image)

    print("\nType your question or one of the commands:")
    print("  /reload         - Rescan images folder and pick the top image")
    print("  /list           - List all images in images/ folder")
    print("  /select <name>  - Select a specific image by name")
    print("  /info           - Show active image info")
    print("  /exit, quit     - Exit the CLI\n")

    # Interactive Loop
    while True:
        try:
            prompt_label = f"[{selected_image.name if selected_image else 'No Image'}] > "
            user_input = input(prompt_label).strip()
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
            print_current_image(selected_image)
            continue

        if cmd == "/list":
            imgs = get_image_list(images_dir)
            if not imgs:
                print("No images found in 'images/' folder.")
            else:
                print(f"Images in '{images_dir.name}/' (alphabetical order):")
                for idx, img in enumerate(imgs, 1):
                    prefix = "👉" if selected_image and img.name == selected_image.name else "  "
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
                print_current_image(selected_image)
            else:
                print(f"File not found in {images_dir}: {target_name}")
            continue

        if cmd == "/info":
            print_current_image(selected_image)
            continue

        if cmd == "/help":
            print("Commands:")
            print("  /reload         - Rescan images folder and pick the top image")
            print("  /list           - List all images in images/ folder")
            print("  /select <name>  - Select a specific image by name")
            print("  /info           - Show active image info")
            print("  /exit, quit     - Exit the CLI")
            continue

        # Inference request
        if not selected_image or not selected_image.exists():
            print("❌ Cannot query: No image loaded. Drop an image into 'images/' and type '/reload'.")
            continue

        print("\n🤔 Thinking...", flush=True)
        t_start = time.time()
        try:
            data_uri = image_to_base64_data_uri(selected_image)
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

            t_elapsed = time.time() - t_start
            answer = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            tok_per_sec = (completion_tokens / t_elapsed) if t_elapsed > 0 and completion_tokens else 0

            print(f"\n💬 Assistant:\n{answer.strip()}\n")
            if completion_tokens:
                print(f"⏱️  [{t_elapsed:.2f}s | {completion_tokens} tokens | {tok_per_sec:.1f} tok/s]\n")
            else:
                print(f"⏱️  [{t_elapsed:.2f}s]\n")

        except Exception as e:
            print(f"\n[Error during inference]: {e}\n")


if __name__ == "__main__":
    main()
