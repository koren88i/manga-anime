"""
Kumiko panel detection for manga pages.

Runs OpenCV-based panel detection (Kumiko) on manga pages and saves
normalized bounding boxes for comparison with VLM-detected panels.

Usage:
    python kumiko_panels.py --chapter-id 0001
    python kumiko_panels.py --chapter-id 0001 --no-rtl
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally (001 < 002 < 010)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


def ensure_kumiko():
    """Try to import Kumiko, cloning from GitHub if needed. Returns the Kumiko class."""
    vendor_dir = BASE_DIR / "vendor" / "kumiko"

    # Add vendor path to sys.path if it exists
    vendor_str = str(vendor_dir)
    if vendor_dir.exists() and vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)

    try:
        from kumikolib import Kumiko
        return Kumiko
    except ImportError:
        pass

    # Auto-clone from GitHub
    print("Kumiko not found. Cloning from GitHub...")
    vendor_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(
            ["git", "clone", "https://github.com/njean42/kumiko.git", vendor_str],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone Kumiko: {e}")
        print("Please clone manually: git clone https://github.com/njean42/kumiko.git vendor/kumiko")
        sys.exit(1)
    except FileNotFoundError:
        print("git not found. Please install git or clone Kumiko manually:")
        print("  git clone https://github.com/njean42/kumiko.git vendor/kumiko")
        sys.exit(1)

    sys.path.insert(0, vendor_str)
    from kumikolib import Kumiko
    print("Kumiko cloned and imported successfully.")
    return Kumiko


def discover_pages(chapter_dir):
    """Discover manga page images, sorted naturally."""
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(chapter_dir.glob(ext))
    image_files.sort(key=lambda p: natural_sort_key(p.name))
    return image_files


def process_chapter(chapter_id, rtl=True):
    """Run Kumiko panel detection on all pages of a chapter."""
    Kumiko = ensure_kumiko()

    manga_dir = BASE_DIR / "manga" / chapter_id
    if not manga_dir.exists():
        print(f"Error: manga directory not found: {manga_dir}")
        sys.exit(1)

    output_dir = BASE_DIR / "output" / "kumiko_panels" / chapter_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover pages
    image_files = discover_pages(manga_dir)
    if not image_files:
        print(f"Error: no images found in {manga_dir}")
        sys.exit(1)

    print(f"Chapter {chapter_id}: {len(image_files)} pages")
    print(f"RTL reading order: {rtl}")
    print()

    # Run Kumiko on all pages at once
    start = time.time()
    k = Kumiko({"rtl": rtl})
    filenames = [str(p) for p in image_files]
    k.parse_images(filenames)
    all_infos = list(k.get_infos())
    elapsed = time.time() - start

    # Build result
    result = {
        "chapter_id": chapter_id,
        "processing_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rtl": rtl,
        "processing_time_seconds": round(elapsed, 2),
        "pages": {},
    }

    total_panels = 0
    for img_path, page_info in zip(image_files, all_infos):
        img_size = page_info.get("size", [0, 0])  # [width, height]
        img_w, img_h = img_size

        raw_panels = page_info.get("panels", [])
        panels = []
        for i, (x, y, w, h) in enumerate(raw_panels):
            if img_w <= 0 or img_h <= 0:
                continue
            panels.append({
                "panel_index": i,
                "bbox": [
                    round(x / img_w, 4),
                    round(y / img_h, 4),
                    round((x + w) / img_w, 4),
                    round((y + h) / img_h, 4),
                ],
                "bbox_pixels": [x, y, x + w, y + h],
            })

        result["pages"][img_path.name] = {
            "image_size": img_size,
            "panels": panels,
        }
        total_panels += len(panels)

        status = f"  {img_path.name}: {len(panels)} panels"
        print(status)

    # Save
    output_path = output_dir / "kumiko_panels.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Pages: {len(result['pages'])}")
    print(f"Total panels: {total_panels}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Kumiko panel detection on manga pages.",
        epilog="Example: python kumiko_panels.py --chapter-id 0001",
    )
    parser.add_argument(
        "--chapter-id", required=True,
        help="Chapter ID, zero-padded (e.g. '0001')",
    )
    parser.add_argument(
        "--no-rtl", action="store_true",
        help="Disable right-to-left reading order (default: RTL enabled for manga)",
    )
    args = parser.parse_args()
    process_chapter(args.chapter_id, rtl=not args.no_rtl)


if __name__ == "__main__":
    main()
