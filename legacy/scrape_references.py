"""
scrape_references.py
Scrape character roster from AnimeCharactersDatabase, then use Google Image
Search to build a comprehensive reference image library for CLIP matching.

Usage:
    python scrape_references.py                     # Full scrape + search
    python scrape_references.py --skip-search       # DB scrape only, no Google
    python scrape_references.py --max-pages 3       # Limit listing pages
    python scrape_references.py --images-per-char 5 # Fewer Google images
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, parse_qs, urlparse

import io

import requests
from bs4 import BeautifulSoup
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Anime database
    "anime_db_id": 1447,  # One Piece
    "anime_title": "One Piece",
    "base_url": "https://www.animecharactersdatabase.com",
    "listing_url_template": "https://www.animecharactersdatabase.com/allchars.php?x={offset}&id={anime_id}&tile=",

    # Output
    "output_dir": Path(r"c:\Users\User\helloworld_claude\mange-anime\output"),
    "catalog_filename": "character_catalog.json",

    # Scraping behavior
    "max_pages": 20,
    "listing_page_delay": 2.0,
    "download_delay": 0.3,
    "request_timeout": 15,

    # Google Image Search
    "google_images_per_char": 10,
    "search_query_template": "One Piece {char_name} anime character",
    "search_delay": 2.0,

    # Filtering
    "placeholder_patterns": ["filtered.png", "noimage", "placeholder"],
    "min_image_size_bytes": 2000,
    "min_image_dimension": 50,

    # HTTP
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# ============================================================
# CHARACTER NAME MAPPING (DB full name -> display name for folders)
# Mirrors the aliases in voice_clone.py / manga_metadata.py
# ============================================================
DB_NAME_TO_DISPLAY = {
    "monkey d. luffy": "Luffy",
    "roronoa zoro": "Zoro",
    "nami": "Nami",
    "usopp": "Usopp",
    "sanji": "Sanji",
    "tony tony chopper": "Chopper",
    "nico robin": "Robin",
    "franky": "Franky",
    "brook": "Brook",
    "jinbe": "Jinbe",
    "shanks": "Shanks",
    "portgas d. ace": "Ace",
    "nefeltari vivi": "Vivi",
    "trafalgar law": "Law",
    "boa hancock": "Hancock",
    "dracule mihawk": "Mihawk",
    "marshall d. teach": "Blackbeard",
    "edward newgate": "Whitebeard",
    "donquixote doflamingo": "Doflamingo",
    "buggy the star clown": "Buggy",
    "smoker": "Smoker",
    "crocodile": "Crocodile",
    "enel": "Enel",
    "rob lucci": "Rob Lucci",
    "gecko moria": "Gecko Moria",
    "bartholomew kuma": "Kuma",
    "silvers rayleigh": "Rayleigh",
    "monkey d. garp": "Garp",
    "monkey d. dragon": "Dragon",
    "sengoku": "Sengoku",
    "sakazuki": "Akainu",
    "borsalino": "Kizaru",
    "kuzan": "Aokiji",
    "eustass kid": "Kid",
    "gold roger": "Gold Roger",
    "gol d. roger": "Gold Roger",
    "arlong": "Arlong",
    "alvida": "Alvida",
    "coby": "Coby",
    "captain morgan": "Captain Morgan",
    "belle-mere": "Belle-mere",
    "nojiko": "Nojiko",
    "makino": "Makino",
    "hina": "Hina",
    "tashigi": "Tashigi",
    "bon clay": "Bon Clay",
    "emporio ivankov": "Ivankov",
    "sabo (young)": "Sabo",
    "marco the phoenix": "Marco",
    "yamato": "Yamato",
    "carrot": "Carrot",
    "rebecca": "Rebecca",
    "viola": "Viola",
    "baby 5": "Baby 5",
    "bartolomeo": "Bartolomeo",
    "caesar clown": "Caesar Clown",
    "jewelry bonney": "Bonney",
    "perona": "Perona",
    "kaya": "Kaya",
    "koala": "Koala",
    "killer": "Killer",
    "shirahoshi": "Shirahoshi",
}


def normalize_character_id(name):
    """Normalize a character name to a stable snake_case ID."""
    normalized = name.lower().strip()
    # Check display name mapping
    if normalized in DB_NAME_TO_DISPLAY:
        display = DB_NAME_TO_DISPLAY[normalized]
        return re.sub(r"[^a-z0-9]+", "_", display.lower()).strip("_")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def resolve_display_name(db_name):
    """Map a DB full name to a short display name for folder naming."""
    key = db_name.lower().strip()
    if key in DB_NAME_TO_DISPLAY:
        return DB_NAME_TO_DISPLAY[key]
    # Clean up for use as folder name (remove Windows-illegal chars)
    cleaned = re.sub(r'[\\/:*?"<>|]', '', db_name).strip()
    return cleaned if cleaned else db_name.strip()


def sanitize_folder_name(name):
    """Make a string safe for use as a Windows directory name."""
    sanitized = re.sub(r'[\\/:*?"<>|]', '', name).strip()
    # Truncate to avoid path length issues
    if len(sanitized) > 60:
        sanitized = sanitized[:60].strip()
    return sanitized or "Unknown"


def is_placeholder(url):
    """Check if a URL points to a placeholder/filtered image."""
    url_lower = url.lower()
    return any(p in url_lower for p in CONFIG["placeholder_patterns"])


def create_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": CONFIG["user_agent"]})
    return session


def download_image(session, url, dest_path):
    """Download a single image, validate it, return True on success."""
    try:
        resp = session.get(url, timeout=CONFIG["request_timeout"], stream=True)
        resp.raise_for_status()

        content = resp.content
        if len(content) < CONFIG["min_image_size_bytes"]:
            return False

        # Validate with PIL in memory before writing to disk
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()
            img = Image.open(io.BytesIO(content))
            w, h = img.size
            if w < CONFIG["min_image_dimension"] or h < CONFIG["min_image_dimension"]:
                return False
        except Exception:
            return False

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)
        return True
    except Exception:
        return False


# ============================================================
# STAGE 1: Scrape Character Listing Pages
# ============================================================
def scrape_listing_pages(session, config):
    """Scrape all character listing pages and return list of character dicts."""
    characters = []
    seen_ids = set()

    for page_num in range(config["max_pages"]):
        offset = page_num * 100
        url = config["listing_url_template"].format(
            offset=offset, anime_id=config["anime_db_id"]
        )
        print(f"  Fetching page {page_num + 1} (offset={offset})...")

        try:
            resp = session.get(url, timeout=config["request_timeout"])
            resp.raise_for_status()
        except Exception as e:
            print(f"  Error fetching page {page_num + 1}: {e}")
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Find character entries — they are in <li> tags containing links to characters.php
        page_chars = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "characters.php?id=" not in href:
                continue

            # Extract DB id
            try:
                params = parse_qs(urlparse(href).query)
                db_id = int(params.get("id", [0])[0])
            except (ValueError, IndexError):
                continue

            if db_id in seen_ids or db_id == 0:
                continue

            # Find the image inside this link
            img = link.find("img")
            if not img:
                continue

            thumbnail_url = img.get("src", "")
            if not thumbnail_url:
                continue

            # Make absolute URL
            if thumbnail_url.startswith("//"):
                thumbnail_url = "https:" + thumbnail_url
            elif thumbnail_url.startswith("/"):
                thumbnail_url = config["base_url"] + thumbnail_url

            # Skip placeholders
            if is_placeholder(thumbnail_url):
                continue

            # Get character name — try alt text, then link text
            name = img.get("alt", "").strip()
            if not name:
                name = link.get_text(strip=True)
            if not name:
                continue

            seen_ids.add(db_id)
            page_chars.append({
                "db_name": name,
                "db_id": db_id,
                "detail_url": urljoin(config["base_url"] + "/", href),
                "thumbnail_url": thumbnail_url,
                "display_name": resolve_display_name(name),
            })

        characters.extend(page_chars)
        print(f"    Found {len(page_chars)} characters (total: {len(characters)})")

        if len(page_chars) == 0:
            print("  No more characters found, stopping pagination.")
            break

        # Check for "Next" link to confirm more pages
        next_link = soup.find("a", string=re.compile(r"Next", re.IGNORECASE))
        if not next_link:
            print("  No 'Next' link found, this is the last page.")
            break

        time.sleep(config["listing_page_delay"])

    return characters


# ============================================================
# STAGE 2: Download DB Thumbnails
# ============================================================
def download_thumbnails(session, characters, config):
    """Download the DB thumbnail for each character as db_001."""
    refs_dir = config["output_dir"] / "references"
    downloaded = 0
    skipped = 0

    for char in characters:
        display = sanitize_folder_name(char["display_name"])
        char_dir = refs_dir / display
        dest = char_dir / "db_001.jpg"

        if dest.exists():
            skipped += 1
            continue

        if download_image(session, char["thumbnail_url"], dest):
            downloaded += 1
        else:
            skipped += 1

        time.sleep(config["download_delay"])

    print(f"  Thumbnails: {downloaded} downloaded, {skipped} skipped (existing or failed)")
    return downloaded


# ============================================================
# STAGE 3: Google Image Search
# ============================================================
def _fetch_google_image_urls(session, query, num_needed):
    """Scrape image URLs from Google Image Search results."""
    url = "https://www.google.com/search"
    params = {"q": query, "tbm": "isch", "ijn": "0"}
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []

    # Google embeds image URLs in the page source
    img_urls = re.findall(r'"(https?://[^"]+\.(?:jpg|jpeg|png))"', resp.text)
    # Filter out Google's own assets
    blocked = ("gstatic.com", "google.com", "googleapis.com", "youtube.com")
    img_urls = [u for u in img_urls if not any(b in u for b in blocked)]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in img_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique[:num_needed + 5]  # grab extras in case some fail


def search_google_images(characters, config):
    """Use Google Image Search to find reference images per character."""
    session = create_session()
    refs_dir = config["output_dir"] / "references"
    target_count = config["google_images_per_char"]
    searched = 0
    skipped = 0

    for i, char in enumerate(characters):
        display = sanitize_folder_name(char["display_name"])
        char_dir = refs_dir / display
        char_dir.mkdir(parents=True, exist_ok=True)

        # Count existing Google images
        existing_google = [p for p in char_dir.glob("google_*")
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(existing_google) >= target_count:
            skipped += 1
            continue

        need = target_count - len(existing_google)
        query = config["search_query_template"].format(char_name=char["display_name"])
        print(f"  [{i+1}/{len(characters)}] Searching: {query} (need {need} more)")

        img_urls = _fetch_google_image_urls(session, query, need)
        if not img_urls:
            print(f"    No results found")
            continue

        next_idx = len(existing_google) + 1
        added = 0

        for img_url in img_urls:
            if added >= need:
                break

            ext = Path(urlparse(img_url).path).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                ext = ".jpg"
            dest = char_dir / f"google_{next_idx:03d}{ext}"

            if download_image(session, img_url, dest):
                next_idx += 1
                added += 1

            time.sleep(config["download_delay"])

        print(f"    Added {added} images (total: {len(existing_google) + added})")
        searched += 1
        time.sleep(config["search_delay"])

    print(f"  Google search: {searched} characters searched, {skipped} skipped (already have enough)")
    return searched


# ============================================================
# STAGE 4: Build Character Catalog JSON
# ============================================================
def build_catalog(characters, config):
    """Build and save the character catalog JSON."""
    refs_dir = config["output_dir"] / "references"
    catalog = {
        "anime_title": config["anime_title"],
        "anime_db_id": config["anime_db_id"],
        "scraped_at": datetime.now().isoformat(),
        "total_characters": len(characters),
        "characters": {},
        "name_mapping": {},
    }

    for char in characters:
        display = sanitize_folder_name(char["display_name"])
        char_dir = refs_dir / display
        if char_dir.exists():
            all_images = [p for p in char_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            db_images = [p for p in all_images if p.name.startswith("db_")]
            google_images = [p for p in all_images if p.name.startswith("google_")]
        else:
            all_images, db_images, google_images = [], [], []

        catalog["characters"][display] = {
            "db_name": char["db_name"],
            "display_name": display,
            "normalized_id": normalize_character_id(char["db_name"]),
            "db_id": char["db_id"],
            "detail_url": char["detail_url"],
            "num_reference_images": len(all_images),
            "num_db_images": len(db_images),
            "num_google_images": len(google_images),
        }

        # Build name mapping (multiple keys -> same display name)
        db_lower = char["db_name"].lower().strip()
        catalog["name_mapping"][db_lower] = display
        display_lower = display.lower().strip()
        if display_lower != db_lower:
            catalog["name_mapping"][display_lower] = display

    catalog_path = config["output_dir"] / config["catalog_filename"]
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"  Catalog saved: {catalog_path}")
    return catalog


# ============================================================
# STAGE 5: Summary Report
# ============================================================
def print_report(catalog):
    """Print a summary of the scrape results."""
    chars = catalog["characters"]
    total = len(chars)
    total_images = sum(c["num_reference_images"] for c in chars.values())
    with_images = sum(1 for c in chars.values() if c["num_reference_images"] > 0)
    no_images = [name for name, c in chars.items() if c["num_reference_images"] == 0]
    few_images = [name for name, c in chars.items() if 0 < c["num_reference_images"] < 3]

    print(f"\n{'='*60}")
    print(f"SCRAPE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total characters:   {total}")
    print(f"  With images:        {with_images}")
    print(f"  Total images:       {total_images}")
    print(f"  Avg images/char:    {total_images / max(total, 1):.1f}")

    if no_images:
        print(f"\n  Characters with NO images ({len(no_images)}):")
        for name in no_images[:20]:
            print(f"    - {name}")
        if len(no_images) > 20:
            print(f"    ... and {len(no_images) - 20} more")

    if few_images:
        print(f"\n  Characters with <3 images ({len(few_images)}):")
        for name in few_images[:10]:
            c = chars[name]
            print(f"    - {name}: {c['num_reference_images']} images")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Scrape character reference images")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip Google Image Search, only scrape DB")
    parser.add_argument("--max-pages", type=int, default=CONFIG["max_pages"],
                        help="Max listing pages to scrape")
    parser.add_argument("--images-per-char", type=int, default=CONFIG["google_images_per_char"],
                        help="Target number of Google images per character")
    args = parser.parse_args()

    CONFIG["max_pages"] = args.max_pages
    CONFIG["google_images_per_char"] = args.images_per_char

    session = create_session()

    # Stage 1: Scrape listing pages
    print("\n=== STAGE 1: Scraping character listing pages ===")
    characters = scrape_listing_pages(session, CONFIG)
    print(f"  Total characters found: {len(characters)}")

    if not characters:
        print("No characters found. Check the URL and network connection.")
        return

    # Stage 2: Download thumbnails
    print("\n=== STAGE 2: Downloading DB thumbnails ===")
    download_thumbnails(session, characters, CONFIG)

    # Stage 3: Google Image Search
    if not args.skip_search:
        print(f"\n=== STAGE 3: Google Image Search ({args.images_per_char} images/char) ===")
        search_google_images(characters, CONFIG)
    else:
        print("\n=== STAGE 3: Skipped (--skip-search) ===")

    # Stage 4: Build catalog
    print("\n=== STAGE 4: Building character catalog ===")
    catalog = build_catalog(characters, CONFIG)

    # Stage 5: Report
    print_report(catalog)


if __name__ == "__main__":
    main()
