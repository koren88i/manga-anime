"""
Manga Reader with TTS
A web-based manga page reader with browser text-to-speech.
Run: python reader.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import glob
import re
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, render_template_string

# Import from manga_metadata for debug panel (system prompt + wiki context)
try:
    from extraction.manga_metadata import build_system_prompt, fetch_wiki_chapter_context
    HAS_MANGA_METADATA = True
except ImportError:
    HAS_MANGA_METADATA = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
PORT = 5000

# Chapter management (populated at startup)
chapter_cache = {}   # chapter_id -> {pages, metadata_map, debug_data, manga_dir, metadata_dir}
current_chapter_id = None

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def natural_sort_key(s):
    """Sort strings with embedded numbers naturally (001 < 002 < 010)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


def load_chapter_pages(chapter_id):
    """Load all manga images and metadata for a chapter."""
    manga_dir = BASE_DIR / "manga" / chapter_id
    metadata_dir = BASE_DIR / "output" / "manga_metadata" / chapter_id

    # Discover images
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(manga_dir.glob(ext))
    image_files.sort(key=lambda p: natural_sort_key(p.name))
    image_names = [f.name for f in image_files]

    # Load all metadata, keyed by source_file
    metadata_map = {}
    if metadata_dir.exists():
        for mf in metadata_dir.glob("page_*.json"):
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata_map[data.get("source_file", "")] = data

    # Build page list
    pages = []
    for i, img in enumerate(image_names):
        meta = metadata_map.get(img)
        pages.append({
            "index": i,
            "image": img,
            "has_metadata": meta is not None,
            "page_type": meta.get("page_type", "unknown") if meta else "unknown",
        })

    return pages, metadata_map, manga_dir, metadata_dir


def build_script(metadata):
    """Flatten panels + text_elements into an ordered reading script."""
    if not metadata or not metadata.get("panels"):
        return {"script": [], "sound_effects": [], "panels": [],
                "locations": [], "time_context": None, "characters": [],
                "page_narrative": None}

    script_items = []
    all_sfx = []
    panel_info = []
    seen_locations = []
    seen_chars = {}
    time_context = None
    panels = sorted(metadata["panels"], key=lambda p: p.get("reading_order", 0))

    for panel in panels:
        panel_id = panel.get("panel_id", "")
        panel_order = panel.get("reading_order", 0)
        scene = panel.get("scene_description", "")

        # Collect panel-level metadata
        location = panel.get("location")
        tc = panel.get("time_context")
        if tc and not time_context:
            time_context = tc
        if location and location not in seen_locations:
            seen_locations.append(location)

        chars_present = panel.get("characters_present", [])
        for ch in chars_present:
            cid = ch.get("character_id")
            if cid and cid not in seen_chars:
                seen_chars[cid] = ch.get("name", cid)

        panel_info.append({
            "panel_id": panel_id,
            "panel_order": panel_order,
            "bbox": panel.get("bbox"),
            "panel_narrative": panel.get("panel_narrative", ""),
            "location": location,
            "time_context": tc,
            "characters_present": chars_present,
        })

        texts = sorted(
            panel.get("text_elements", []),
            key=lambda t: t.get("reading_order", 0),
        )

        for te in texts:
            content = te.get("content", "").strip()
            if not content:
                continue
            # Skip purely Japanese text (no latin chars or digits)
            if content and not re.search(r'[a-zA-Z0-9]', content):
                continue

            script_items.append({
                "index": len(script_items),
                "text_id": te.get("text_id", ""),
                "panel_id": panel_id,
                "panel_order": panel_order,
                "scene": scene,
                "type": te.get("type", "dialog"),
                "content": content,
                "speaker": te.get("speaker"),
                "speaker_id": te.get("speaker_id"),
                "emotion": te.get("emotion"),
                "speech_bubble_type": te.get("speech_bubble_type"),
            })

        for sfx in panel.get("sound_effects", []):
            all_sfx.append({**sfx, "panel_order": panel_order})

    characters = [{"character_id": cid, "name": name}
                  for cid, name in seen_chars.items()]

    return {
        "page_number": metadata.get("page_number"),
        "source_file": metadata.get("source_file"),
        "page_type": metadata.get("page_type"),
        "page_narrative": metadata.get("page_narrative"),
        "panels": panel_info,
        "locations": seen_locations,
        "time_context": time_context,
        "characters": characters,
        "script": script_items,
        "sound_effects": all_sfx,
    }


def load_chapter_debug(chapter_id):
    """Load chapter-level debug data: manifest, character registry, wiki context."""
    metadata_dir = BASE_DIR / "output" / "manga_metadata" / chapter_id
    debug = {}

    # Chapter manifest
    manifest_path = metadata_dir / "chapter_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        debug["processing_metadata"] = manifest.get("processing_metadata", {})
        debug["statistics"] = manifest.get("statistics", {})
        debug["locations"] = manifest.get("locations", [])
        debug["arc"] = manifest.get("arc", "")
        debug["saga"] = manifest.get("saga", "")
        debug["chapter_title"] = manifest.get("chapter_title", "")

    # Character registry
    registry_path = metadata_dir / "character_registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            debug["character_registry"] = json.load(f)

    # Wiki context (live fetch via same function the extraction pipeline uses)
    debug["wiki_context"] = None
    if HAS_MANGA_METADATA:
        try:
            wiki_ctx = fetch_wiki_chapter_context(chapter_id)
            if wiki_ctx:
                debug["wiki_context"] = wiki_ctx
                print(f"  Wiki context loaded: {len(wiki_ctx.get('characters', []))} characters, "
                      f"{len(wiki_ctx.get('synopsis') or '')} char synopsis, "
                      f"{len(wiki_ctx.get('chapter_notes') or [])} notes")
        except Exception as e:
            print(f"  Wiki context fetch failed: {e}")

    return debug


def discover_chapters():
    """Scan manga/ directory for available chapters."""
    manga_base = BASE_DIR / "manga"
    metadata_base = BASE_DIR / "output" / "manga_metadata"
    chapters = []
    if not manga_base.exists():
        return chapters
    for d in sorted(manga_base.iterdir(), key=lambda p: natural_sort_key(p.name)):
        if not d.is_dir():
            continue
        chapter_id = d.name
        # Check if has images
        has_images = (any(d.glob('*.jpg')) or any(d.glob('*.png'))
                      or any(d.glob('*.jpeg')))
        if not has_images:
            continue
        # Check for manifest to get title
        title = None
        has_metadata = False
        manifest_path = metadata_base / chapter_id / "chapter_manifest.json"
        if manifest_path.exists():
            has_metadata = True
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                title = manifest.get('chapter_title', '')
            except Exception:
                pass
        chapters.append({
            'chapter_id': chapter_id,
            'title': title or f'Chapter {chapter_id}',
            'has_metadata': has_metadata,
        })
    return chapters


def load_kumiko_panels(chapter_id):
    """Load Kumiko panel detection data for a chapter, if available."""
    kumiko_path = BASE_DIR / "output" / "kumiko_panels" / chapter_id / "kumiko_panels.json"
    if not kumiko_path.exists():
        return None
    with open(kumiko_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_chapter(chapter_id=None):
    """Get cached chapter data, loading if needed."""
    cid = chapter_id or current_chapter_id
    if cid not in chapter_cache:
        pages, metadata_map, manga_dir, metadata_dir = load_chapter_pages(cid)
        debug = load_chapter_debug(cid)
        kumiko_data = load_kumiko_panels(cid)
        chapter_cache[cid] = {
            'pages': pages,
            'metadata_map': metadata_map,
            'debug_data': debug,
            'manga_dir': manga_dir,
            'metadata_dir': metadata_dir,
            'kumiko_data': kumiko_data,
        }
        kumiko_status = f", kumiko: {len(kumiko_data['pages'])} pages" if kumiko_data else ""
        print(f"Loaded chapter {cid}: {len(pages)} pages, "
              f"{sum(1 for p in pages if p['has_metadata'])} with metadata{kumiko_status}")
    return chapter_cache[cid]


def set_chapter(chapter_id):
    """Switch to a different chapter."""
    global current_chapter_id
    current_chapter_id = chapter_id
    get_chapter(chapter_id)  # ensure loaded


AVAILABLE_CHAPTERS = discover_chapters()
if AVAILABLE_CHAPTERS:
    initial_chapter = AVAILABLE_CHAPTERS[0]['chapter_id']
    set_chapter(initial_chapter)
    print(f"Available chapters: {[c['chapter_id'] for c in AVAILABLE_CHAPTERS]}")
else:
    print("Warning: No chapters found in manga/ directory")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chapters")
def api_chapters():
    return jsonify({
        'chapters': AVAILABLE_CHAPTERS,
        'current': current_chapter_id,
    })


@app.route("/api/set-chapter/<chapter_id>", methods=["POST"])
def api_set_chapter(chapter_id):
    if not any(c['chapter_id'] == chapter_id for c in AVAILABLE_CHAPTERS):
        return jsonify({'error': 'Chapter not found'}), 404
    set_chapter(chapter_id)
    return jsonify({'current': current_chapter_id})


@app.route("/api/pages")
def api_pages():
    chap = get_chapter()
    return jsonify({
        'pages': chap['pages'],
        'has_kumiko': chap.get('kumiko_data') is not None,
    })


@app.route("/api/page/<int:idx>/script")
def api_script(idx):
    chap = get_chapter()
    pages = chap['pages']
    if idx < 0 or idx >= len(pages):
        return jsonify({"script": [], "sound_effects": []}), 404
    page = pages[idx]
    meta = chap['metadata_map'].get(page["image"])
    return jsonify(build_script(meta))


@app.route("/api/page/<int:idx>/kumiko")
def api_kumiko(idx):
    chap = get_chapter()
    kumiko = chap.get('kumiko_data')
    if not kumiko:
        return jsonify({"available": False, "panels": []})
    pages = chap['pages']
    if idx < 0 or idx >= len(pages):
        return jsonify({"available": False, "panels": []}), 404
    page = pages[idx]
    page_data = kumiko.get("pages", {}).get(page["image"], {})
    return jsonify({
        "available": True,
        "panels": page_data.get("panels", []),
    })


@app.route("/manga/<chapter_id>/<path:filename>")
def serve_image(chapter_id, filename):
    manga_dir = BASE_DIR / "manga" / chapter_id
    return send_from_directory(str(manga_dir), filename)


@app.route("/api/debug/chapter")
def api_debug_chapter():
    chap = get_chapter()
    return jsonify(chap['debug_data'])


@app.route("/api/debug/page/<int:idx>")
def api_debug_page(idx):
    chap = get_chapter()
    pages = chap['pages']
    debug_data = chap['debug_data']

    if idx < 0 or idx >= len(pages):
        return jsonify({"raw_metadata": None}), 404
    page = pages[idx]
    raw = chap['metadata_map'].get(page["image"])

    # Reconstruct user prompt to match manga_metadata.py build_prompt()
    context_parts = []
    wiki = debug_data.get("wiki_context")

    # 1. Chapter synopsis
    if wiki and wiki.get("synopsis"):
        synopsis = wiki["synopsis"]
        if len(synopsis) > 1500:
            synopsis = synopsis[:1500] + "..."
        context_parts.append(f"CHAPTER SYNOPSIS:\n  {synopsis}")

    # 2. Key chapter events
    if wiki and wiki.get("chapter_notes"):
        context_parts.append("KEY CHAPTER EVENTS:")
        for note in wiki["chapter_notes"][:8]:
            context_parts.append(f"  - {note}")

    # 3. Context from previous pages (reconstruct from metadata)
    prev_context = []
    for pi in range(max(0, idx - 3), idx):  # 3-page context window
        prev_page = pages[pi]
        prev_meta = chap['metadata_map'].get(prev_page["image"])
        if prev_meta and prev_meta.get("page_narrative"):
            prev_context.append(f"  - {prev_meta['page_narrative']}")
    if prev_context:
        context_parts.append("\nCONTEXT FROM PREVIOUS PAGES:")
        context_parts.extend(prev_context)

    # 4. Known characters discovered so far
    registry = debug_data.get("character_registry")
    if registry and registry.get("characters"):
        chars_so_far = {}
        for pi in range(0, idx):
            prev_page = pages[pi]
            prev_meta = chap['metadata_map'].get(prev_page["image"])
            if prev_meta:
                for panel in prev_meta.get("panels", []):
                    for ch in panel.get("characters_present", []):
                        cid = ch.get("character_id")
                        if cid and cid in registry["characters"]:
                            chars_so_far[cid] = registry["characters"][cid]
        if chars_so_far:
            context_parts.append("\nKNOWN CHARACTERS DISCOVERED SO FAR:")
            for cid, cdata in chars_so_far.items():
                name = cdata.get("name", cid)
                desc = cdata.get("visual_description", "")
                context_parts.append(f"  - {cid}: {name} - {desc}")

    context_text = "\n".join(context_parts) if context_parts else ""

    ch_id = current_chapter_id
    ch_ref = f"One Piece Chapter {int(ch_id)}" if ch_id.isdigit() else f"Chapter {ch_id}"
    page_num = page.get("index", idx)
    img_name = page.get("image", "")

    # Match the exact format from manga_metadata.py build_prompt()
    prompt_preview = f"""Analyze page {page_num:03d} ({img_name}) of {ch_ref}.

{context_text}

Extract ALL metadata following the JSON schema from the system prompt.
Be thorough: capture every text element, every character visible, every sound effect.
Use the exact character names from the character guide when possible."""

    # System prompt via actual builder from manga_metadata.py
    system_prompt_preview = None
    if HAS_MANGA_METADATA:
        system_prompt_preview = build_system_prompt(debug_data.get("wiki_context"))

    return jsonify({
        "raw_metadata": raw,
        "user_prompt_preview": prompt_preview,
        "system_prompt_preview": system_prompt_preview,
    })


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Manga Reader</title>
<style>
:root {
    --bg-dark: #0d0d1a;
    --bg-panel: #161625;
    --bg-card: #1e1e32;
    --bg-highlight: rgba(233, 69, 96, 0.15);
    --text-primary: #e8e8f0;
    --text-secondary: #8888a0;
    --text-muted: #555570;
    --accent: #e94560;
    --accent-dim: rgba(233, 69, 96, 0.5);
    --border: #2a2a40;
    --sfx-color: #ffd700;
    --narration-color: #88bbff;
    --panel-header-bg: #12121f;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg-dark);
    color: var(--text-primary);
}

/* Layout */
#app { display: flex; flex-direction: column; height: 100vh; }

/* Top bar */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: 12px;
    flex-shrink: 0;
    z-index: 10;
}
.top-bar .title {
    font-size: 15px;
    font-weight: 600;
    white-space: nowrap;
    color: var(--text-secondary);
}
.top-bar .title span { color: var(--accent); }
.nav-group, .tts-group {
    display: flex;
    align-items: center;
    gap: 8px;
}
.page-info {
    font-size: 14px;
    color: var(--text-secondary);
    min-width: 90px;
    text-align: center;
}
.page-input {
    width: 36px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 14px;
    text-align: center;
    padding: 2px 4px;
    font-family: inherit;
    outline: none;
    -moz-appearance: textfield;
}
.page-input::-webkit-outer-spin-button,
.page-input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.page-input:focus {
    border-color: var(--accent);
    background: var(--bg-highlight);
}
.page-type-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--bg-card);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
button {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 14px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s;
}
button:hover:not(:disabled) {
    background: var(--accent);
    border-color: var(--accent);
}
button:disabled {
    opacity: 0.35;
    cursor: default;
}
button.active {
    background: var(--accent);
    border-color: var(--accent);
}
.speed-control {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: var(--text-muted);
}
.speed-control input[type="range"] {
    width: 70px;
    accent-color: var(--accent);
}
.speed-value {
    min-width: 30px;
    text-align: center;
    color: var(--text-secondary);
}

/* Chapter selector */
.chapter-select {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    font-family: inherit;
    cursor: pointer;
    outline: none;
    max-width: 280px;
}
.chapter-select:focus {
    border-color: var(--accent);
}
.chapter-select option {
    background: var(--bg-card);
    color: var(--text-primary);
}

/* Main content */
.main-content {
    display: flex;
    flex: 1;
    overflow: hidden;
}

/* Debug panel (far left, hidden by default) */
.debug-panel {
    width: 340px;
    min-width: 300px;
    overflow-y: auto;
    background: #0e0e18;
    border-right: 1px solid var(--border);
    flex-shrink: 0;
    display: none;
    font-size: 11px;
}
.debug-panel.visible { display: block; }
.debug-panel::-webkit-scrollbar { width: 6px; }
.debug-panel::-webkit-scrollbar-track { background: transparent; }
.debug-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.debug-section {
    border-bottom: 1px solid var(--border);
}
.debug-section-header {
    padding: 8px 12px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--accent);
    background: #0a0a14;
    cursor: pointer;
    user-select: none;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.debug-section-header:hover { background: #12121f; }
.debug-section-header .chevron {
    font-size: 8px;
    transition: transform 0.2s;
}
.debug-section.collapsed .chevron { transform: rotate(-90deg); }
.debug-section.collapsed .debug-body { display: none; }
.debug-body {
    padding: 8px 12px;
}
.debug-kv {
    display: flex;
    gap: 6px;
    margin-bottom: 3px;
    line-height: 1.4;
}
.debug-kv .dk-key {
    color: var(--text-muted);
    white-space: nowrap;
    min-width: 70px;
}
.debug-kv .dk-val {
    color: var(--text-secondary);
    word-break: break-word;
}
.debug-list {
    margin: 4px 0;
    padding-left: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
}
.debug-list li { margin-bottom: 2px; }
.debug-json {
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 10px;
    line-height: 1.35;
    color: #8888a0;
    background: #08080f;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow-y: auto;
}
.debug-prompt {
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 10px;
    line-height: 1.4;
    color: #88bbaa;
    background: #08080f;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 300px;
    overflow-y: auto;
}
.debug-subtitle {
    font-size: 10px;
    font-weight: 600;
    color: var(--text-muted);
    margin: 8px 0 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.debug-group-label {
    font-size: 10px;
    font-weight: 600;
    color: var(--accent-dim);
    margin-top: 6px;
    margin-bottom: 2px;
}

/* Narrative panel (left) */
.narrative-panel {
    width: 280px;
    min-width: 240px;
    overflow-y: auto;
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    padding: 0;
    flex-shrink: 0;
}
.narrative-panel::-webkit-scrollbar { width: 6px; }
.narrative-panel::-webkit-scrollbar-track { background: transparent; }
.narrative-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Page metadata section */
.page-meta {
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--panel-header-bg);
}
.page-meta .meta-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 6px;
    font-size: 12px;
}
.page-meta .meta-row:last-child { margin-bottom: 0; }
.page-meta .meta-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    white-space: nowrap;
}
.page-meta .meta-value {
    color: var(--text-secondary);
    font-size: 12px;
}
.page-narrative-text {
    padding: 10px 14px;
    font-size: 12px;
    line-height: 1.5;
    color: var(--text-secondary);
    font-style: italic;
    border-bottom: 1px solid var(--border);
}

/* Character badges */
.char-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}
.char-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 10px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    white-space: nowrap;
}

/* Narrative card */
.narrative-card {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    border-left: 3px solid transparent;
    cursor: pointer;
    transition: all 0.2s;
}
.narrative-card:hover {
    background: rgba(255,255,255,0.03);
}
.narrative-card.active {
    background: var(--bg-highlight);
    border-left-color: var(--accent);
}
.narrative-card .nc-header {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.narrative-card .nc-text {
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-secondary);
}
.narrative-card .nc-chars {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    margin-top: 6px;
}
.narrative-card .nc-chars .char-badge {
    font-size: 9px;
    padding: 1px 5px;
}

/* Image panel */
.image-panel {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-dark);
    padding: 12px;
    position: relative;
    min-width: 0;
}
.img-container {
    position: relative;
    display: inline-block;
    max-width: 100%;
    max-height: 100%;
}
.img-container img {
    max-width: 100%;
    max-height: calc(100vh - 60px);
    object-fit: contain;
    border-radius: 4px;
    transition: opacity 0.2s;
    display: block;
}
.img-container img.loading { opacity: 0.3; }

/* Panel bbox overlays */
.overlay-layer {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    transition: opacity 0.2s;
}
.overlay-layer.hidden {
    opacity: 0;
    pointer-events: none !important;
}
.overlay-layer.hidden .panel-overlay {
    pointer-events: none;
}
.panel-overlay {
    position: absolute;
    border: 2px solid rgba(233, 69, 96, 0.25);
    border-radius: 2px;
    transition: all 0.25s ease;
    pointer-events: auto;
    cursor: pointer;
}
.panel-overlay:hover {
    border-color: rgba(233, 69, 96, 0.6);
    background: rgba(233, 69, 96, 0.08);
}
.panel-overlay.active {
    border-color: var(--accent);
    background: rgba(233, 69, 96, 0.12);
    box-shadow: 0 0 12px rgba(233, 69, 96, 0.3);
}
.panel-overlay .po-label {
    position: absolute;
    top: 2px;
    left: 4px;
    font-size: 9px;
    font-weight: 700;
    color: var(--accent-dim);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
}
.panel-overlay:hover .po-label,
.panel-overlay.active .po-label {
    opacity: 1;
}

/* Kumiko panel bbox overlays */
.kumiko-overlay-layer {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    transition: opacity 0.2s;
}
.kumiko-overlay-layer.hidden {
    opacity: 0;
}
.kumiko-panel-overlay {
    position: absolute;
    border: 2px solid rgba(0, 200, 220, 0.4);
    border-radius: 2px;
    pointer-events: none;
    transition: border-color 0.25s ease, background 0.25s ease;
}
.kumiko-overlay-layer:not(.hidden) .kumiko-panel-overlay {
    pointer-events: auto;
    cursor: default;
}
.kumiko-overlay-layer:not(.hidden) .kumiko-panel-overlay:hover {
    border-color: rgba(0, 200, 220, 0.7);
    background: rgba(0, 200, 220, 0.06);
}
.kumiko-panel-overlay .kpo-label {
    position: absolute;
    top: 2px;
    right: 4px;
    font-size: 9px;
    font-weight: 700;
    color: rgba(0, 200, 220, 0.6);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
}
.kumiko-overlay-layer:not(.hidden) .kumiko-panel-overlay:hover .kpo-label {
    opacity: 1;
}

/* Click zones for page navigation */
.click-zone {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 15%;
    cursor: pointer;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: opacity 0.2s;
    font-size: 32px;
    color: white;
    text-shadow: 0 0 10px rgba(0,0,0,0.8);
}
.click-zone:hover { opacity: 0.6; }
.click-zone.left { left: 0; }
.click-zone.right { right: 0; }

/* Text panel */
.text-panel {
    width: 380px;
    min-width: 320px;
    overflow-y: auto;
    background: var(--bg-panel);
    border-left: 1px solid var(--border);
    padding: 0;
    flex-shrink: 0;
}
.text-panel::-webkit-scrollbar { width: 6px; }
.text-panel::-webkit-scrollbar-track { background: transparent; }
.text-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Panel group */
.panel-group {
    border-bottom: 1px solid var(--border);
}
.panel-header {
    padding: 8px 14px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    background: var(--panel-header-bg);
    position: sticky;
    top: 0;
    z-index: 1;
}

/* Text item */
.text-item {
    padding: 8px 14px;
    border-left: 3px solid transparent;
    transition: all 0.25s ease;
    cursor: pointer;
}
.text-item:hover {
    background: rgba(255,255,255,0.03);
}
.text-item.active {
    background: var(--bg-highlight);
    border-left-color: var(--accent);
}
.text-item .speaker-name {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}
.text-item .text-content {
    font-size: 14px;
    line-height: 1.45;
    color: var(--text-primary);
}
.text-item .text-meta {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
}
/* Type-specific styling */
.text-item.type-narration .text-content,
.text-item.type-caption .text-content,
.text-item.type-internal_monologue .text-content {
    font-style: italic;
    color: var(--narration-color);
}
.text-item.type-sfx .text-content {
    font-weight: 700;
    color: var(--sfx-color);
    font-size: 13px;
}
.text-item.type-sfx .speaker-name {
    color: var(--sfx-color);
}

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
    padding: 40px;
    text-align: center;
    gap: 8px;
}
.empty-state .icon { font-size: 40px; opacity: 0.4; }
.empty-state .msg { font-size: 14px; }

/* Responsive */
@media (max-width: 1400px) {
    .debug-panel { display: none !important; }
}
@media (max-width: 1100px) {
    .narrative-panel { display: none; }
}
@media (max-width: 900px) {
    .main-content { flex-direction: column; }
    .image-panel { max-height: 55vh; }
    .text-panel { width: 100%; min-width: 0; max-height: 45vh; border-left: none; border-top: 1px solid var(--border); }
    .narrative-panel { display: none; }
}

/* Keyboard hint */
.kbd-hints {
    position: fixed;
    bottom: 8px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 11px;
    color: var(--text-muted);
    opacity: 0.5;
    pointer-events: none;
}
kbd {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 10px;
}
</style>
</head>
<body>
<div id="app">
    <!-- Top bar -->
    <div class="top-bar">
        <div class="title">Manga <span>Reader</span></div>
        <select id="chapter-select" class="chapter-select" title="Switch chapter"></select>
        <div class="nav-group">
            <button id="btn-prev" title="Previous page">&#9664;</button>
            <span class="page-info"><input type="text" id="page-input" class="page-input" value="1" title="Type a page number and press Enter"> / <span id="page-total">--</span></span>
            <button id="btn-next" title="Next page">&#9654;</button>
            <span class="page-type-badge" id="page-type"></span>
        </div>
        <div class="tts-group">
            <button id="btn-panels" title="Toggle VLM panel overlays (P)">VLM Panels</button>
            <button id="btn-kumiko" title="Toggle Kumiko panel overlays (K)" style="display:none">Kumiko</button>
            <button id="btn-debug" title="Toggle debug panel (D)">Debug</button>
            <button id="btn-read" title="Read page aloud (Space)">&#9654; Read</button>
            <button id="btn-stop" title="Stop reading (Esc)" disabled>&#9632; Stop</button>
            <div class="speed-control">
                <span>Speed</span>
                <input type="range" id="speed" min="0.5" max="2.0" step="0.1" value="1.0">
                <span class="speed-value" id="speed-val">1.0x</span>
            </div>
        </div>
    </div>

    <!-- Main content -->
    <div class="main-content">
        <div class="debug-panel" id="debug-panel"></div>
        <div class="narrative-panel" id="narrative-panel">
            <div class="empty-state">
                <div class="icon">&#128214;</div>
                <div class="msg">Loading...</div>
            </div>
        </div>
        <div class="image-panel">
            <div class="click-zone left" id="zone-prev">&#9664;</div>
            <div class="img-container" id="img-container">
                <img id="page-img" src="" alt="Manga page">
                <div class="overlay-layer" id="overlay-layer"></div>
                <div class="kumiko-overlay-layer hidden" id="kumiko-overlay-layer"></div>
            </div>
            <div class="click-zone right" id="zone-next">&#9654;</div>
        </div>
        <div class="text-panel" id="text-panel">
            <div class="empty-state">
                <div class="icon">&#128214;</div>
                <div class="msg">Loading...</div>
            </div>
        </div>
    </div>
</div>

<div class="kbd-hints">
    <kbd>&#8592;</kbd><kbd>&#8594;</kbd> navigate &nbsp;
    <kbd>Space</kbd> read/pause &nbsp;
    <kbd>Esc</kbd> stop &nbsp;
    <kbd>P</kbd> VLM panels &nbsp;
    <kbd>K</kbd> Kumiko panels
</div>

<script>
// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
    pages: [],
    currentIndex: 0,
    currentChapter: null,
    chapters: [],
    script: [],
    panels: [],
    isReading: false,
    isPaused: false,
    currentItemIdx: -1,
    activePanelId: null,
    overlaysVisible: true,
    kumikoVisible: false,
    hasKumiko: false,
    kumikoCache: {},
    debugVisible: false,
    debugChapter: null,
    debugPageCache: {},
};

// DOM refs
const $chapterSelect = document.getElementById('chapter-select');
const $img = document.getElementById('page-img');
const $pageInput = document.getElementById('page-input');
const $pageTotal = document.getElementById('page-total');
const $pageType = document.getElementById('page-type');
const $textPanel = document.getElementById('text-panel');
const $narrativePanel = document.getElementById('narrative-panel');
const $imgContainer = document.getElementById('img-container');
const $overlayLayer = document.getElementById('overlay-layer');
const $btnPrev = document.getElementById('btn-prev');
const $btnNext = document.getElementById('btn-next');
const $btnRead = document.getElementById('btn-read');
const $btnStop = document.getElementById('btn-stop');
const $btnPanels = document.getElementById('btn-panels');
const $btnKumiko = document.getElementById('btn-kumiko');
const $kumikoOverlayLayer = document.getElementById('kumiko-overlay-layer');
const $btnDebug = document.getElementById('btn-debug');
const $debugPanel = document.getElementById('debug-panel');
const $speed = document.getElementById('speed');
const $speedVal = document.getElementById('speed-val');

// ---------------------------------------------------------------------------
// Character colors (deterministic from speaker_id)
// ---------------------------------------------------------------------------
const CHAR_COLORS = {};
const COLOR_PALETTE = [
    '#e94560', '#4ecdc4', '#ff6b6b', '#95e1d3', '#f38181',
    '#aa96da', '#fcbad3', '#a8d8ea', '#ffd93d', '#6bcb77',
    '#c9b1ff', '#ff9a76', '#679b9b', '#e8a87c', '#41b3a3',
];
let colorIdx = 0;

function getCharColor(speakerId) {
    if (!speakerId) return '#88bbff';
    if (!CHAR_COLORS[speakerId]) {
        CHAR_COLORS[speakerId] = COLOR_PALETTE[colorIdx % COLOR_PALETTE.length];
        colorIdx++;
    }
    return CHAR_COLORS[speakerId];
}

// ---------------------------------------------------------------------------
// Data fetching
// ---------------------------------------------------------------------------
const scriptCache = {};

async function loadChapters() {
    const resp = await fetch('/api/chapters');
    const data = await resp.json();
    state.chapters = data.chapters;
    state.currentChapter = data.current;

    // Populate dropdown
    $chapterSelect.innerHTML = '';
    for (const ch of state.chapters) {
        const opt = document.createElement('option');
        opt.value = ch.chapter_id;
        const num = parseInt(ch.chapter_id, 10);
        const label = ch.title && ch.title !== `Chapter ${ch.chapter_id}`
            ? `Ch. ${num} - ${ch.title}`
            : `Chapter ${num}`;
        opt.textContent = label;
        if (ch.chapter_id === state.currentChapter) opt.selected = true;
        $chapterSelect.appendChild(opt);
    }
}

async function switchChapter(chapterId) {
    if (chapterId === state.currentChapter) return;
    stopReading();

    // Tell server to switch
    await fetch(`/api/set-chapter/${chapterId}`, { method: 'POST' });
    state.currentChapter = chapterId;

    // Clear caches
    for (const k in scriptCache) delete scriptCache[k];
    state.debugChapter = null;
    state.debugPageCache = {};
    state.kumikoCache = {};

    // Reload pages and go to first page
    await loadPages();
    await goToPage(0);

    // Refresh debug if visible
    if (state.debugVisible) refreshDebug();
}

async function loadPages() {
    const resp = await fetch('/api/pages');
    const data = await resp.json();
    state.pages = data.pages;
    state.hasKumiko = data.has_kumiko;
    state.kumikoCache = {};
    $btnKumiko.style.display = state.hasKumiko ? '' : 'none';
}

async function loadScript(idx) {
    if (scriptCache[idx]) return scriptCache[idx];
    const resp = await fetch(`/api/page/${idx}/script`);
    const data = await resp.json();
    scriptCache[idx] = data;
    return data;
}

async function loadKumikoPanels(idx) {
    if (!state.hasKumiko) return [];
    if (state.kumikoCache[idx] !== undefined) return state.kumikoCache[idx];
    const resp = await fetch(`/api/page/${idx}/kumiko`);
    const data = await resp.json();
    const panels = data.available ? data.panels : [];
    state.kumikoCache[idx] = panels;
    return panels;
}

// Prefetch adjacent pages
function prefetch(idx) {
    if (idx + 1 < state.pages.length) loadScript(idx + 1);
    if (idx - 1 >= 0) loadScript(idx - 1);
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
async function goToPage(idx) {
    if (idx < 0 || idx >= state.pages.length) return;

    stopReading();
    state.currentIndex = idx;
    state.activePanelId = null;
    const page = state.pages[idx];

    // Update image
    $img.classList.add('loading');
    $img.onload = () => {
        $img.classList.remove('loading');
        renderOverlays(state.panels);
        if (state.hasKumiko && state.kumikoCache[idx] !== undefined) {
            renderKumikoOverlays(state.kumikoCache[idx]);
        }
    };
    $img.src = `/manga/${state.currentChapter}/${page.image}`;

    // Update page info
    $pageInput.value = idx + 1;
    $pageTotal.textContent = state.pages.length;
    $pageType.textContent = page.page_type || '';

    // Update nav buttons
    $btnPrev.disabled = idx === 0;
    $btnNext.disabled = idx === state.pages.length - 1;

    // Load and render script
    if (page.has_metadata) {
        const data = await loadScript(idx);
        state.script = data.script || [];
        state.panels = data.panels || [];
        renderTextPanel(state.script);
        renderNarrativePanel(data);
        renderOverlays(state.panels);
        $btnRead.disabled = state.script.length === 0;
    } else {
        state.script = [];
        state.panels = [];
        renderEmptyPanel('No metadata for this page');
        renderNarrativeEmpty('No metadata for this page');
        clearOverlays();
        $btnRead.disabled = true;
    }

    // Load and render Kumiko panels (independent of VLM metadata)
    if (state.hasKumiko) {
        const kumikoData = await loadKumikoPanels(idx);
        renderKumikoOverlays(kumikoData);
    } else {
        clearKumikoOverlays();
    }

    prefetch(idx);
    refreshDebug();
}

// ---------------------------------------------------------------------------
// Text panel rendering
// ---------------------------------------------------------------------------
function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

function renderTextPanel(script) {
    if (!script || script.length === 0) {
        renderEmptyPanel('No dialogue on this page');
        return;
    }

    // Group by panel_id
    const panelGroups = [];
    let currentPanel = null;

    for (const item of script) {
        if (!currentPanel || currentPanel.panel_id !== item.panel_id) {
            currentPanel = { panel_id: item.panel_id, panel_order: item.panel_order, items: [] };
            panelGroups.push(currentPanel);
        }
        currentPanel.items.push(item);
    }

    let html = '';
    for (const group of panelGroups) {
        html += `<div class="panel-group">`;
        html += `<div class="panel-header">Panel ${group.panel_order}</div>`;

        for (const item of group.items) {
            const typeClass = `type-${item.type || 'dialog'}`;
            const color = getCharColor(item.speaker_id);

            let speakerLabel = '';
            if (item.type === 'sfx') {
                speakerLabel = 'SFX';
            } else if (item.type === 'narration' || item.type === 'caption') {
                speakerLabel = 'Narration';
            } else if (item.type === 'internal_monologue') {
                speakerLabel = (item.speaker || 'Thought') + ' (thinking)';
            } else {
                speakerLabel = item.speaker || 'Unknown';
            }

            let contentText = escapeHtml(item.content);
            if (item.type === 'dialog' || item.type === 'internal_monologue') {
                contentText = `&ldquo;${contentText}&rdquo;`;
            }

            const emotionBadge = item.emotion
                ? `<span class="text-meta">${escapeHtml(item.emotion)}</span>`
                : '';

            html += `<div class="text-item ${typeClass}" data-idx="${item.index}" id="ti-${item.index}">`;
            html += `<div class="speaker-name" style="color: ${color}">${escapeHtml(speakerLabel)}</div>`;
            html += `<div class="text-content">${contentText}</div>`;
            html += emotionBadge;
            html += `</div>`;
        }

        html += `</div>`;
    }

    $textPanel.innerHTML = html;

    // Click-to-jump on text items
    $textPanel.querySelectorAll('.text-item').forEach(el => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.idx, 10);
            if (state.isReading) {
                jumpToItem(idx);
            } else {
                startReadingFrom(idx);
            }
        });
    });
}

function renderEmptyPanel(msg) {
    $textPanel.innerHTML = `
        <div class="empty-state">
            <div class="icon">&#128214;</div>
            <div class="msg">${escapeHtml(msg)}</div>
        </div>`;
}

function renderNarrativeEmpty(msg) {
    $narrativePanel.innerHTML = `
        <div class="empty-state">
            <div class="icon">&#128214;</div>
            <div class="msg">${escapeHtml(msg)}</div>
        </div>`;
}

function renderNarrativePanel(data) {
    if (!data || !data.panels || data.panels.length === 0) {
        renderNarrativeEmpty('No panel data');
        return;
    }

    let html = '';

    // Page metadata: location, time, characters
    const locations = data.locations || [];
    const timeCtx = data.time_context;
    const chars = data.characters || [];

    if (locations.length > 0 || timeCtx || chars.length > 0) {
        html += '<div class="page-meta">';
        if (locations.length > 0) {
            html += `<div class="meta-row">
                <span class="meta-label">Location</span>
                <span class="meta-value">${escapeHtml(locations.join(', '))}</span>
            </div>`;
        }
        if (timeCtx) {
            html += `<div class="meta-row">
                <span class="meta-label">Time</span>
                <span class="meta-value">${escapeHtml(timeCtx)}</span>
            </div>`;
        }
        if (chars.length > 0) {
            html += `<div class="meta-row" style="flex-direction:column; gap:4px;">
                <span class="meta-label">Characters</span>
                <div class="char-badges">`;
            for (const ch of chars) {
                const color = getCharColor(ch.character_id);
                html += `<span class="char-badge" style="color:${color}; border-color:${color}">${escapeHtml(ch.name)}</span>`;
            }
            html += `</div></div>`;
        }
        html += '</div>';
    }

    // Page narrative
    if (data.page_narrative) {
        html += `<div class="page-narrative-text">${escapeHtml(data.page_narrative)}</div>`;
    }

    // Per-panel narrative cards
    for (const panel of data.panels) {
        const narrative = panel.panel_narrative || panel.scene_description || '';
        if (!narrative) continue;

        html += `<div class="narrative-card" data-panel-id="${escapeHtml(panel.panel_id)}" id="nc-${escapeHtml(panel.panel_id)}">`;
        html += `<div class="nc-header">Panel ${panel.panel_order}</div>`;
        html += `<div class="nc-text">${escapeHtml(narrative)}</div>`;

        // Per-panel characters with actions
        if (panel.characters_present && panel.characters_present.length > 0) {
            html += '<div class="nc-chars">';
            for (const ch of panel.characters_present) {
                const color = getCharColor(ch.character_id);
                const label = ch.action ? `${ch.name}: ${ch.action}` : ch.name;
                html += `<span class="char-badge" style="color:${color}; border-color:${color}" title="${escapeHtml(label)}">${escapeHtml(ch.name)}</span>`;
            }
            html += '</div>';
        }
        html += '</div>';
    }

    $narrativePanel.innerHTML = html;

    // Click on narrative card -> highlight panel + jump to dialogue if any
    $narrativePanel.querySelectorAll('.narrative-card').forEach(el => {
        el.addEventListener('click', () => {
            const panelId = el.dataset.panelId;
            setActivePanel(panelId);
            const firstItem = state.script.find(s => s.panel_id === panelId);
            if (firstItem) {
                if (state.isReading) {
                    jumpToItem(firstItem.index);
                } else {
                    startReadingFrom(firstItem.index);
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Panel bbox overlays
// ---------------------------------------------------------------------------
function clearOverlays() {
    $overlayLayer.innerHTML = '';
}

function renderOverlays(panels) {
    clearOverlays();
    if (!panels || panels.length === 0) return;

    for (const panel of panels) {
        const bbox = panel.bbox;
        if (!bbox || bbox.length < 4) continue;

        const [x1, y1, x2, y2] = bbox;
        const div = document.createElement('div');
        div.className = 'panel-overlay';
        div.dataset.panelId = panel.panel_id;
        div.id = `po-${panel.panel_id}`;
        div.style.left = `${x1 * 100}%`;
        div.style.top = `${y1 * 100}%`;
        div.style.width = `${(x2 - x1) * 100}%`;
        div.style.height = `${(y2 - y1) * 100}%`;

        const label = document.createElement('span');
        label.className = 'po-label';
        label.textContent = `Panel ${panel.panel_order}`;
        div.appendChild(label);

        // Click overlay -> highlight panel + jump to dialogue if any
        div.addEventListener('click', () => {
            setActivePanel(panel.panel_id);
            const firstItem = state.script.find(s => s.panel_id === panel.panel_id);
            if (firstItem) {
                if (state.isReading) {
                    jumpToItem(firstItem.index);
                } else {
                    startReadingFrom(firstItem.index);
                }
            }
        });

        $overlayLayer.appendChild(div);
    }

    // Highlight active if we have one
    if (state.activePanelId) {
        setActivePanel(state.activePanelId);
    }
}

function clearKumikoOverlays() {
    $kumikoOverlayLayer.innerHTML = '';
}

function renderKumikoOverlays(panels) {
    clearKumikoOverlays();
    if (!panels || panels.length === 0) return;

    for (const panel of panels) {
        const bbox = panel.bbox;
        if (!bbox || bbox.length < 4) continue;

        const [x1, y1, x2, y2] = bbox;
        const div = document.createElement('div');
        div.className = 'kumiko-panel-overlay';
        div.style.left = `${x1 * 100}%`;
        div.style.top = `${y1 * 100}%`;
        div.style.width = `${(x2 - x1) * 100}%`;
        div.style.height = `${(y2 - y1) * 100}%`;

        const label = document.createElement('span');
        label.className = 'kpo-label';
        label.textContent = `K${panel.panel_index + 1}`;
        div.appendChild(label);

        $kumikoOverlayLayer.appendChild(div);
    }
}

function setActivePanel(panelId) {
    state.activePanelId = panelId;

    // Update overlays
    $overlayLayer.querySelectorAll('.panel-overlay').forEach(el => {
        el.classList.toggle('active', el.dataset.panelId === panelId);
    });

    // Update narrative cards
    $narrativePanel.querySelectorAll('.narrative-card').forEach(el => {
        const isActive = el.dataset.panelId === panelId;
        el.classList.toggle('active', isActive);
        if (isActive) {
            el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });
}

function clearActivePanel() {
    state.activePanelId = null;
    $overlayLayer.querySelectorAll('.panel-overlay.active').forEach(el => el.classList.remove('active'));
    $narrativePanel.querySelectorAll('.narrative-card.active').forEach(el => el.classList.remove('active'));
}

// ---------------------------------------------------------------------------
// TTS Engine (Web Speech API)
// ---------------------------------------------------------------------------
function getRate() {
    return parseFloat($speed.value);
}

function speak(text, opts = {}) {
    return new Promise((resolve) => {
        // Handle single punctuation like "?" or "!"
        let ttsText = text;
        if (ttsText === '?') ttsText = 'Huh?';
        else if (ttsText === '!') ttsText = 'Oh!';
        else if (ttsText === '...') ttsText = 'Hmm...';
        else if (ttsText === '!!') ttsText = 'What?!';

        const utter = new SpeechSynthesisUtterance(ttsText);
        utter.rate = opts.rate || getRate();
        utter.pitch = opts.pitch || 1.0;
        utter.volume = opts.volume || 1.0;
        utter.lang = 'en-US';

        utter.onend = () => resolve();
        utter.onerror = (e) => {
            if (e.error !== 'canceled') resolve();
        };

        speechSynthesis.speak(utter);
    });
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function getProsody(item) {
    const opts = {};
    const bubble = item.speech_bubble_type;
    if (bubble === 'shout') {
        opts.rate = getRate() * 1.1;
        opts.volume = 1.0;
    } else if (bubble === 'whisper') {
        opts.rate = getRate() * 0.9;
        opts.volume = 0.4;
    } else if (bubble === 'thought') {
        opts.rate = getRate() * 0.95;
        opts.pitch = 0.9;
    }
    if (item.type === 'narration' || item.type === 'caption') {
        opts.rate = (opts.rate || getRate()) * 0.95;
        opts.pitch = 0.85;
    }
    return opts;
}

function highlightItem(idx) {
    // Remove previous highlight
    $textPanel.querySelectorAll('.text-item.active').forEach(el => el.classList.remove('active'));
    const el = document.getElementById(`ti-${idx}`);
    if (el) {
        el.classList.add('active');
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    state.currentItemIdx = idx;

    // Update active panel highlight (overlay + narrative)
    const item = state.script[idx];
    if (item && item.panel_id !== state.activePanelId) {
        setActivePanel(item.panel_id);
    }
}

function clearHighlight() {
    $textPanel.querySelectorAll('.text-item.active').forEach(el => el.classList.remove('active'));
    state.currentItemIdx = -1;
    clearActivePanel();
}

async function startReadingFrom(startIdx) {
    if (state.script.length === 0) return;

    state.isReading = true;
    state.isPaused = false;
    $btnRead.disabled = true;
    $btnRead.textContent = '\u23F8 Pause';
    $btnRead.disabled = false;
    $btnRead.classList.add('active');
    $btnStop.disabled = false;

    let prevPanelId = null;

    for (let i = startIdx; i < state.script.length; i++) {
        if (!state.isReading) break;

        // Wait if paused
        while (state.isPaused && state.isReading) {
            await sleep(100);
        }
        if (!state.isReading) break;

        const item = state.script[i];

        // Panel transition pause
        if (prevPanelId !== null && item.panel_id !== prevPanelId) {
            await sleep(500);
        }
        if (!state.isReading) break;

        // Highlight and speak
        highlightItem(item.index);
        await speak(item.content, getProsody(item));

        prevPanelId = item.panel_id;

        // Brief pause between items
        if (state.isReading && i < state.script.length - 1) {
            await sleep(200);
        }
    }

    finishReading();
}

function readPage() {
    if (state.isPaused) {
        // Resume
        state.isPaused = false;
        speechSynthesis.resume();
        $btnRead.textContent = '\u23F8 Pause';
        return;
    }
    if (state.isReading) {
        // Pause
        state.isPaused = true;
        speechSynthesis.pause();
        $btnRead.textContent = '\u25B6 Resume';
        return;
    }
    startReadingFrom(0);
}

function jumpToItem(idx) {
    speechSynthesis.cancel();
    state.isReading = false;
    state.isPaused = false;
    // Short delay then restart from the clicked position
    setTimeout(() => startReadingFrom(idx), 50);
}

function stopReading() {
    state.isReading = false;
    state.isPaused = false;
    speechSynthesis.cancel();
    finishReading();
}

function finishReading() {
    state.isReading = false;
    state.isPaused = false;
    clearHighlight();
    $btnRead.textContent = '\u25B6 Read';
    $btnRead.classList.remove('active');
    $btnRead.disabled = state.script.length === 0;
    $btnStop.disabled = true;
}

// ---------------------------------------------------------------------------
// Debug panel
// ---------------------------------------------------------------------------
async function loadDebugChapter() {
    if (state.debugChapter) return state.debugChapter;
    const resp = await fetch('/api/debug/chapter');
    state.debugChapter = await resp.json();
    return state.debugChapter;
}

async function loadDebugPage(idx) {
    if (state.debugPageCache[idx]) return state.debugPageCache[idx];
    const resp = await fetch(`/api/debug/page/${idx}`);
    const data = await resp.json();
    state.debugPageCache[idx] = data;
    return data;
}

function renderDebugPanel(chapter, page) {
    let html = '';

    // Chapter info section
    if (chapter) {
        const pm = chapter.processing_metadata || {};
        const stats = chapter.statistics || {};

        html += `<div class="debug-section">
            <div class="debug-section-header" onclick="this.parentElement.classList.toggle('collapsed')">
                Chapter Info <span class="chevron">&#9660;</span>
            </div>
            <div class="debug-body">
                <div class="debug-kv"><span class="dk-key">Title</span><span class="dk-val">${escapeHtml(chapter.chapter_title || '')}</span></div>
                <div class="debug-kv"><span class="dk-key">Arc</span><span class="dk-val">${escapeHtml(chapter.arc || '')} / ${escapeHtml(chapter.saga || '')}</span></div>
                <div class="debug-kv"><span class="dk-key">Model</span><span class="dk-val">${escapeHtml(pm.model_used || '')}</span></div>
                <div class="debug-kv"><span class="dk-key">Tokens</span><span class="dk-val">${(pm.total_tokens_used || 0).toLocaleString()}</span></div>
                <div class="debug-kv"><span class="dk-key">Time</span><span class="dk-val">${Math.round((pm.processing_time_seconds || 0) / 60)}m</span></div>
                <div class="debug-kv"><span class="dk-key">Wiki</span><span class="dk-val">${pm.wiki_grounded ? 'Yes' : 'No'}</span></div>
                <div class="debug-kv"><span class="dk-key">Date</span><span class="dk-val">${escapeHtml(pm.processing_date || '')}</span></div>
                <div class="debug-subtitle">Statistics</div>
                <div class="debug-kv"><span class="dk-key">Panels</span><span class="dk-val">${stats.total_panels || 0}</span></div>
                <div class="debug-kv"><span class="dk-key">Dialog</span><span class="dk-val">${stats.total_dialog_lines || 0}</span></div>
                <div class="debug-kv"><span class="dk-key">Narration</span><span class="dk-val">${stats.total_narration_lines || 0}</span></div>
                <div class="debug-kv"><span class="dk-key">SFX</span><span class="dk-val">${stats.total_sfx || 0}</span></div>
                <div class="debug-kv"><span class="dk-key">Characters</span><span class="dk-val">${stats.unique_characters || 0}</span></div>
            </div>
        </div>`;

        // Wiki grounding section
        const wiki = chapter.wiki_context;
        if (wiki) {
            html += `<div class="debug-section collapsed">
                <div class="debug-section-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    Wiki Grounding <span class="chevron">&#9660;</span>
                </div>
                <div class="debug-body">`;

            if (wiki.synopsis) {
                html += `<div class="debug-subtitle">Chapter Synopsis (VLM input)</div>
                    <div class="debug-prompt">${escapeHtml(wiki.synopsis)}</div>`;
            }
            if (wiki.chapter_notes && wiki.chapter_notes.length > 0) {
                html += `<div class="debug-subtitle">Key Events (VLM input)</div><ul class="debug-list">`;
                for (const note of wiki.chapter_notes) {
                    html += `<li>${escapeHtml(note)}</li>`;
                }
                html += '</ul>';
            }
            if (wiki.character_groups) {
                html += `<div class="debug-subtitle">Character Groups (VLM input)</div>`;
                for (const [group, names] of Object.entries(wiki.character_groups)) {
                    html += `<div class="debug-group-label">${escapeHtml(group)}</div>`;
                    html += `<div style="color:var(--text-secondary);margin-bottom:4px">${escapeHtml(names.join(', '))}</div>`;
                }
            }
            if (wiki.cover_page) {
                html += `<div class="debug-subtitle">Cover Page</div>
                    <div style="color:var(--text-secondary)">${escapeHtml(wiki.cover_page)}</div>`;
            }
            html += '</div></div>';
        }

        // Character registry section
        const reg = chapter.character_registry;
        if (reg && reg.characters) {
            html += `<div class="debug-section collapsed">
                <div class="debug-section-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    Character Registry (${Object.keys(reg.characters).length}) <span class="chevron">&#9660;</span>
                </div>
                <div class="debug-body">`;
            for (const [cid, cdata] of Object.entries(reg.characters)) {
                const color = getCharColor(cid);
                const appCount = (cdata.appearances || []).length;
                const emotions = (cdata.voice_profile && cdata.voice_profile.observed_emotions) || [];
                html += `<div style="margin-bottom:8px;border-left:2px solid ${color};padding-left:8px">
                    <div style="color:${color};font-weight:600;font-size:11px">${escapeHtml(cdata.name || cid)}</div>
                    <div class="debug-kv"><span class="dk-key">Pages</span><span class="dk-val">${appCount}</span></div>`;
                if (cdata.visual_description) {
                    html += `<div class="debug-kv"><span class="dk-key">Visual</span><span class="dk-val">${escapeHtml(cdata.visual_description)}</span></div>`;
                }
                if (cdata.voice_profile && cdata.voice_profile.suggested_voice_style) {
                    html += `<div class="debug-kv"><span class="dk-key">Voice</span><span class="dk-val">${escapeHtml(cdata.voice_profile.suggested_voice_style)}</span></div>`;
                }
                if (emotions.length > 0) {
                    html += `<div class="debug-kv"><span class="dk-key">Emotions</span><span class="dk-val">${escapeHtml(emotions.slice(0, 8).join(', '))}${emotions.length > 8 ? '...' : ''}</span></div>`;
                }
                html += '</div>';
            }
            html += '</div></div>';
        }

        // Locations section
        if (chapter.locations && chapter.locations.length > 0) {
            html += `<div class="debug-section collapsed">
                <div class="debug-section-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    Locations (${chapter.locations.length}) <span class="chevron">&#9660;</span>
                </div>
                <div class="debug-body">
                    <div style="color:var(--text-secondary)">${escapeHtml(chapter.locations.join(', '))}</div>
                </div>
            </div>`;
        }
    }

    // Per-page section
    if (page) {
        // System prompt (same for all pages, collapsed by default)
        if (page.system_prompt_preview) {
            html += `<div class="debug-section collapsed">
                <div class="debug-section-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    System Prompt <span class="chevron">&#9660;</span>
                </div>
                <div class="debug-body">
                    <div class="debug-prompt">${escapeHtml(page.system_prompt_preview)}</div>
                </div>
            </div>`;
        }

        // User prompt (per-page, open by default)
        html += `<div class="debug-section">
            <div class="debug-section-header" onclick="this.parentElement.classList.toggle('collapsed')">
                Page ${state.currentIndex + 1} User Prompt <span class="chevron">&#9660;</span>
            </div>
            <div class="debug-body">
                <div class="debug-prompt">${escapeHtml(page.user_prompt_preview || 'N/A')}</div>
            </div>
        </div>`;

        html += `<div class="debug-section collapsed">
            <div class="debug-section-header" onclick="this.parentElement.classList.toggle('collapsed')">
                Page ${state.currentIndex + 1} Raw JSON <span class="chevron">&#9660;</span>
            </div>
            <div class="debug-body">
                <div class="debug-json">${page.raw_metadata ? escapeHtml(JSON.stringify(page.raw_metadata, null, 2)) : 'No metadata'}</div>
            </div>
        </div>`;
    }

    $debugPanel.innerHTML = html;
}

async function refreshDebug() {
    if (!state.debugVisible) return;
    const [chapter, page] = await Promise.all([
        loadDebugChapter(),
        loadDebugPage(state.currentIndex),
    ]);
    renderDebugPanel(chapter, page);
}

function toggleDebug() {
    state.debugVisible = !state.debugVisible;
    $debugPanel.classList.toggle('visible', state.debugVisible);
    $btnDebug.classList.toggle('active', state.debugVisible);
    if (state.debugVisible) {
        refreshDebug();
    }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------
$btnPrev.addEventListener('click', () => goToPage(state.currentIndex - 1));
$btnNext.addEventListener('click', () => goToPage(state.currentIndex + 1));
document.getElementById('zone-prev').addEventListener('click', () => goToPage(state.currentIndex - 1));
document.getElementById('zone-next').addEventListener('click', () => goToPage(state.currentIndex + 1));
$btnRead.addEventListener('click', readPage);
$btnStop.addEventListener('click', stopReading);

// Page jump input
$pageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        const num = parseInt($pageInput.value, 10);
        if (!isNaN(num) && num >= 1 && num <= state.pages.length) {
            goToPage(num - 1);
        } else {
            $pageInput.value = state.currentIndex + 1;
        }
        $pageInput.blur();
    } else if (e.key === 'Escape') {
        $pageInput.value = state.currentIndex + 1;
        $pageInput.blur();
    }
});
$pageInput.addEventListener('focus', () => $pageInput.select());

// Panel overlays toggle
$btnPanels.addEventListener('click', () => {
    state.overlaysVisible = !state.overlaysVisible;
    $overlayLayer.classList.toggle('hidden', !state.overlaysVisible);
    $btnPanels.classList.toggle('active', state.overlaysVisible);
});
$btnPanels.classList.add('active');

// Kumiko overlays toggle
$btnKumiko.addEventListener('click', () => {
    state.kumikoVisible = !state.kumikoVisible;
    $kumikoOverlayLayer.classList.toggle('hidden', !state.kumikoVisible);
    $btnKumiko.classList.toggle('active', state.kumikoVisible);
});

// Debug panel toggle
$btnDebug.addEventListener('click', toggleDebug);

// Chapter selector
$chapterSelect.addEventListener('change', () => switchChapter($chapterSelect.value));

$speed.addEventListener('input', () => {
    $speedVal.textContent = parseFloat($speed.value).toFixed(1) + 'x';
});

document.addEventListener('keydown', (e) => {
    // Don't capture if focused on an input or select
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    switch (e.key) {
        case 'ArrowLeft':
            e.preventDefault();
            goToPage(state.currentIndex - 1);
            break;
        case 'ArrowRight':
            e.preventDefault();
            goToPage(state.currentIndex + 1);
            break;
        case ' ':
            e.preventDefault();
            readPage();
            break;
        case 'Escape':
            e.preventDefault();
            stopReading();
            break;
        case 'p':
            $btnPanels.click();
            break;
        case 'k':
            if (state.hasKumiko) $btnKumiko.click();
            break;
        case 'd':
            toggleDebug();
            break;
    }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
(async function init() {
    await loadChapters();
    await loadPages();
    await goToPage(0);
})();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import webbrowser
    import threading

    parser = argparse.ArgumentParser(description="Manga Reader")
    parser.add_argument("--chapter", type=str, default=None,
                        help="Chapter ID to open on startup (e.g. 0002)")
    args = parser.parse_args()

    # Switch to requested chapter if specified and valid
    if args.chapter and any(c['chapter_id'] == args.chapter for c in AVAILABLE_CHAPTERS):
        set_chapter(args.chapter)
        print(f"Starting with chapter {args.chapter}")

    url = f"http://localhost:{PORT}"
    print(f"Starting Manga Reader at {url}")

    # Open browser after a short delay
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
