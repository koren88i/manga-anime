"""
Alignment Viewer -- Interactive review UI for manga-to-anime alignment results.

Browse alignment records side-by-side: manga panels vs anime keyframes, dialogue
comparison, confidence scores, speaker/location match indicators, and gap review.
Flask + embedded HTML5/CSS/JS (same pattern as reader.py, anime_viewer.py).

Layout:
  Top bar: episode selector, title, stats (anchored/interpolated counts, avg
    confidence, arc badge), gap mode toggle, keyboard shortcut hints.
  Left panel (260px, toggleable): alignment stats, filter/sort controls,
    scrollable record list with confidence dots and tier badges.
  Center: side-by-side manga panel + anime keyframe comparison, dialogue diff
    below (manga vs anime text with word-level highlighting), manga context
    window strip (before/after panels as thumbnails).
  Right panel (360px, scrollable): alignment detail -- confidence bar, OpenAI/BOW
    scores, speaker/location/time match indicators, time range, explanation text,
    collapsible transformation fields.
  Bottom: timeline bar of all records colored by confidence tier, click to navigate.

Gap mode (G key): replaces record list with gap list, shows bounding anchors and
  unanchored manga/anime content for each gap.

Keyboard: Up/Down or J/K navigate records, G toggle gaps, O toggle overview,
  1/2/3 filter by tier, C clear filters, Home/End first/last.

Data: loads alignment_manifest.json + alignment_index.json on startup; loads
  pass1_anchors.json + pass1_gaps.json on gap mode toggle. Resolves keyframe
  paths from shot_index.json and dialogue text from chapter/episode manifests.

Run: python alignment_viewer.py [--episode 0001] [--port 7863] [--no-browser]
Opens at http://localhost:7863
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import re
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, send_file, render_template_string

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
ALIGNMENT_DIR = BASE_DIR / "output" / "alignment"
ANIME_META_DIR = BASE_DIR / "output" / "anime_metadata"
MANGA_META_DIR = BASE_DIR / "output" / "manga_metadata"
MANGA_DIR = BASE_DIR / "manga"
PORT = 7863

app = Flask(__name__)

# Cache: episode_id -> {manifest, index, anchors, gaps, summary}
alignment_cache = {}
startup_episode = "0001"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_alignments():
    """Scan output/alignment/ for episode dirs with alignment_manifest.json."""
    episodes = []
    if not ALIGNMENT_DIR.exists():
        return episodes

    for d in sorted(ALIGNMENT_DIR.iterdir()):
        if not d.is_dir() or not re.match(r"^\d{4}$", d.name):
            continue
        ep_id = d.name
        manifest_path = d / "alignment_manifest.json"
        if not manifest_path.exists():
            continue

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            records = m.get("records", [])
            tier_counts = {}
            total_conf = 0
            for r in records:
                t = r.get("confidence_tier", 0)
                tier_counts[t] = tier_counts.get(t, 0) + 1
                total_conf += r.get("confidence", 0)
            episodes.append({
                "episode_id": ep_id,
                "chapter_ids": m.get("chapter_ids", []),
                "arc_id": m.get("arc_id", ""),
                "total_records": len(records),
                "tier_counts": tier_counts,
                "avg_confidence": round(total_conf / len(records), 3) if records else 0,
            })
        except Exception:
            continue
    return episodes


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_cache(episode_id):
    if episode_id not in alignment_cache:
        alignment_cache[episode_id] = {
            "manifest": None,
            "index": None,
            "anchors": None,
            "gaps": None,
            "summary": None,
            "shot_keyframes": None,
            "manga_dialogue": None,
            "anime_dialogue": None,
        }
    return alignment_cache[episode_id]


def _load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_manifest(episode_id):
    cache = get_cache(episode_id)
    if cache["manifest"] is None:
        cache["manifest"] = _load_json(ALIGNMENT_DIR / episode_id / "alignment_manifest.json")
    return cache["manifest"]


def load_index(episode_id):
    cache = get_cache(episode_id)
    if cache["index"] is None:
        cache["index"] = _load_json(ALIGNMENT_DIR / episode_id / "alignment_index.json")
    return cache["index"]


def load_anchors(episode_id):
    cache = get_cache(episode_id)
    if cache["anchors"] is None:
        cache["anchors"] = _load_json(ALIGNMENT_DIR / episode_id / "pass1_anchors.json")
    return cache["anchors"]


def load_gaps(episode_id):
    cache = get_cache(episode_id)
    if cache["gaps"] is None:
        cache["gaps"] = _load_json(ALIGNMENT_DIR / episode_id / "pass1_gaps.json")
    return cache["gaps"]


def load_shot_keyframes(episode_id):
    """Build shot_id -> primary keyframe filename map from shot_index.json."""
    cache = get_cache(episode_id)
    if cache["shot_keyframes"] is None:
        mapping = {}
        # Try standard dir first, then {episode_id}_baseline
        for suffix in ["", "_baseline"]:
            shot_path = ANIME_META_DIR / f"{episode_id}{suffix}" / "shot_index.json"
            if shot_path.exists():
                shots = _load_json(shot_path)
                if shots:
                    for s in shots:
                        sid = s.get("shot_id")
                        kf_paths = s.get("keyframe_paths", [])
                        if sid is not None and kf_paths:
                            # Extract just the filename from the path
                            raw = kf_paths[0].replace("\\", "/")
                            fname = raw.split("/")[-1]
                            mapping[int(sid)] = fname
                break
        cache["shot_keyframes"] = mapping
    return cache["shot_keyframes"]


def load_manga_dialogue(episode_id):
    """Load manga dialogue index for the chapters aligned to this episode."""
    cache = get_cache(episode_id)
    if cache["manga_dialogue"] is None:
        manifest = load_manifest(episode_id)
        dialogue = {}
        if manifest:
            for ch_id in manifest.get("chapter_ids", []):
                ch_manifest = _load_json(MANGA_META_DIR / ch_id / "chapter_manifest.json")
                if ch_manifest:
                    for d in ch_manifest.get("dialogue_index", []):
                        dialogue[d["seq"]] = d
        cache["manga_dialogue"] = dialogue
    return cache["manga_dialogue"]


def load_anime_dialogue(episode_id):
    """Load anime dialogue index for this episode."""
    cache = get_cache(episode_id)
    if cache["anime_dialogue"] is None:
        dialogue = {}
        for suffix in ["", "_baseline"]:
            ep_manifest_path = ANIME_META_DIR / f"{episode_id}{suffix}" / "episode_manifest.json"
            if ep_manifest_path.exists():
                ep_manifest = _load_json(ep_manifest_path)
                if ep_manifest:
                    for d in ep_manifest.get("dialogue_index", []):
                        dialogue[d["seq"]] = d
                break
        cache["anime_dialogue"] = dialogue
    return cache["anime_dialogue"]


def build_summary(episode_id):
    """Build enriched summary: index records + resolved keyframes + dialogue text."""
    cache = get_cache(episode_id)
    if cache["summary"] is not None:
        return cache["summary"]

    manifest = load_manifest(episode_id)
    if not manifest:
        return None

    shot_kf = load_shot_keyframes(episode_id)
    manga_dlg = load_manga_dialogue(episode_id)
    anime_dlg = load_anime_dialogue(episode_id)

    enriched_records = []
    for rec in manifest.get("records", []):
        manga_side = rec.get("manga_side", {})
        anime_side = rec.get("anime_side", {})

        # Resolve keyframe filenames for each shot
        kf_files = []
        for sid in anime_side.get("shot_ids", []):
            sid_int = int(sid) if sid is not None else None
            if sid_int is not None and sid_int in shot_kf:
                kf_files.append(shot_kf[sid_int])
        # Resolve panel image paths
        panel_images = manga_side.get("panel_image_paths", [])

        # Resolve dialogue text
        manga_lines = []
        for seq in manga_side.get("dialogue_line_ids", []):
            d = manga_dlg.get(seq)
            if d:
                manga_lines.append({
                    "seq": seq,
                    "content": d.get("content", ""),
                    "speaker": d.get("speaker", ""),
                    "speaker_id": d.get("speaker_id", ""),
                    "emotion": d.get("emotion", ""),
                })

        anime_lines = []
        for seq in anime_side.get("dialogue_line_ids", []):
            d = anime_dlg.get(seq)
            if d:
                anime_lines.append({
                    "seq": seq,
                    "content": d.get("content", ""),
                    "speaker": d.get("speaker", ""),
                    "speaker_id": d.get("speaker_id", ""),
                    "emotion": d.get("emotion", ""),
                })

        enriched_records.append({
            "alignment_id": rec.get("alignment_id"),
            "arc_id": rec.get("arc_id"),
            "anchor_type": rec.get("anchor_type"),
            "confidence": rec.get("confidence"),
            "confidence_tier": rec.get("confidence_tier"),
            "score_openai": rec.get("score_openai"),
            "score_bow": rec.get("score_bow"),
            "speaker_match": rec.get("speaker_match"),
            "location_match": rec.get("location_match"),
            "time_context_match": rec.get("time_context_match"),
            "alignment_explanation": rec.get("alignment_explanation"),
            # Manga side
            "chapter_id": manga_side.get("chapter_id"),
            "panel_ids": manga_side.get("panel_ids", []),
            "page_range": manga_side.get("page_range"),
            "panel_images": panel_images,
            "manga_lines": manga_lines,
            "manga_context": manga_side.get("manga_context_window", {}),
            # Anime side
            "episode_id": anime_side.get("episode_id"),
            "scene_numbers": anime_side.get("scene_numbers", []),
            "shot_ids": anime_side.get("shot_ids", []),
            "time_range": anime_side.get("time_range"),
            "keyframe_files": kf_files,
            "anime_lines": anime_lines,
            # Transformation
            "transformation_delta": rec.get("transformation_delta"),
            "transformation_analysis": rec.get("transformation_analysis"),
            "imperative_delta_prompt": rec.get("imperative_delta_prompt"),
            "anime_original_segments": rec.get("anime_original_segments", []),
        })

    summary = {
        "episode_id": episode_id,
        "chapter_ids": manifest.get("chapter_ids", []),
        "arc_id": manifest.get("arc_id", ""),
        "records": enriched_records,
    }
    cache["summary"] = summary
    return summary


# ---------------------------------------------------------------------------
# Routes: API
# ---------------------------------------------------------------------------

@app.route("/api/episodes")
def api_episodes():
    eps = discover_alignments()
    return jsonify({"episodes": eps, "current": startup_episode})


@app.route("/api/alignment/<episode_id>/summary")
def api_summary(episode_id):
    s = build_summary(episode_id)
    if not s:
        return jsonify({"error": "Not found"}), 404
    return jsonify(s)


@app.route("/api/alignment/<episode_id>/anchors")
def api_anchors(episode_id):
    a = load_anchors(episode_id)
    if not a:
        return jsonify({"error": "Not found"}), 404
    return jsonify(a)


@app.route("/api/alignment/<episode_id>/gaps")
def api_gaps(episode_id):
    g = load_gaps(episode_id)
    if not g:
        return jsonify({"error": "Not found"}), 404
    return jsonify(g)


# ---------------------------------------------------------------------------
# Routes: Image serving
# ---------------------------------------------------------------------------

@app.route("/panel/<chapter_id>/<path:filename>")
def serve_panel(chapter_id, filename):
    """Serve manga panel images."""
    filepath = MANGA_META_DIR / chapter_id / "panels" / filename
    if filepath.exists():
        return send_file(str(filepath), mimetype="image/png")
    return "Not found", 404


@app.route("/keyframe/<episode_id>/<path:filename>")
def serve_keyframe(episode_id, filename):
    """Serve anime keyframe images."""
    for suffix in ["", "_baseline"]:
        filepath = ANIME_META_DIR / f"{episode_id}{suffix}" / "keyframes" / filename
        if filepath.exists():
            return send_file(str(filepath), mimetype="image/jpeg")
    return "Not found", 404


@app.route("/manga-page/<chapter_id>/<path:filename>")
def serve_manga_page(chapter_id, filename):
    """Serve full manga page images."""
    filepath = MANGA_DIR / chapter_id / filename
    if filepath.exists():
        mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        return send_file(str(filepath), mimetype=mime)
    return "Not found", 404


# ---------------------------------------------------------------------------
# Routes: Main page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, episode_id=startup_episode)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alignment Viewer</title>
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
    --tier1: #2a6a9a;
    --tier2: #9a8a3a;
    --tier3: #9a3a3a;
    --match-yes: #3a9a5a;
    --match-no: #9a3a3a;
    --manga-color: #88bbff;
    --anime-color: #ffaa66;
    --panel-header-bg: #12121f;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg-dark);
    color: var(--text-primary);
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Layout */
#app { display: flex; flex-direction: column; height: 100vh; }

/* Top bar */
.top-bar {
    display: flex;
    align-items: center;
    padding: 6px 14px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: 12px;
    flex-shrink: 0;
    z-index: 10;
}
.top-bar select {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 13px;
}
.top-bar .title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    white-space: nowrap;
}
.top-bar .title span { color: var(--accent); }
.top-bar .stats {
    font-size: 12px;
    color: var(--text-muted);
    white-space: nowrap;
}
.badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-tier1 { background: var(--tier1); color: #cce; }
.badge-tier2 { background: var(--tier2); color: #eec; }
.badge-tier3 { background: var(--tier3); color: #ecc; }
.badge-anchor { background: #2a5a3a; color: #aea; }
.badge-interp { background: #5a4a2a; color: #eda; }
.top-bar .shortcuts {
    margin-left: auto;
    font-size: 11px;
    color: var(--text-muted);
}
.shortcuts kbd {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10px;
    font-family: inherit;
}

/* Main content */
.main-content {
    display: flex;
    flex: 1;
    overflow: hidden;
}

/* Left panel */
.left-panel {
    width: 260px;
    overflow-y: auto;
    flex-shrink: 0;
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    font-size: 12px;
    display: flex;
    flex-direction: column;
}
.left-panel.hidden { display: none; }
.panel-section {
    border-bottom: 1px solid var(--border);
    padding: 10px;
}
.panel-section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    color: var(--text-secondary);
}
.stat-row .val { color: var(--text-primary); font-weight: 600; }

/* Filter controls */
.filter-row {
    display: flex;
    gap: 4px;
    margin-bottom: 6px;
    flex-wrap: wrap;
}
.filter-btn {
    padding: 2px 8px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--bg-card);
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s;
}
.filter-btn:hover { background: var(--bg-highlight); }
.filter-btn.active {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--bg-highlight);
}
.sort-select {
    width: 100%;
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 3px 6px;
    border-radius: 3px;
    font-size: 11px;
    margin-top: 4px;
}

/* Record list */
.record-list {
    flex: 1;
    overflow-y: auto;
    padding: 4px;
}
.record-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 6px;
    border-radius: 3px;
    cursor: pointer;
    margin-bottom: 1px;
    border-left: 3px solid transparent;
    transition: background 0.1s;
}
.record-item:hover { background: var(--bg-card); }
.record-item.active {
    background: var(--bg-highlight);
    border-left-color: var(--accent);
}
.conf-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.record-item .rec-id {
    font-size: 11px;
    color: var(--text-muted);
    width: 36px;
    text-align: right;
    font-family: 'Consolas', monospace;
}
.record-item .rec-summary {
    flex: 1;
    font-size: 11px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.record-item .rec-badge {
    font-size: 9px;
    padding: 0 4px;
    border-radius: 2px;
    font-weight: 600;
}

/* Center panel */
.center-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    overflow-y: auto;
    background: var(--bg-dark);
}

/* Comparison area */
.comparison-grid {
    display: flex;
    gap: 12px;
    padding: 14px;
    justify-content: center;
    align-items: flex-start;
}
.compare-side {
    flex: 1;
    max-width: 400px;
    text-align: center;
}
.compare-side .side-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
    padding: 3px 8px;
    border-radius: 3px;
}
.side-manga .side-label { color: var(--manga-color); background: rgba(136,187,255,0.1); }
.side-anime .side-label { color: var(--anime-color); background: rgba(255,170,102,0.1); }
.compare-side img {
    max-width: 100%;
    max-height: 320px;
    border-radius: 4px;
    border: 1px solid var(--border);
    object-fit: contain;
    background: #111;
}
.compare-side .no-image {
    width: 100%;
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-muted);
    font-size: 12px;
}
.multi-thumb-strip {
    display: flex;
    gap: 4px;
    justify-content: center;
    margin-top: 4px;
}
.multi-thumb-strip img {
    width: 60px;
    height: 40px;
    object-fit: cover;
    border-radius: 2px;
    border: 1px solid var(--border);
    cursor: pointer;
    opacity: 0.6;
    transition: opacity 0.15s;
}
.multi-thumb-strip img:hover,
.multi-thumb-strip img.active { opacity: 1; border-color: var(--accent); }

/* Dialogue comparison */
.dialogue-section {
    padding: 0 14px 10px;
}
.dialogue-section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.dialogue-comparison {
    display: flex;
    gap: 12px;
}
.dialogue-col {
    flex: 1;
    min-width: 0;
}
.dialogue-col-header {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 4px 8px;
    border-radius: 3px 3px 0 0;
    margin-bottom: 0;
}
.dcol-manga .dialogue-col-header { color: var(--manga-color); background: rgba(136,187,255,0.08); }
.dcol-anime .dialogue-col-header { color: var(--anime-color); background: rgba(255,170,102,0.08); }
.dialogue-line {
    padding: 4px 8px;
    border-left: 2px solid var(--border);
    margin-bottom: 3px;
    font-size: 12px;
}
.dialogue-line .speaker {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 1px;
}
.dialogue-line .content { color: var(--text-secondary); }
.dialogue-line .emotion {
    font-size: 10px;
    color: var(--text-muted);
    font-style: italic;
    margin-top: 1px;
}

/* Context strip */
.context-section {
    padding: 0 14px 14px;
}
.context-strip {
    display: flex;
    gap: 4px;
    align-items: center;
    justify-content: center;
}
.context-strip .ctx-thumb {
    width: 60px;
    height: 80px;
    object-fit: cover;
    border-radius: 3px;
    border: 1px solid var(--border);
    opacity: 0.5;
    transition: opacity 0.15s;
    cursor: pointer;
}
.context-strip .ctx-thumb:hover { opacity: 0.8; }
.context-strip .ctx-thumb.current {
    opacity: 1;
    border-color: var(--accent);
    box-shadow: 0 0 6px var(--accent-dim);
}
.context-strip .ctx-divider {
    width: 2px;
    height: 40px;
    background: var(--accent);
    margin: 0 4px;
    border-radius: 1px;
}

/* Right panel */
.right-panel {
    width: 360px;
    overflow-y: auto;
    flex-shrink: 0;
    background: var(--bg-panel);
    border-left: 1px solid var(--border);
    font-size: 12px;
}
.detail-header {
    padding: 10px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
}
.detail-header .align-id {
    font-weight: 700;
    font-family: 'Consolas', monospace;
}
.detail-section {
    padding: 10px;
    border-bottom: 1px solid var(--border);
}
.detail-section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.score-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 10px;
    align-items: center;
}
.score-grid .label {
    color: var(--text-secondary);
    font-size: 11px;
}
.score-grid .value {
    font-weight: 600;
    font-size: 12px;
}
.conf-bar-outer {
    height: 6px;
    background: var(--bg-card);
    border-radius: 3px;
    overflow: hidden;
    width: 100%;
}
.conf-bar-inner {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}
.match-icon {
    font-size: 12px;
    font-weight: 700;
}
.match-yes { color: var(--match-yes); }
.match-no { color: var(--match-no); }
.explanation-text {
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.5;
    font-style: italic;
}

/* Collapsible */
.collapsible-header {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 0;
}
.collapsible-header:hover { color: var(--text-primary); }
.collapsible-header .arrow {
    font-size: 10px;
    transition: transform 0.2s;
    color: var(--text-muted);
}
.collapsible-header .arrow.open { transform: rotate(90deg); }
.collapsible-body {
    display: none;
    padding: 6px 0;
}
.collapsible-body.open { display: block; }
.collapsible-body pre {
    background: var(--bg-dark);
    padding: 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'Consolas', monospace;
    color: var(--text-secondary);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow-y: auto;
}

/* Timeline */
.timeline-bar {
    flex-shrink: 0;
    background: var(--bg-panel);
    border-top: 1px solid var(--border);
    padding: 6px 10px;
    display: flex;
    align-items: center;
    gap: 1px;
    overflow-x: auto;
    min-height: 32px;
}
.timeline-bar .tl-label {
    font-size: 10px;
    color: var(--text-muted);
    margin-right: 6px;
    white-space: nowrap;
    flex-shrink: 0;
}
.timeline-blocks {
    display: flex;
    gap: 1px;
    flex: 1;
    min-width: 0;
    align-items: center;
}
.tl-block {
    height: 16px;
    min-width: 3px;
    border-radius: 2px;
    cursor: pointer;
    transition: filter 0.1s;
    flex-shrink: 0;
}
.tl-block:hover { filter: brightness(1.4); }
.tl-block.active {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
    filter: brightness(1.2);
}

/* Gap mode */
.gap-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 6px;
    border-radius: 3px;
    cursor: pointer;
    margin-bottom: 1px;
    border-left: 3px solid transparent;
    transition: background 0.1s;
}
.gap-item:hover { background: var(--bg-card); }
.gap-item.active {
    background: var(--bg-highlight);
    border-left-color: var(--accent);
}
.gap-item .gap-id {
    font-size: 11px;
    color: var(--text-muted);
    width: 24px;
    text-align: right;
    font-family: 'Consolas', monospace;
}
.gap-item .gap-summary {
    flex: 1;
    font-size: 11px;
    color: var(--text-secondary);
}
.gap-item .gap-counts {
    font-size: 10px;
    color: var(--text-muted);
    white-space: nowrap;
}
.gap-center {
    padding: 14px;
}
.gap-anchor-pair {
    display: flex;
    gap: 12px;
    margin-bottom: 14px;
}
.gap-anchor-card {
    flex: 1;
    background: var(--bg-card);
    border-radius: 4px;
    padding: 8px;
    border: 1px solid var(--border);
}
.gap-anchor-card .anchor-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.gap-unanchored {
    display: flex;
    gap: 12px;
}
.gap-col {
    flex: 1;
    min-width: 0;
}
.gap-col-header {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 4px 8px;
    border-radius: 3px;
    margin-bottom: 6px;
}
.gap-col-manga .gap-col-header { color: var(--manga-color); background: rgba(136,187,255,0.08); }
.gap-col-anime .gap-col-header { color: var(--anime-color); background: rgba(255,170,102,0.08); }
.gap-line {
    padding: 3px 8px;
    border-left: 2px solid var(--border);
    margin-bottom: 2px;
    font-size: 11px;
    color: var(--text-secondary);
}
.gap-line .speaker {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 1px;
}

/* Empty state */
.empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
    font-size: 13px;
    padding: 20px;
    text-align: center;
}

/* Responsive */
@media (max-width: 1400px) { .left-panel { display: none; } }
@media (max-width: 1100px) { .right-panel { width: 300px; } }
@media (max-width: 900px) {
    .main-content { flex-direction: column; }
    .right-panel { width: 100%; border-left: none; border-top: 1px solid var(--border); max-height: 300px; }
}
</style>
</head>
<body>
<div id="app">
    <!-- Top bar -->
    <div class="top-bar">
        <select id="episode-select"></select>
        <div class="title">Alignment <span>Viewer</span></div>
        <div class="stats" id="top-stats"></div>
        <div class="shortcuts">
            <kbd>J</kbd><kbd>K</kbd> nav
            <kbd>G</kbd> gaps
            <kbd>O</kbd> overview
            <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> tier
            <kbd>C</kbd> clear
        </div>
    </div>

    <!-- Main content -->
    <div class="main-content">
        <!-- Left panel -->
        <div class="left-panel" id="left-panel">
            <div class="panel-section" id="stats-section"></div>
            <div class="panel-section" id="filter-section"></div>
            <div class="record-list" id="record-list"></div>
        </div>

        <!-- Center panel -->
        <div class="center-panel" id="center-panel">
            <div class="empty-state" id="center-empty">Select an alignment record to view comparison</div>
        </div>

        <!-- Right panel -->
        <div class="right-panel" id="right-panel">
            <div class="empty-state" id="right-empty">No record selected</div>
        </div>
    </div>

    <!-- Timeline -->
    <div class="timeline-bar">
        <div class="tl-label" id="tl-label">Records</div>
        <div class="timeline-blocks" id="timeline-blocks"></div>
    </div>
</div>

<script>
// ========================================================================
// State
// ========================================================================
const state = {
    episodes: [],
    currentEpisode: '{{ episode_id }}',
    summary: null,       // enriched records
    anchors: null,       // pass1_anchors
    gaps: null,          // pass1_gaps
    mangaDialogue: {},   // seq -> dialogue entry
    animeDialogue: {},   // seq -> dialogue entry

    // Navigation
    filteredRecords: [],
    currentIdx: -1,

    // Filters
    tierFilter: null,      // null = all, 1/2/3
    typeFilter: null,      // null = all, 'anchor'/'interp'
    speakerMatchOnly: false,
    sortField: 'alignment_id',

    // Gap mode
    gapsMode: false,
    filteredGaps: [],
    currentGapIdx: -1,

    // UI
    overviewVisible: true,
};

// DOM refs
const $episodeSelect = document.getElementById('episode-select');
const $topStats = document.getElementById('top-stats');
const $statsSection = document.getElementById('stats-section');
const $filterSection = document.getElementById('filter-section');
const $recordList = document.getElementById('record-list');
const $centerPanel = document.getElementById('center-panel');
const $centerEmpty = document.getElementById('center-empty');
const $rightPanel = document.getElementById('right-panel');
const $rightEmpty = document.getElementById('right-empty');
const $leftPanel = document.getElementById('left-panel');
const $tlLabel = document.getElementById('tl-label');
const $tlBlocks = document.getElementById('timeline-blocks');

// ========================================================================
// Utilities
// ========================================================================
function esc(s) {
    if (!s) return '';
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function charColor(id) {
    if (!id) return 'var(--text-muted)';
    let h = 0;
    for (let i = 0; i < id.length; i++) h = id.charCodeAt(i) + ((h << 5) - h);
    return `hsl(${((h % 360) + 360) % 360}, 65%, 60%)`;
}

function tierColor(tier) {
    if (tier === 1) return 'var(--tier1)';
    if (tier === 2) return 'var(--tier2)';
    return 'var(--tier3)';
}

function fmt(sec) {
    if (sec == null) return '--';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
}

function confBarColor(conf) {
    if (conf >= 0.7) return 'var(--tier1)';
    if (conf >= 0.4) return 'var(--tier2)';
    return 'var(--tier3)';
}

function matchIcon(val) {
    if (val === true) return '<span class="match-icon match-yes">Y</span>';
    if (val === false) return '<span class="match-icon match-no">N</span>';
    return '<span class="match-icon" style="color:var(--text-muted)">--</span>';
}

// ========================================================================
// Init
// ========================================================================
async function init() {
    const res = await fetch('/api/episodes');
    const data = await res.json();
    state.episodes = data.episodes || [];

    $episodeSelect.innerHTML = state.episodes.map(e =>
        `<option value="${e.episode_id}" ${e.episode_id === state.currentEpisode ? 'selected' : ''}>` +
        `Ep ${e.episode_id} (${e.total_records} records)</option>`
    ).join('');

    $episodeSelect.addEventListener('change', () => {
        state.currentEpisode = $episodeSelect.value;
        loadEpisode(state.currentEpisode);
    });

    await loadEpisode(state.currentEpisode);
}

// ========================================================================
// Episode loading
// ========================================================================
async function loadEpisode(epId) {
    state.summary = null;
    state.anchors = null;
    state.gaps = null;
    state.gapsMode = false;
    state.currentIdx = -1;
    state.currentGapIdx = -1;

    const res = await fetch(`/api/alignment/${epId}/summary`);
    if (!res.ok) {
        $centerPanel.innerHTML = '<div class="empty-state">Failed to load alignment data</div>';
        return;
    }
    state.summary = await res.json();

    // Build dialogue lookup from enriched records (for gap mode)
    state.mangaDialogue = {};
    state.animeDialogue = {};

    applyFilters();
    renderStats();
    renderFilters();
    renderRecordList();
    renderTimeline();
    renderTopStats();

    // Select first record
    if (state.filteredRecords.length > 0) {
        selectRecord(0);
    } else {
        $centerPanel.innerHTML = '<div class="empty-state">No alignment records found</div>';
        $rightPanel.innerHTML = '<div class="empty-state">No record selected</div>';
    }
}

// ========================================================================
// Filtering & sorting
// ========================================================================
function applyFilters() {
    if (!state.summary) { state.filteredRecords = []; return; }

    let records = state.summary.records.slice();

    // Tier filter
    if (state.tierFilter !== null) {
        records = records.filter(r => r.confidence_tier === state.tierFilter);
    }
    // Type filter
    if (state.typeFilter === 'anchor') {
        records = records.filter(r => r.anchor_type === 'dialogue_anchored');
    } else if (state.typeFilter === 'interp') {
        records = records.filter(r => r.anchor_type !== 'dialogue_anchored');
    }
    // Speaker match
    if (state.speakerMatchOnly) {
        records = records.filter(r => r.speaker_match === true);
    }

    // Sort
    const field = state.sortField;
    records.sort((a, b) => {
        if (field === 'confidence') return (b.confidence || 0) - (a.confidence || 0);
        if (field === 'score_openai') return (b.score_openai || 0) - (a.score_openai || 0);
        if (field === 'time_range') {
            const ta = (a.time_range && a.time_range[0]) || 9999;
            const tb = (b.time_range && b.time_range[0]) || 9999;
            return ta - tb;
        }
        // Default: alignment_id
        return (a.alignment_id || '').localeCompare(b.alignment_id || '');
    });

    state.filteredRecords = records;
}

// ========================================================================
// Render: top stats
// ========================================================================
function renderTopStats() {
    if (!state.summary) { $topStats.textContent = ''; return; }
    const recs = state.summary.records;
    const anchored = recs.filter(r => r.anchor_type === 'dialogue_anchored').length;
    const interp = recs.length - anchored;
    const avgConf = recs.length > 0 ? (recs.reduce((s, r) => s + (r.confidence || 0), 0) / recs.length).toFixed(3) : '0';
    const arc = state.summary.arc_id || '';
    $topStats.innerHTML =
        `<span class="badge badge-anchor">${anchored} anchored</span> ` +
        `<span class="badge badge-interp">${interp} interpolated</span> ` +
        `avg conf: ${avgConf}` +
        (arc ? ` | <span class="badge" style="background:var(--bg-card)">${esc(arc)}</span>` : '') +
        (state.gapsMode ? ' | <span class="badge" style="background:var(--accent);color:#fff">GAP MODE</span>' : '');
}

// ========================================================================
// Render: stats panel
// ========================================================================
function renderStats() {
    if (!state.summary) { $statsSection.innerHTML = ''; return; }
    const recs = state.summary.records;
    const t1 = recs.filter(r => r.confidence_tier === 1).length;
    const t2 = recs.filter(r => r.confidence_tier === 2).length;
    const t3 = recs.filter(r => r.confidence_tier === 3).length;
    const avgConf = recs.length > 0 ? (recs.reduce((s, r) => s + (r.confidence || 0), 0) / recs.length).toFixed(3) : '0';
    const chapters = state.summary.chapter_ids.join(', ') || '--';

    $statsSection.innerHTML = `
        <div class="panel-section-title">Alignment Stats</div>
        <div class="stat-row"><span>Total records</span><span class="val">${recs.length}</span></div>
        <div class="stat-row"><span>Tier 1 (anchored)</span><span class="val" style="color:var(--tier1)">${t1}</span></div>
        <div class="stat-row"><span>Tier 2 (interpolated)</span><span class="val" style="color:var(--tier2)">${t2}</span></div>
        <div class="stat-row"><span>Tier 3 (low conf)</span><span class="val" style="color:var(--tier3)">${t3}</span></div>
        <div class="stat-row"><span>Avg confidence</span><span class="val">${avgConf}</span></div>
        <div class="stat-row"><span>Chapters</span><span class="val">${esc(chapters)}</span></div>
        <div class="stat-row"><span>Episode</span><span class="val">${esc(state.summary.episode_id)}</span></div>
    `;
}

// ========================================================================
// Render: filter controls
// ========================================================================
function renderFilters() {
    $filterSection.innerHTML = `
        <div class="panel-section-title">Filters</div>
        <div class="filter-row">
            <button class="filter-btn ${state.tierFilter===1?'active':''}" data-tier="1">Tier 1</button>
            <button class="filter-btn ${state.tierFilter===2?'active':''}" data-tier="2">Tier 2</button>
            <button class="filter-btn ${state.tierFilter===3?'active':''}" data-tier="3">Tier 3</button>
        </div>
        <div class="filter-row">
            <button class="filter-btn ${state.typeFilter==='anchor'?'active':''}" data-type="anchor">Anchored</button>
            <button class="filter-btn ${state.typeFilter==='interp'?'active':''}" data-type="interp">Interp</button>
            <button class="filter-btn ${state.speakerMatchOnly?'active':''}" data-speaker="1">Speaker Match</button>
        </div>
        <select class="sort-select" id="sort-select">
            <option value="alignment_id" ${state.sortField==='alignment_id'?'selected':''}>Sort: ID</option>
            <option value="confidence" ${state.sortField==='confidence'?'selected':''}>Sort: Confidence</option>
            <option value="score_openai" ${state.sortField==='score_openai'?'selected':''}>Sort: OpenAI Score</option>
            <option value="time_range" ${state.sortField==='time_range'?'selected':''}>Sort: Time</option>
        </select>
    `;

    // Bind events
    $filterSection.querySelectorAll('[data-tier]').forEach(btn => {
        btn.addEventListener('click', () => {
            const t = parseInt(btn.dataset.tier);
            state.tierFilter = state.tierFilter === t ? null : t;
            onFilterChange();
        });
    });
    $filterSection.querySelectorAll('[data-type]').forEach(btn => {
        btn.addEventListener('click', () => {
            const t = btn.dataset.type;
            state.typeFilter = state.typeFilter === t ? null : t;
            onFilterChange();
        });
    });
    $filterSection.querySelector('[data-speaker]')?.addEventListener('click', () => {
        state.speakerMatchOnly = !state.speakerMatchOnly;
        onFilterChange();
    });
    document.getElementById('sort-select')?.addEventListener('change', (e) => {
        state.sortField = e.target.value;
        onFilterChange();
    });
}

function onFilterChange() {
    applyFilters();
    renderFilters();
    renderRecordList();
    renderTimeline();
    if (state.filteredRecords.length > 0) {
        selectRecord(0);
    } else {
        state.currentIdx = -1;
        $centerPanel.innerHTML = '<div class="empty-state">No records match filters</div>';
        $rightPanel.innerHTML = '<div class="empty-state">No record selected</div>';
    }
}

// ========================================================================
// Render: record list
// ========================================================================
function renderRecordList() {
    if (state.gapsMode) { renderGapList(); return; }

    const recs = state.filteredRecords;
    $recordList.innerHTML = recs.map((r, i) => {
        const tier = r.confidence_tier || 3;
        const isAnchor = r.anchor_type === 'dialogue_anchored';
        const num = (r.alignment_id || '').replace('align_', '');
        const summary = r.manga_lines.length > 0 ? r.manga_lines[0].content.substring(0, 30) + '...' : (r.panel_ids.join(',') || '--');
        return `<div class="record-item ${i===state.currentIdx?'active':''}" data-idx="${i}">
            <div class="conf-dot" style="background:${tierColor(tier)}"></div>
            <div class="rec-id">${esc(num)}</div>
            <div class="rec-summary" title="${esc(r.alignment_id)}">${esc(summary)}</div>
            <span class="rec-badge ${isAnchor?'badge-anchor':'badge-interp'}"
                  style="background:${isAnchor?'#2a5a3a':'#5a4a2a'};color:${isAnchor?'#aea':'#eda'}">${isAnchor?'A':'I'}</span>
        </div>`;
    }).join('');

    $recordList.querySelectorAll('.record-item').forEach(el => {
        el.addEventListener('click', () => selectRecord(parseInt(el.dataset.idx)));
    });
}

function selectRecord(idx) {
    if (idx < 0 || idx >= state.filteredRecords.length) return;
    state.currentIdx = idx;
    const rec = state.filteredRecords[idx];

    // Update list highlight
    $recordList.querySelectorAll('.record-item').forEach((el, i) => {
        el.classList.toggle('active', i === idx);
    });
    // Scroll active into view
    const active = $recordList.querySelector('.record-item.active');
    if (active) active.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

    // Update timeline
    $tlBlocks.querySelectorAll('.tl-block').forEach((el, i) => {
        el.classList.toggle('active', i === idx);
    });

    renderComparison(rec);
    renderDetail(rec);
}

// ========================================================================
// Render: center panel - comparison
// ========================================================================
function renderComparison(rec) {
    const chId = rec.chapter_id || state.summary.chapter_ids[0] || '';
    const epId = rec.episode_id || state.summary.episode_id || '';

    // Manga panel image(s)
    let mangaImg = '';
    const panelImgs = rec.panel_images || [];
    if (panelImgs.length > 0) {
        const primary = panelImgs[0].replace(/\\/g, '/');
        const fname = primary.split('/').pop();
        mangaImg = `<img src="/panel/${chId}/${esc(fname)}" loading="lazy" alt="Manga panel">`;
        if (panelImgs.length > 1) {
            mangaImg += '<div class="multi-thumb-strip">' + panelImgs.map((p, i) => {
                const fn = p.replace(/\\/g, '/').split('/').pop();
                return `<img src="/panel/${chId}/${esc(fn)}" class="${i===0?'active':''}" loading="lazy" alt="Panel ${i+1}">`;
            }).join('') + '</div>';
        }
    } else {
        mangaImg = '<div class="no-image">No panel image</div>';
    }

    // Anime keyframe image(s)
    let animeImg = '';
    const kfFiles = rec.keyframe_files || [];
    if (kfFiles.length > 0) {
        animeImg = `<img src="/keyframe/${epId}/${esc(kfFiles[0])}" loading="lazy" alt="Anime keyframe">`;
        if (kfFiles.length > 1) {
            animeImg += '<div class="multi-thumb-strip">' + kfFiles.map((f, i) =>
                `<img src="/keyframe/${epId}/${esc(f)}" class="${i===0?'active':''}" loading="lazy" alt="Keyframe ${i+1}">`
            ).join('') + '</div>';
        }
    } else {
        animeImg = '<div class="no-image">No keyframe</div>';
    }

    // Dialogue comparison
    let dialogueHtml = '';
    if (rec.manga_lines.length > 0 || rec.anime_lines.length > 0) {
        dialogueHtml = `
        <div class="dialogue-section">
            <div class="dialogue-section-title">Dialogue Comparison</div>
            <div class="dialogue-comparison">
                <div class="dialogue-col dcol-manga">
                    <div class="dialogue-col-header">Manga (${rec.manga_lines.length} lines)</div>
                    ${rec.manga_lines.map(l => `
                        <div class="dialogue-line" style="border-left-color:${charColor(l.speaker_id)}">
                            <div class="speaker" style="color:${charColor(l.speaker_id)}">${esc(l.speaker || l.speaker_id)}</div>
                            <div class="content">${esc(l.content)}</div>
                            ${l.emotion ? `<div class="emotion">${esc(l.emotion)}</div>` : ''}
                        </div>
                    `).join('')}
                    ${rec.manga_lines.length === 0 ? '<div style="color:var(--text-muted);padding:4px 8px;font-size:11px">No manga dialogue</div>' : ''}
                </div>
                <div class="dialogue-col dcol-anime">
                    <div class="dialogue-col-header">Anime (${rec.anime_lines.length} lines)</div>
                    ${rec.anime_lines.map(l => `
                        <div class="dialogue-line" style="border-left-color:${charColor(l.speaker_id)}">
                            <div class="speaker" style="color:${charColor(l.speaker_id)}">${esc(l.speaker || l.speaker_id)}</div>
                            <div class="content">${esc(l.content)}</div>
                            ${l.emotion ? `<div class="emotion">${esc(l.emotion)}</div>` : ''}
                        </div>
                    `).join('')}
                    ${rec.anime_lines.length === 0 ? '<div style="color:var(--text-muted);padding:4px 8px;font-size:11px">No anime dialogue</div>' : ''}
                </div>
            </div>
        </div>`;
    }

    // Context strip
    let contextHtml = '';
    const ctx = rec.manga_context || {};
    const before = ctx.before || [];
    const after = ctx.after || [];
    if (before.length > 0 || after.length > 0) {
        contextHtml = `
        <div class="context-section">
            <div class="dialogue-section-title">Manga Context</div>
            <div class="context-strip">
                ${before.map(c => {
                    const fn = (c.panel_image_path || '').replace(/\\/g, '/').split('/').pop();
                    return fn ? `<img class="ctx-thumb" src="/panel/${chId}/${esc(fn)}" loading="lazy"
                        title="Page ${c.page || '?'} - ${esc(c.location || '')}">` : '';
                }).join('')}
                <div class="ctx-divider"></div>
                ${panelImgs.length > 0 ? (() => {
                    const fn = panelImgs[0].replace(/\\/g, '/').split('/').pop();
                    return `<img class="ctx-thumb current" src="/panel/${chId}/${esc(fn)}" loading="lazy" title="Current panel">`;
                })() : ''}
                <div class="ctx-divider"></div>
                ${after.map(c => {
                    const fn = (c.panel_image_path || '').replace(/\\/g, '/').split('/').pop();
                    return fn ? `<img class="ctx-thumb" src="/panel/${chId}/${esc(fn)}" loading="lazy"
                        title="Page ${c.page || '?'} - ${esc(c.location || '')}">` : '';
                }).join('')}
            </div>
        </div>`;
    }

    $centerPanel.innerHTML = `
        <div class="comparison-grid">
            <div class="compare-side side-manga">
                <div class="side-label">Manga - Ch ${esc(chId)} p${rec.page_range ? rec.page_range[0] : '?'}</div>
                ${mangaImg}
            </div>
            <div class="compare-side side-anime">
                <div class="side-label">Anime - Ep ${esc(epId)} ${rec.time_range ? fmt(rec.time_range[0])+'-'+fmt(rec.time_range[1]) : ''}</div>
                ${animeImg}
            </div>
        </div>
        ${dialogueHtml}
        ${contextHtml}
    `;
}

// ========================================================================
// Render: right panel - detail
// ========================================================================
function renderDetail(rec) {
    const conf = rec.confidence || 0;
    const tier = rec.confidence_tier || 3;
    const isAnchor = rec.anchor_type === 'dialogue_anchored';

    let transformHtml = '';
    if (rec.transformation_analysis || rec.imperative_delta_prompt || rec.transformation_delta) {
        transformHtml = `<div class="detail-section">
            <div class="detail-section-title">Transformation</div>`;
        if (rec.transformation_delta) {
            transformHtml += `
            <div class="collapsible-header" onclick="toggleCollapse(this)">
                <span class="arrow">&#9654;</span> Delta
            </div>
            <div class="collapsible-body"><pre>${esc(JSON.stringify(rec.transformation_delta, null, 2))}</pre></div>`;
        }
        if (rec.transformation_analysis) {
            transformHtml += `
            <div class="collapsible-header" onclick="toggleCollapse(this)">
                <span class="arrow">&#9654;</span> Analysis
            </div>
            <div class="collapsible-body"><pre>${esc(rec.transformation_analysis)}</pre></div>`;
        }
        if (rec.imperative_delta_prompt) {
            transformHtml += `
            <div class="collapsible-header" onclick="toggleCollapse(this)">
                <span class="arrow">&#9654;</span> Delta Prompt
            </div>
            <div class="collapsible-body"><pre>${esc(rec.imperative_delta_prompt)}</pre></div>`;
        }
        transformHtml += '</div>';
    }

    $rightPanel.innerHTML = `
        <div class="detail-header">
            <span class="align-id">${esc(rec.alignment_id)}</span>
            <span class="badge badge-tier${tier}">Tier ${tier}</span>
            <span class="badge ${isAnchor?'badge-anchor':'badge-interp'}">${isAnchor?'Anchored':'Interpolated'}</span>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Confidence</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <div class="conf-bar-outer" style="flex:1">
                    <div class="conf-bar-inner" style="width:${(conf*100).toFixed(1)}%;background:${confBarColor(conf)}"></div>
                </div>
                <span style="font-weight:700;font-size:14px">${conf.toFixed(3)}</span>
            </div>
            <div class="score-grid">
                <span class="label">OpenAI embedding</span>
                <span class="value">${rec.score_openai != null ? rec.score_openai.toFixed(4) : '--'}</span>
                <span class="label">Bag-of-words</span>
                <span class="value">${rec.score_bow != null ? rec.score_bow.toFixed(4) : '--'}</span>
            </div>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Match Signals</div>
            <div class="score-grid">
                <span class="label">Speaker match</span>
                <span class="value">${matchIcon(rec.speaker_match)}</span>
                <span class="label">Location match</span>
                <span class="value">${matchIcon(rec.location_match)}</span>
                <span class="label">Time context</span>
                <span class="value">${matchIcon(rec.time_context_match)}</span>
            </div>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Mapping</div>
            <div class="score-grid">
                <span class="label">Panel IDs</span>
                <span class="value" style="font-size:11px">${esc(rec.panel_ids.join(', ') || '--')}</span>
                <span class="label">Page range</span>
                <span class="value">${rec.page_range ? rec.page_range.join('-') : '--'}</span>
                <span class="label">Shot IDs</span>
                <span class="value" style="font-size:11px">${esc((rec.shot_ids||[]).join(', ') || '--')}</span>
                <span class="label">Scene(s)</span>
                <span class="value">${esc((rec.scene_numbers||[]).join(', ') || '--')}</span>
                <span class="label">Time range</span>
                <span class="value" style="font-family:'Consolas',monospace">${rec.time_range ? fmt(rec.time_range[0])+' - '+fmt(rec.time_range[1]) : '--'}</span>
            </div>
        </div>
        ${rec.alignment_explanation ? `
        <div class="detail-section">
            <div class="detail-section-title">Explanation</div>
            <div class="explanation-text">${esc(rec.alignment_explanation)}</div>
        </div>` : ''}
        ${rec.anime_original_segments && rec.anime_original_segments.length > 0 ? `
        <div class="detail-section">
            <div class="detail-section-title">Anime-Original Segments</div>
            ${rec.anime_original_segments.map(s => `<div style="font-size:11px;color:var(--text-secondary);padding:2px 0">${esc(JSON.stringify(s))}</div>`).join('')}
        </div>` : ''}
        ${transformHtml}
    `;
}

function toggleCollapse(header) {
    const arrow = header.querySelector('.arrow');
    const body = header.nextElementSibling;
    if (arrow) arrow.classList.toggle('open');
    if (body) body.classList.toggle('open');
}

// ========================================================================
// Render: timeline
// ========================================================================
function renderTimeline() {
    if (state.gapsMode) { renderGapTimeline(); return; }

    const recs = state.filteredRecords;
    if (recs.length === 0) { $tlBlocks.innerHTML = ''; return; }

    // Compute widths proportional to time range, with minimum
    const durations = recs.map(r => {
        if (r.time_range && r.time_range[1] > r.time_range[0])
            return r.time_range[1] - r.time_range[0];
        return 2; // min 2s equivalent
    });
    const maxDur = Math.max(...durations, 1);

    $tlBlocks.innerHTML = recs.map((r, i) => {
        const w = Math.max(3, Math.round((durations[i] / maxDur) * 30));
        const tier = r.confidence_tier || 3;
        return `<div class="tl-block ${i===state.currentIdx?'active':''}"
            data-idx="${i}"
            style="width:${w}px;background:${tierColor(tier)}"
            title="${esc(r.alignment_id)} (conf: ${(r.confidence||0).toFixed(2)})"></div>`;
    }).join('');

    $tlBlocks.querySelectorAll('.tl-block').forEach(el => {
        el.addEventListener('click', () => selectRecord(parseInt(el.dataset.idx)));
    });

    $tlLabel.textContent = `Records (${recs.length})`;
}

// ========================================================================
// Gap mode
// ========================================================================
async function toggleGapsMode() {
    state.gapsMode = !state.gapsMode;
    renderTopStats();

    if (state.gapsMode) {
        // Load anchors and gaps
        if (!state.anchors) {
            const res = await fetch(`/api/alignment/${state.currentEpisode}/anchors`);
            if (res.ok) state.anchors = await res.json();
        }
        if (!state.gaps) {
            const res = await fetch(`/api/alignment/${state.currentEpisode}/gaps`);
            if (res.ok) state.gaps = await res.json();
        }
        // Also load dialogue indices for gap content
        if (!state.summary) return;
        await loadDialogueIndices();

        state.filteredGaps = state.gaps || [];
        state.currentGapIdx = -1;
        renderGapList();
        renderGapTimeline();
        if (state.filteredGaps.length > 0) {
            selectGap(0);
        } else {
            $centerPanel.innerHTML = '<div class="empty-state">No gaps found</div>';
            $rightPanel.innerHTML = '<div class="empty-state">No gap selected</div>';
        }
    } else {
        state.currentGapIdx = -1;
        renderRecordList();
        renderTimeline();
        if (state.filteredRecords.length > 0) {
            selectRecord(state.currentIdx >= 0 ? state.currentIdx : 0);
        }
    }
}

async function loadDialogueIndices() {
    // Fetch manga dialogue from summary records
    if (Object.keys(state.mangaDialogue).length === 0 && state.summary) {
        for (const rec of state.summary.records) {
            for (const l of (rec.manga_lines || [])) {
                state.mangaDialogue[l.seq] = l;
            }
        }
    }
    if (Object.keys(state.animeDialogue).length === 0 && state.summary) {
        for (const rec of state.summary.records) {
            for (const l of (rec.anime_lines || [])) {
                state.animeDialogue[l.seq] = l;
            }
        }
    }
}

function renderGapList() {
    const gaps = state.filteredGaps;
    $recordList.innerHTML = gaps.map((g, i) => {
        const mCount = (g.manga_seqs_unanchored || []).length;
        const aCount = (g.anime_seqs_unanchored || []).length;
        return `<div class="gap-item ${i===state.currentGapIdx?'active':''}" data-idx="${i}">
            <div class="gap-id">${g.gap_id}</div>
            <div class="gap-summary">
                ${g.start_anchor_id ? esc(g.start_anchor_id) : 'start'} ... ${g.end_anchor_id ? esc(g.end_anchor_id) : 'end'}
            </div>
            <div class="gap-counts">M:${mCount} A:${aCount}</div>
        </div>`;
    }).join('');

    $recordList.querySelectorAll('.gap-item').forEach(el => {
        el.addEventListener('click', () => selectGap(parseInt(el.dataset.idx)));
    });
}

function selectGap(idx) {
    if (idx < 0 || idx >= state.filteredGaps.length) return;
    state.currentGapIdx = idx;
    const gap = state.filteredGaps[idx];

    $recordList.querySelectorAll('.gap-item').forEach((el, i) => {
        el.classList.toggle('active', i === idx);
    });

    renderGapCenter(gap);
    renderGapDetail(gap);

    // Timeline highlight
    $tlBlocks.querySelectorAll('.tl-block').forEach((el, i) => {
        el.classList.toggle('active', i === idx);
    });
}

function renderGapCenter(gap) {
    // Find bounding anchors
    const anchors = state.anchors || [];
    const startAnchor = anchors.find(a => a.anchor_id === gap.start_anchor_id);
    const endAnchor = anchors.find(a => a.anchor_id === gap.end_anchor_id);

    let anchorPairHtml = '<div class="gap-anchor-pair">';
    if (startAnchor) {
        anchorPairHtml += `<div class="gap-anchor-card">
            <div class="anchor-label">Start Anchor: ${esc(startAnchor.anchor_id)}</div>
            <div style="font-size:11px;color:var(--manga-color);margin-bottom:2px">M: "${esc(startAnchor.manga_content)}"</div>
            <div style="font-size:11px;color:var(--anime-color)">A: "${esc(startAnchor.anime_content)}"</div>
            <div style="font-size:10px;color:var(--text-muted);margin-top:2px">${esc(startAnchor.manga_speaker_id||'')} | score: ${(startAnchor.score_openai||0).toFixed(3)}</div>
        </div>`;
    } else {
        anchorPairHtml += '<div class="gap-anchor-card"><div class="anchor-label">Start: Beginning of episode</div></div>';
    }
    if (endAnchor) {
        anchorPairHtml += `<div class="gap-anchor-card">
            <div class="anchor-label">End Anchor: ${esc(endAnchor.anchor_id)}</div>
            <div style="font-size:11px;color:var(--manga-color);margin-bottom:2px">M: "${esc(endAnchor.manga_content)}"</div>
            <div style="font-size:11px;color:var(--anime-color)">A: "${esc(endAnchor.anime_content)}"</div>
            <div style="font-size:10px;color:var(--text-muted);margin-top:2px">${esc(endAnchor.manga_speaker_id||'')} | score: ${(endAnchor.score_openai||0).toFixed(3)}</div>
        </div>`;
    } else {
        anchorPairHtml += '<div class="gap-anchor-card"><div class="anchor-label">End: End of episode</div></div>';
    }
    anchorPairHtml += '</div>';

    // Unanchored content
    const mangaSeqs = gap.manga_seqs_unanchored || [];
    const animeSeqs = gap.anime_seqs_unanchored || [];

    let mangaLinesHtml = mangaSeqs.map(seq => {
        const d = state.mangaDialogue[seq];
        if (d) return `<div class="gap-line"><div class="speaker" style="color:${charColor(d.speaker_id)}">${esc(d.speaker||d.speaker_id)} [${seq}]</div><div>${esc(d.content)}</div></div>`;
        return `<div class="gap-line" style="color:var(--text-muted)">Seq ${seq} (not loaded)</div>`;
    }).join('');

    let animeLinesHtml = animeSeqs.map(seq => {
        const d = state.animeDialogue[seq];
        if (d) return `<div class="gap-line"><div class="speaker" style="color:${charColor(d.speaker_id)}">${esc(d.speaker||d.speaker_id)} [${seq}]</div><div>${esc(d.content)}</div></div>`;
        return `<div class="gap-line" style="color:var(--text-muted)">Seq ${seq} (not loaded)</div>`;
    }).join('');

    $centerPanel.innerHTML = `
        <div class="gap-center">
            ${anchorPairHtml}
            <div class="dialogue-section-title">Unanchored Content</div>
            <div class="gap-unanchored">
                <div class="gap-col gap-col-manga">
                    <div class="gap-col-header">Manga (${mangaSeqs.length} lines)</div>
                    ${mangaLinesHtml || '<div style="color:var(--text-muted);padding:4px 8px;font-size:11px">No unanchored manga lines</div>'}
                </div>
                <div class="gap-col gap-col-anime">
                    <div class="gap-col-header">Anime (${animeSeqs.length} lines)</div>
                    ${animeLinesHtml || '<div style="color:var(--text-muted);padding:4px 8px;font-size:11px">No unanchored anime lines</div>'}
                </div>
            </div>
        </div>
    `;
}

function renderGapDetail(gap) {
    const timeRange = gap.anime_time_range || [0, 0];
    const mPanels = gap.manga_panel_ids || [];
    const aShots = gap.anime_shot_ids || [];

    $rightPanel.innerHTML = `
        <div class="detail-header">
            <span class="align-id">Gap ${gap.gap_id}</span>
            <span class="badge" style="background:var(--bg-card)">${gap.start_anchor_id || 'start'} .. ${gap.end_anchor_id || 'end'}</span>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Gap Stats</div>
            <div class="score-grid">
                <span class="label">Manga lines</span>
                <span class="value">${(gap.manga_seqs_unanchored||[]).length}</span>
                <span class="label">Anime lines</span>
                <span class="value">${(gap.anime_seqs_unanchored||[]).length}</span>
                <span class="label">Manga panels</span>
                <span class="value">${mPanels.length}</span>
                <span class="label">Anime shots</span>
                <span class="value">${aShots.length}</span>
                <span class="label">Time range</span>
                <span class="value" style="font-family:'Consolas',monospace">${fmt(timeRange[0])} - ${fmt(timeRange[1])}</span>
                <span class="label">Duration</span>
                <span class="value" style="font-family:'Consolas',monospace">${fmt(timeRange[1] - timeRange[0])}</span>
            </div>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Manga Panels</div>
            <div style="font-size:11px;color:var(--text-secondary);word-break:break-all">${esc(mPanels.join(', ') || 'None')}</div>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Anime Shots</div>
            <div style="font-size:11px;color:var(--text-secondary);word-break:break-all">${esc(aShots.join(', ') || 'None')}</div>
        </div>
    `;
}

function renderGapTimeline() {
    const gaps = state.filteredGaps;
    if (gaps.length === 0) { $tlBlocks.innerHTML = ''; return; }

    $tlBlocks.innerHTML = gaps.map((g, i) => {
        const mCount = (g.manga_seqs_unanchored || []).length;
        const aCount = (g.anime_seqs_unanchored || []).length;
        const w = Math.max(4, Math.round(Math.max(mCount, aCount) * 2));
        const color = mCount === 0 ? 'var(--anime-color)' : aCount === 0 ? 'var(--manga-color)' : 'var(--tier2)';
        return `<div class="tl-block ${i===state.currentGapIdx?'active':''}"
            data-idx="${i}" style="width:${w}px;background:${color};opacity:0.6"
            title="Gap ${g.gap_id}: ${mCount}M / ${aCount}A"></div>`;
    }).join('');

    $tlBlocks.querySelectorAll('.tl-block').forEach(el => {
        el.addEventListener('click', () => selectGap(parseInt(el.dataset.idx)));
    });

    $tlLabel.textContent = `Gaps (${gaps.length})`;
}

// ========================================================================
// Keyboard shortcuts
// ========================================================================
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;

    switch(e.key) {
        case 'j':
        case 'J':
        case 'ArrowDown':
            e.preventDefault();
            if (state.gapsMode) {
                if (state.currentGapIdx < state.filteredGaps.length - 1) selectGap(state.currentGapIdx + 1);
            } else {
                if (state.currentIdx < state.filteredRecords.length - 1) selectRecord(state.currentIdx + 1);
            }
            break;
        case 'k':
        case 'K':
        case 'ArrowUp':
            e.preventDefault();
            if (state.gapsMode) {
                if (state.currentGapIdx > 0) selectGap(state.currentGapIdx - 1);
            } else {
                if (state.currentIdx > 0) selectRecord(state.currentIdx - 1);
            }
            break;
        case 'g':
        case 'G':
            e.preventDefault();
            toggleGapsMode();
            break;
        case 'o':
        case 'O':
            e.preventDefault();
            state.overviewVisible = !state.overviewVisible;
            $leftPanel.classList.toggle('hidden', !state.overviewVisible);
            break;
        case '1':
            e.preventDefault();
            state.tierFilter = state.tierFilter === 1 ? null : 1;
            onFilterChange();
            break;
        case '2':
            e.preventDefault();
            state.tierFilter = state.tierFilter === 2 ? null : 2;
            onFilterChange();
            break;
        case '3':
            e.preventDefault();
            state.tierFilter = state.tierFilter === 3 ? null : 3;
            onFilterChange();
            break;
        case 'c':
        case 'C':
            e.preventDefault();
            state.tierFilter = null;
            state.typeFilter = null;
            state.speakerMatchOnly = false;
            state.sortField = 'alignment_id';
            onFilterChange();
            break;
        case 'Home':
            e.preventDefault();
            if (state.gapsMode) {
                if (state.filteredGaps.length > 0) selectGap(0);
            } else {
                if (state.filteredRecords.length > 0) selectRecord(0);
            }
            break;
        case 'End':
            e.preventDefault();
            if (state.gapsMode) {
                if (state.filteredGaps.length > 0) selectGap(state.filteredGaps.length - 1);
            } else {
                if (state.filteredRecords.length > 0) selectRecord(state.filteredRecords.length - 1);
            }
            break;
    }
});

// ========================================================================
// Start
// ========================================================================
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignment Viewer")
    parser.add_argument("--episode", default="0001", help="Episode ID to load (default: 0001)")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    startup_episode = args.episode
    PORT = args.port

    print(f"Alignment Viewer starting on http://localhost:{PORT}")
    print(f"Default episode: {startup_episode}")

    if not args.no_browser:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    app.run(host="0.0.0.0", port=PORT, debug=False)
