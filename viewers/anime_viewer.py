"""
Anime Metadata Viewer -- Interactive episode viewer with extraction artifact overlays.

Play an episode and see all metadata produced by anime_metadata.py in context.
Flask + embedded HTML5/CSS/JS (same pattern as reader.py, voice_tagger.py).

Layout:
  Top bar: episode selector, title, arc badge, stats, keyboard shortcut hints.
  Left panel (260px, toggleable): episode stats, character registry with colored
    badges, clickable scene list (scene number, type badge, time range).
  Center: HTML5 video player (range-request seeking) + keyframes/crops strip
    below showing all keyframes, background plate, and character crops for the
    current shot with confidence badges.
  Right panel (360px, scrollable): scene details (narrative, location, time
    context, emotional tone, characters with actions/emotions, time-synced
    dialogue list, shot type sequence bar, collapsible reconstruction prompt)
    + shot details (type, content origin, motion intensity, fade type, audio
    classification, color palette swatches, dialogue IDs).
  Bottom: dual-row timeline -- scene row (colored blocks by scene_type:
    story/flashback/opening/ending/eyecatch/transition) + shot row (zoomed to
    current scene, individual shots colored by shot_type) + synced red playhead.

Keyboard: Space play/pause, arrows seek, [ ] scene nav, , . shot nav, O overview.

Data: loads episode_manifest.json + shot_index.json + character_registry.json
on startup; lazy-loads scene_NNN.json on demand. Remuxes MKV->MP4 via ffmpeg
stream copy (one-time, cached). Serves keyframes/crops/plates from
output/anime_metadata/{episode_id}/keyframes/.

Run: python anime_viewer.py [--episode 0001] [--port 7862] [--no-browser]
Opens at http://localhost:7862
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import re
import subprocess
import webbrowser
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, render_template_string

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "anime_metadata"
ANIME_DIR = BASE_DIR / "anime"
PORT = 7862

app = Flask(__name__)

# Cache: episode_id -> {manifest, shots, characters, scenes: {num: data}}
episode_cache = {}
startup_episode = "0001"

# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

def discover_episodes():
    """Scan output/anime_metadata/ for episode dirs and match MKV files."""
    episodes = []
    if not OUTPUT_DIR.exists():
        return episodes

    for d in sorted(OUTPUT_DIR.iterdir()):
        if not d.is_dir() or not re.match(r"^\d{4}$", d.name):
            continue
        ep_id = d.name
        manifest_path = d / "episode_manifest.json"
        if not manifest_path.exists():
            continue

        # Find matching MKV
        mkv_path = find_mkv(ep_id)

        # Quick manifest peek for title/stats
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            episodes.append({
                "episode_id": ep_id,
                "title": m.get("episode_title", f"Episode {ep_id}"),
                "arc_id": m.get("arc_id", ""),
                "total_scenes": m.get("total_scenes", 0),
                "total_shots": m.get("total_shots", 0),
                "duration": m.get("total_duration_seconds", 0),
                "has_video": mkv_path is not None,
            })
        except Exception:
            continue
    return episodes


def find_mkv(episode_id):
    """Find the MKV file for an episode by scanning anime/ directory."""
    if not ANIME_DIR.exists():
        return None
    pattern = f"- {episode_id} -"
    for f in ANIME_DIR.iterdir():
        if f.suffix.lower() == ".mkv" and pattern in f.name:
            return f
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_cache(episode_id):
    if episode_id not in episode_cache:
        episode_cache[episode_id] = {
            "manifest": None,
            "shots": None,
            "characters": None,
            "scenes": {},
        }
    return episode_cache[episode_id]


def load_manifest(episode_id):
    cache = get_cache(episode_id)
    if cache["manifest"] is None:
        path = OUTPUT_DIR / episode_id / "episode_manifest.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cache["manifest"] = json.load(f)
    return cache["manifest"]


def load_shots(episode_id):
    cache = get_cache(episode_id)
    if cache["shots"] is None:
        path = OUTPUT_DIR / episode_id / "shot_index.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cache["shots"] = json.load(f)
    return cache["shots"]


def load_characters(episode_id):
    cache = get_cache(episode_id)
    if cache["characters"] is None:
        path = OUTPUT_DIR / episode_id / "character_registry.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cache["characters"] = json.load(f)
    return cache["characters"]


def load_scene(episode_id, scene_num):
    cache = get_cache(episode_id)
    if scene_num not in cache["scenes"]:
        path = OUTPUT_DIR / episode_id / f"scene_{scene_num:03d}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cache["scenes"][scene_num] = json.load(f)
    return cache["scenes"].get(scene_num)


# ---------------------------------------------------------------------------
# Video remux (MKV -> MP4)
# ---------------------------------------------------------------------------

def ensure_episode_mp4(episode_id):
    """Remux MKV to browser-friendly MP4 (stream copy, one-time)."""
    mkv_path = find_mkv(episode_id)
    if not mkv_path or not mkv_path.exists():
        return None

    mp4_dir = OUTPUT_DIR / episode_id
    mp4_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = mp4_dir / "episode.mp4"
    if mp4_path.exists():
        return mp4_path

    print(f"Remuxing {mkv_path.name} -> MP4 ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mkv_path),
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-c", "copy", "-sn",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Remux failed: {result.stderr[:300]}")
        return None
    print("Remux complete.")
    return mp4_path


# ---------------------------------------------------------------------------
# Routes: Video serving
# ---------------------------------------------------------------------------

@app.route("/video/<episode_id>")
def serve_video(episode_id):
    """Serve episode MP4 with range request support for seeking."""
    mp4_path = ensure_episode_mp4(episode_id)
    if not mp4_path or not mp4_path.exists():
        return "Video not found", 404

    file_size = mp4_path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        byte_start = 0
        byte_end = file_size - 1
        match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            byte_start = int(match.group(1))
            if match.group(2):
                byte_end = int(match.group(2))

        content_length = byte_end - byte_start + 1

        def generate():
            with open(mp4_path, "rb") as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return Response(
            generate(),
            status=206,
            mimetype="video/mp4",
            headers={
                "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": content_length,
            },
        )

    return send_file(str(mp4_path), mimetype="video/mp4")


# ---------------------------------------------------------------------------
# Routes: API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/episodes")
def api_episodes():
    eps = discover_episodes()
    return jsonify({"episodes": eps, "current": startup_episode})


@app.route("/api/episode/<episode_id>/manifest")
def api_manifest(episode_id):
    m = load_manifest(episode_id)
    if not m:
        return jsonify({"error": "Not found"}), 404
    return jsonify(m)


@app.route("/api/episode/<episode_id>/shots")
def api_shots(episode_id):
    s = load_shots(episode_id)
    if not s:
        return jsonify({"error": "Not found"}), 404
    return jsonify(s)


@app.route("/api/episode/<episode_id>/scene/<int:scene_num>")
def api_scene(episode_id, scene_num):
    s = load_scene(episode_id, scene_num)
    if not s:
        return jsonify({"error": "Not found"}), 404
    return jsonify(s)


@app.route("/api/episode/<episode_id>/characters")
def api_characters(episode_id):
    c = load_characters(episode_id)
    if not c:
        return jsonify({"error": "Not found"}), 404
    return jsonify(c)


# ---------------------------------------------------------------------------
# Routes: Keyframe / image serving
# ---------------------------------------------------------------------------

@app.route("/keyframe/<episode_id>/<path:filename>")
def serve_keyframe(episode_id, filename):
    """Serve keyframe, background plate, or character crop images."""
    base = OUTPUT_DIR / episode_id / "keyframes"
    filepath = base / filename
    if filepath.exists():
        return send_file(str(filepath), mimetype="image/jpeg")
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
<title>Anime Metadata Viewer</title>
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
    --narration-color: #88bbff;
    --sfx-color: #ffd700;
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

/* Overview panel (left) */
.overview-panel {
    width: 260px;
    overflow-y: auto;
    flex-shrink: 0;
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    font-size: 12px;
}
.overview-panel.hidden { display: none; }
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
.char-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 6px;
    border-radius: 3px;
    cursor: pointer;
    margin-bottom: 2px;
}
.char-item:hover { background: var(--bg-card); }
.char-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.char-item .name { flex: 1; }
.char-item .count { color: var(--text-muted); font-size: 11px; }

/* Scene list in overview */
.scene-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 6px;
    border-radius: 3px;
    cursor: pointer;
    margin-bottom: 1px;
    border-left: 3px solid transparent;
}
.scene-item:hover { background: var(--bg-card); }
.scene-item.active {
    background: var(--bg-highlight);
    border-left-color: var(--accent);
}
.scene-item .num {
    font-weight: 700;
    font-size: 11px;
    color: var(--text-muted);
    width: 20px;
    text-align: right;
}
.scene-item .type-badge {
    font-size: 9px;
    padding: 0 4px;
    border-radius: 2px;
    font-weight: 600;
    text-transform: uppercase;
}
.scene-item .time {
    margin-left: auto;
    font-size: 10px;
    color: var(--text-muted);
    font-family: 'Consolas', monospace;
}

/* Center column */
.center-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: #000;
}
.video-wrapper {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    position: relative;
}
.video-wrapper video {
    max-width: 100%;
    max-height: 100%;
    width: 100%;
}

/* Keyframes strip */
.keyframes-strip {
    height: 100px;
    background: var(--bg-panel);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    overflow-x: auto;
    flex-shrink: 0;
}
.kf-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    flex-shrink: 0;
}
.kf-group-label {
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kf-row {
    display: flex;
    gap: 4px;
}
.kf-thumb {
    width: 110px;
    height: 62px;
    object-fit: cover;
    border-radius: 3px;
    border: 1px solid var(--border);
}
.kf-thumb.active-kf {
    border-color: var(--accent);
    box-shadow: 0 0 6px var(--accent-dim);
}
.crop-thumb {
    width: 48px;
    height: 64px;
    object-fit: cover;
    border-radius: 3px;
    border: 1px solid var(--border);
    position: relative;
}
.crop-container {
    position: relative;
    flex-shrink: 0;
}
.crop-conf {
    position: absolute;
    bottom: 1px;
    right: 1px;
    font-size: 8px;
    background: rgba(0,0,0,0.7);
    color: var(--text-secondary);
    padding: 0 3px;
    border-radius: 2px;
}
.kf-divider {
    width: 1px;
    height: 60px;
    background: var(--border);
    flex-shrink: 0;
    margin: 0 4px;
}

/* Scene detail panel (right) */
.detail-panel {
    width: 360px;
    overflow-y: auto;
    flex-shrink: 0;
    background: var(--bg-panel);
    border-left: 1px solid var(--border);
    font-size: 12px;
}
.detail-section {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
}
.detail-section-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.scene-header {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-bottom: 6px;
}
.scene-header .scene-num {
    font-size: 18px;
    font-weight: 700;
    color: var(--accent);
}
.scene-header .scene-label {
    font-size: 12px;
    color: var(--text-secondary);
}
.meta-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 2px 10px;
    font-size: 11px;
}
.meta-grid .label { color: var(--text-muted); }
.meta-grid .value { color: var(--text-secondary); }
.narrative-text {
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-primary);
}

/* Characters in scene */
.char-card {
    padding: 4px 0;
    border-bottom: 1px solid rgba(42, 42, 64, 0.5);
}
.char-card:last-child { border-bottom: none; }
.char-card .char-name {
    font-weight: 600;
    font-size: 12px;
}
.char-card .char-action {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 1px;
}
.char-card .char-emotion {
    font-size: 10px;
    display: inline-block;
    padding: 0 5px;
    border-radius: 2px;
    margin-top: 2px;
}

/* Dialogue list */
.dialogue-item {
    padding: 5px 6px;
    border-radius: 3px;
    margin-bottom: 3px;
    border-left: 3px solid transparent;
    transition: background 0.15s;
}
.dialogue-item.active {
    background: var(--bg-highlight);
    border-left-color: var(--accent);
}
.dialogue-item .speaker {
    font-size: 11px;
    font-weight: 600;
}
.dialogue-item .content {
    font-size: 12px;
    margin-top: 1px;
}
.dialogue-item .content.narration {
    font-style: italic;
    color: var(--narration-color);
}
.dialogue-item .meta {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
}

/* Shot type sequence */
.shot-type-bar {
    display: flex;
    gap: 1px;
    align-items: flex-end;
    height: 22px;
}
.shot-type-block {
    width: 7px;
    border-radius: 1px;
    cursor: default;
}

/* Shot details */
.color-palette {
    display: flex;
    gap: 4px;
}
.color-swatch {
    width: 22px;
    height: 22px;
    border-radius: 3px;
    border: 1px solid var(--border);
    cursor: default;
}
.intensity-bar {
    width: 100%;
    height: 6px;
    background: var(--bg-dark);
    border-radius: 3px;
    overflow: hidden;
}
.intensity-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--accent);
}

/* Reconstruction prompt */
.recon-prompt {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.5;
    background: var(--bg-dark);
    padding: 8px;
    border-radius: 4px;
    white-space: pre-wrap;
    word-break: break-word;
}
.collapsible-header {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    user-select: none;
}
.collapsible-header .arrow {
    font-size: 10px;
    transition: transform 0.15s;
    color: var(--text-muted);
}
.collapsible-header .arrow.open { transform: rotate(90deg); }
.collapsible-body { display: none; margin-top: 6px; }
.collapsible-body.open { display: block; }

/* Timeline bar */
.timeline-bar {
    flex-shrink: 0;
    background: var(--bg-panel);
    border-top: 1px solid var(--border);
    padding: 6px 10px;
    position: relative;
    user-select: none;
}
.timeline-label {
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
}
.timeline-row {
    position: relative;
    height: 32px;
    margin-bottom: 4px;
    border-radius: 3px;
    overflow: hidden;
    background: var(--bg-dark);
    cursor: pointer;
}
.timeline-block {
    position: absolute;
    top: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
    overflow: hidden;
    border-right: 1px solid rgba(0,0,0,0.3);
    transition: filter 0.1s;
}
.timeline-block:hover { filter: brightness(1.3); }
.timeline-block.current-block {
    box-shadow: inset 0 0 0 2px var(--accent);
    z-index: 2;
}
.playhead {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--accent);
    z-index: 5;
    pointer-events: none;
}
.playhead::after {
    content: '';
    position: absolute;
    top: -3px;
    left: -3px;
    width: 8px;
    height: 8px;
    background: var(--accent);
    border-radius: 50%;
}

/* Scene type colors */
.stype-story { background: #2a4a7a; }
.stype-flashback { background: #7a5a2a; }
.stype-opening { background: #5a2a7a; }
.stype-ending { background: #5a2a7a; }
.stype-eyecatch { background: #2a7a5a; }
.stype-transition { background: #3a3a4a; }

/* Shot type colors */
.shottype-wide { background: #2a5a8a; }
.shottype-medium { background: #4a6a5a; }
.shottype-close-up { background: #8a4a3a; }
.shottype-reaction { background: #7a3a6a; }
.shottype-aerial { background: #3a6a7a; }
.shottype-over-the-shoulder { background: #5a5a3a; }
.shottype-pov { background: #4a3a6a; }

/* Content origin colors */
.origin-manga_derived { background: #2a5a2a; color: #88ff88; }
.origin-anime_original { background: #5a2a2a; color: #ff8888; }
.origin-establishing { background: #2a4a5a; color: #88ccff; }
.origin-transition { background: #3a3a4a; color: #aaa; }
.origin-reaction_shot { background: #4a2a5a; color: #cc88ff; }
.origin-action_extension { background: #5a4a2a; color: #ffcc88; }

/* Emotion colors */
.emo-calm { color: #88bbff; }
.emo-dramatic { color: #ff8888; }
.emo-tense { color: #ffaa44; }
.emo-comedic { color: #88ff88; }
.emo-action { color: #ff4444; }
.emo-triumphant { color: #ffd700; }
.emo-melancholic { color: #aa88ff; }

/* Scrollbar */
.overview-panel::-webkit-scrollbar,
.detail-panel::-webkit-scrollbar,
.keyframes-strip::-webkit-scrollbar { width: 5px; height: 5px; }
.overview-panel::-webkit-scrollbar-track,
.detail-panel::-webkit-scrollbar-track,
.keyframes-strip::-webkit-scrollbar-track { background: transparent; }
.overview-panel::-webkit-scrollbar-thumb,
.detail-panel::-webkit-scrollbar-thumb,
.keyframes-strip::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

@media (max-width: 1400px) {
    .overview-panel { display: none; }
}
@media (max-width: 1100px) {
    .detail-panel { width: 300px; }
}
</style>
</head>
<body>
<div id="app">
    <!-- Top bar -->
    <div class="top-bar">
        <select id="episode-select" onchange="switchEpisode(this.value)"></select>
        <div class="title"><span>Anime</span> Metadata Viewer</div>
        <div id="ep-stats" class="stats"></div>
        <div class="shortcuts">
            <kbd>Space</kbd> play
            <kbd>[</kbd><kbd>]</kbd> scene
            <kbd>,</kbd><kbd>.</kbd> shot
            <kbd>O</kbd> overview
        </div>
    </div>

    <!-- Main content -->
    <div class="main-content">
        <!-- Overview panel (left) -->
        <div class="overview-panel" id="overview-panel">
            <div class="panel-section" id="ep-overview"></div>
            <div class="panel-section">
                <div class="panel-section-title">Characters</div>
                <div id="char-list"></div>
            </div>
            <div class="panel-section">
                <div class="panel-section-title">Scenes</div>
                <div id="scene-list"></div>
            </div>
        </div>

        <!-- Center: video + keyframes -->
        <div class="center-panel">
            <div class="video-wrapper">
                <video id="video" controls></video>
            </div>
            <div class="keyframes-strip" id="keyframes-strip"></div>
        </div>

        <!-- Detail panel (right) -->
        <div class="detail-panel" id="detail-panel">
            <div id="scene-detail"></div>
            <div id="shot-detail"></div>
        </div>
    </div>

    <!-- Timeline -->
    <div class="timeline-bar">
        <div class="timeline-label">Scenes</div>
        <div class="timeline-row" id="scene-timeline"></div>
        <div class="timeline-label">Shots <span id="shot-timeline-label" style="color:var(--text-secondary)"></span></div>
        <div class="timeline-row" id="shot-timeline"></div>
    </div>
</div>

<script>
// ========================================================================
// State
// ========================================================================
const state = {
    episodes: [],
    currentEpisode: null,
    manifest: null,
    shots: [],
    shotsByScene: {},
    shotLookup: {},
    characters: null,
    sceneCache: {},
    currentSceneNum: null,
    currentShotId: null,
    overviewVisible: true,
};

const SCENE_TYPE_COLORS = {
    story: '#2a4a7a', flashback: '#7a5a2a', opening: '#5a2a7a',
    ending: '#5a2a7a', eyecatch: '#2a7a5a', transition: '#3a3a4a',
};
const SHOT_TYPE_COLORS = {
    'wide': '#2a5a8a', 'medium': '#4a6a5a', 'close-up': '#8a4a3a',
    'reaction': '#7a3a6a', 'aerial': '#3a6a7a', 'over-the-shoulder': '#5a5a3a',
    'pov': '#4a3a6a',
};
const EMOTION_CLASSES = {
    calm: 'emo-calm', dramatic: 'emo-dramatic', tense: 'emo-tense',
    comedic: 'emo-comedic', action: 'emo-action', triumphant: 'emo-triumphant',
    melancholic: 'emo-melancholic',
};

// ========================================================================
// Utility
// ========================================================================
function esc(s) {
    if (!s) return '';
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fmt(sec) {
    if (sec == null) return '--:--';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
}

function charColor(id) {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = id.charCodeAt(i) + ((h << 5) - h);
    return `hsl(${((h % 360) + 360) % 360}, 65%, 60%)`;
}

function kfUrl(rawPath) {
    // Convert "output/anime_metadata\\0001\\keyframes\\shot_0001_kf_00.jpg" -> "/keyframe/0001/shot_0001_kf_00.jpg"
    const norm = rawPath.replace(/\\/g, '/');
    const match = norm.match(/anime_metadata\/(\d{4})\/keyframes\/(.+)/);
    if (match) return '/keyframe/' + match[1] + '/' + match[2];
    return rawPath;
}

function sceneTypeBadge(stype) {
    const colors = {
        story: '#2a4a7a', flashback: '#7a5a2a', opening: '#5a2a7a',
        ending: '#5a2a7a', eyecatch: '#2a7a5a', transition: '#3a3a4a',
    };
    const bg = colors[stype] || '#3a3a4a';
    return `<span class="badge" style="background:${bg}">${esc(stype)}</span>`;
}

function originBadge(origin) {
    const cls = 'origin-' + (origin || 'unknown').replace(/ /g, '_');
    return `<span class="badge ${cls}">${esc(origin || 'unknown')}</span>`;
}

function emotionSpan(emo) {
    const cls = EMOTION_CLASSES[emo] || '';
    return `<span class="${cls}">${esc(emo)}</span>`;
}

// ========================================================================
// Init
// ========================================================================
async function init() {
    const res = await fetch('/api/episodes');
    const data = await res.json();
    state.episodes = data.episodes;

    const sel = document.getElementById('episode-select');
    sel.innerHTML = data.episodes.map(e =>
        `<option value="${e.episode_id}" ${e.episode_id === data.current ? 'selected' : ''}>${e.episode_id} - ${esc(e.title)}</option>`
    ).join('');

    await loadEpisode(data.current || data.episodes[0]?.episode_id);
}

async function loadEpisode(epId) {
    state.currentEpisode = epId;
    state.sceneCache = {};
    state.currentSceneNum = null;
    state.currentShotId = null;

    // Parallel fetch
    const [mRes, sRes, cRes] = await Promise.all([
        fetch(`/api/episode/${epId}/manifest`),
        fetch(`/api/episode/${epId}/shots`),
        fetch(`/api/episode/${epId}/characters`),
    ]);
    state.manifest = await mRes.json();
    state.shots = await sRes.json();
    state.characters = await cRes.json();

    // Build indexes
    state.shotsByScene = {};
    state.shotLookup = {};
    if (Array.isArray(state.shots)) {
        for (const shot of state.shots) {
            const sn = shot.scene_number;
            if (!state.shotsByScene[sn]) state.shotsByScene[sn] = [];
            state.shotsByScene[sn].push(shot);
            state.shotLookup[shot.shot_id] = shot;
        }
    }

    // Set video
    const video = document.getElementById('video');
    video.src = `/video/${epId}`;

    // Render static parts
    renderTopBarStats();
    renderOverview();
    renderSceneTimeline();

    // Load first scene
    const firstScene = state.manifest.scenes?.[0];
    if (firstScene) {
        await onSceneChange(firstScene.scene_number);
    }
}

function switchEpisode(epId) {
    loadEpisode(epId);
}

// ========================================================================
// Top bar
// ========================================================================
function renderTopBarStats() {
    const m = state.manifest;
    if (!m) return;
    const el = document.getElementById('ep-stats');
    el.innerHTML = `${sceneTypeBadge(m.arc_id)} &nbsp; ${m.total_scenes} scenes &middot; ${m.total_shots} shots &middot; ${fmt(m.total_duration_seconds)}`;
}

// ========================================================================
// Overview panel
// ========================================================================
function renderOverview() {
    const m = state.manifest;
    const c = state.characters;

    // Stats
    const ov = document.getElementById('ep-overview');
    ov.innerHTML = `
        <div class="panel-section-title">Episode Overview</div>
        <div class="stat-row"><span>Title</span><span class="val">${esc(m.episode_title)}</span></div>
        <div class="stat-row"><span>Arc</span><span class="val">${esc(m.arc_id)}</span></div>
        <div class="stat-row"><span>Scenes</span><span class="val">${m.total_scenes}</span></div>
        <div class="stat-row"><span>Shots</span><span class="val">${m.total_shots}</span></div>
        <div class="stat-row"><span>Duration</span><span class="val">${fmt(m.total_duration_seconds)}</span></div>
    `;

    // Characters
    const cl = document.getElementById('char-list');
    if (c && c.characters) {
        const chars = Object.values(c.characters);
        cl.innerHTML = chars.map(ch => `
            <div class="char-item" title="${esc(ch.voice_profile?.observed_emotions?.join(', ') || '')}">
                <span class="char-dot" style="background:${charColor(ch.character_id)}"></span>
                <span class="name">${esc(ch.name)}</span>
                <span class="count">${ch.appearances?.length || 0} scenes</span>
            </div>
        `).join('');
    }

    // Scene list
    renderSceneList();
}

function renderSceneList() {
    const sl = document.getElementById('scene-list');
    if (!state.manifest?.scenes) return;
    sl.innerHTML = state.manifest.scenes.map(s => `
        <div class="scene-item ${s.scene_number === state.currentSceneNum ? 'active' : ''}"
             onclick="seekToScene(${s.scene_number})" data-scene="${s.scene_number}">
            <span class="num">${s.scene_number}</span>
            ${sceneTypeBadge(s.scene_type)}
            <span class="type-badge" style="background:${SCENE_TYPE_COLORS[s.scene_type] || '#3a3a4a'}">${esc(s.scene_type)}</span>
            <span class="time">${fmt(s.start_time)}</span>
        </div>
    `).join('');
}

// ========================================================================
// Scene timeline
// ========================================================================
function renderSceneTimeline() {
    const el = document.getElementById('scene-timeline');
    if (!state.manifest?.scenes) return;
    const total = state.manifest.total_duration_seconds || 1;

    let html = '';
    for (const s of state.manifest.scenes) {
        const left = (s.start_time / total) * 100;
        const width = (s.duration / total) * 100;
        const cls = 'stype-' + (s.scene_type || 'story');
        html += `<div class="timeline-block ${cls} ${s.scene_number === state.currentSceneNum ? 'current-block' : ''}"
                      style="left:${left}%;width:${width}%"
                      onclick="seekToTime(${s.start_time})"
                      title="Scene ${s.scene_number}: ${esc(s.scene_type)} (${fmt(s.start_time)}-${fmt(s.end_time)})"
                      data-scene="${s.scene_number}">
                     ${width > 3 ? s.scene_number : ''}
                 </div>`;
    }
    html += '<div class="playhead" id="scene-playhead" style="left:0%"></div>';
    el.innerHTML = html;

    // Click on empty space to seek
    el.addEventListener('click', function(e) {
        if (e.target === el) {
            const rect = el.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            seekToTime(pct * total);
        }
    });
}

function renderShotTimeline(sceneNum) {
    const el = document.getElementById('shot-timeline');
    const label = document.getElementById('shot-timeline-label');
    const shots = state.shotsByScene[sceneNum] || [];
    const scene = state.manifest.scenes?.find(s => s.scene_number === sceneNum);
    if (!scene || shots.length === 0) {
        el.innerHTML = '';
        label.textContent = '';
        return;
    }

    label.textContent = `(Scene ${sceneNum}: ${shots.length} shots)`;
    const sceneStart = scene.start_time;
    const sceneDur = scene.duration || 1;

    let html = '';
    for (const shot of shots) {
        const left = ((shot.start_time_seconds - sceneStart) / sceneDur) * 100;
        const width = (shot.duration_seconds / sceneDur) * 100;
        const bg = SHOT_TYPE_COLORS[shot.shot_type] || '#3a3a5a';
        html += `<div class="timeline-block ${shot.shot_id === state.currentShotId ? 'current-block' : ''}"
                      style="left:${left}%;width:${width}%;background:${bg}"
                      onclick="seekToTime(${shot.start_time_seconds})"
                      title="Shot ${shot.shot_id}: ${esc(shot.shot_type)} / ${esc(shot.content_origin)} (${fmt(shot.start_time_seconds)}-${fmt(shot.end_time_seconds)})"
                      data-shot="${shot.shot_id}">
                     ${width > 4 ? shot.shot_id : ''}
                 </div>`;
    }
    html += '<div class="playhead" id="shot-playhead" style="left:0%"></div>';
    el.innerHTML = html;
}

// ========================================================================
// Scene / shot change handlers
// ========================================================================
async function onSceneChange(sceneNum) {
    if (sceneNum === state.currentSceneNum) return;
    state.currentSceneNum = sceneNum;

    // Fetch scene data if not cached
    if (!state.sceneCache[sceneNum]) {
        const res = await fetch(`/api/episode/${state.currentEpisode}/scene/${sceneNum}`);
        if (res.ok) state.sceneCache[sceneNum] = await res.json();
    }

    renderSceneDetail(state.sceneCache[sceneNum]);
    renderShotTimeline(sceneNum);
    renderSceneList();

    // Highlight in scene timeline
    document.querySelectorAll('#scene-timeline .timeline-block').forEach(b => {
        b.classList.toggle('current-block', parseInt(b.dataset.scene) === sceneNum);
    });
}

function onShotChange(shotId) {
    if (shotId === state.currentShotId) return;
    state.currentShotId = shotId;
    const shot = state.shotLookup[shotId];
    if (shot) {
        renderShotDetail(shot);
        renderKeyframesStrip(shot);
    }

    // Highlight in shot timeline
    document.querySelectorAll('#shot-timeline .timeline-block').forEach(b => {
        b.classList.toggle('current-block', parseInt(b.dataset.shot) === shotId);
    });
}

// ========================================================================
// Scene detail panel
// ========================================================================
function renderSceneDetail(scene) {
    const el = document.getElementById('scene-detail');
    if (!scene) { el.innerHTML = '<div class="detail-section" style="color:var(--text-muted)">No scene data</div>'; return; }

    let html = '';

    // Header
    html += `<div class="detail-section">
        <div class="scene-header">
            <span class="scene-num">Scene ${scene.scene_number}</span>
            ${sceneTypeBadge(scene.scene_type)}
            ${scene.anime_original ? '<span class="badge origin-anime_original">anime original</span>' : ''}
        </div>
        <div class="meta-grid">
            <span class="label">Time</span><span class="value">${fmt(scene.start_time_seconds)} - ${fmt(scene.end_time_seconds)} (${fmt(scene.duration_seconds)})</span>
            <span class="label">Location</span><span class="value">${esc(scene.location || '?')}</span>
            <span class="label">Time ctx</span><span class="value">${esc(scene.time_context || '?')}</span>
            <span class="label">Tone</span><span class="value">${emotionSpan(scene.emotional_tone)}</span>
            <span class="label">Shots</span><span class="value">${scene.shot_count}</span>
            <span class="label">Motion</span><span class="value"><div class="intensity-bar"><div class="intensity-fill" style="width:${(scene.motion_intensity || 0) * 100}%"></div></div></span>
        </div>
    </div>`;

    // Narrative
    if (scene.scene_narrative) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Narrative</div>
            <div class="narrative-text">${esc(scene.scene_narrative)}</div>
        </div>`;
    }

    // Characters
    if (scene.characters_present?.length) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Characters</div>`;
        for (const ch of scene.characters_present) {
            const col = charColor(ch.character_id || ch.name);
            html += `<div class="char-card">
                <span class="char-name" style="color:${col}">${esc(ch.name)}</span>
                <span class="char-emotion badge" style="background:rgba(255,255,255,0.08)">${esc(ch.emotion || '')}</span>
                <div class="char-action">${esc(ch.action || '')}</div>
            </div>`;
        }
        html += `</div>`;
    }

    // Dialogue
    if (scene.text_elements?.length) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Dialogue (${scene.text_elements.length})</div>
            <div id="dialogue-list">`;
        for (const d of scene.text_elements) {
            const col = charColor(d.speaker_id || d.speaker || 'unknown');
            const isNarr = d.type === 'narration';
            html += `<div class="dialogue-item" data-start="${d.start_time_seconds}" data-end="${d.end_time_seconds}">
                <span class="speaker" style="color:${col}">${esc(d.speaker)}</span>
                <span class="meta">${fmt(d.start_time_seconds)}</span>
                <div class="content ${isNarr ? 'narration' : ''}">${esc(d.content)}</div>
                <div class="meta">${esc(d.emotion || '')} ${d.tone_direction ? '- ' + esc(d.tone_direction) : ''}</div>
            </div>`;
        }
        html += `</div></div>`;
    }

    // Shot type sequence
    if (scene.shot_type_sequence?.length) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Shot Type Sequence</div>
            <div class="shot-type-bar">`;
        for (const st of scene.shot_type_sequence) {
            const bg = SHOT_TYPE_COLORS[st] || '#3a3a5a';
            html += `<div class="shot-type-block" style="background:${bg};height:${st === 'wide' ? 22 : st === 'close-up' ? 14 : 18}px" title="${esc(st)}"></div>`;
        }
        html += `</div></div>`;
    }

    // Reconstruction prompt
    if (scene.reconstruction_prompt) {
        html += `<div class="detail-section">
            <div class="collapsible-header" onclick="toggleCollapse(this)">
                <span class="arrow">&#9654;</span>
                <span class="detail-section-title" style="margin-bottom:0">Reconstruction Prompt</span>
            </div>
            <div class="collapsible-body">
                <div class="recon-prompt">${esc(scene.reconstruction_prompt)}</div>
            </div>
        </div>`;
    }

    el.innerHTML = html;
}

function toggleCollapse(header) {
    const arrow = header.querySelector('.arrow');
    const body = header.nextElementSibling;
    arrow.classList.toggle('open');
    body.classList.toggle('open');
}

// ========================================================================
// Shot detail panel
// ========================================================================
function renderShotDetail(shot) {
    const el = document.getElementById('shot-detail');
    if (!shot) { el.innerHTML = ''; return; }

    let html = `<div class="detail-section">
        <div class="detail-section-title">Current Shot</div>
        <div class="scene-header">
            <span class="scene-num" style="font-size:15px">Shot ${shot.shot_id}</span>
            <span class="badge" style="background:${SHOT_TYPE_COLORS[shot.shot_type] || '#3a3a5a'}">${esc(shot.shot_type)}</span>
            ${originBadge(shot.content_origin)}
        </div>
        <div class="meta-grid">
            <span class="label">Time</span><span class="value">${fmt(shot.start_time_seconds)} - ${fmt(shot.end_time_seconds)} (${shot.duration_seconds?.toFixed(1)}s)</span>
            <span class="label">In scene</span><span class="value">#${shot.shot_number_in_scene} of scene ${shot.scene_number}</span>
            <span class="label">Motion</span><span class="value"><div class="intensity-bar"><div class="intensity-fill" style="width:${(shot.motion_intensity || 0) * 100}%"></div></div></span>
            <span class="label">Fade</span><span class="value">${esc(shot.fade_type || 'none')}</span>
        </div>`;

    // Audio classification
    const ac = shot.audio_classification;
    if (ac) {
        html += `<div style="margin-top:8px">
            <div class="detail-section-title">Audio</div>
            <div class="meta-grid">
                <span class="label">Type</span><span class="value"><span class="badge" style="background:#2a3a5a">${esc(ac.audio_type)}</span></span>
                <span class="label">Music</span><span class="value">${ac.has_music ? 'Yes' : 'No'} (${(ac.music_confidence * 100).toFixed(0)}%)</span>
                <span class="label">RMS</span><span class="value">${ac.rms_db?.toFixed(1)} dB</span>
                <span class="label">Spectral</span><span class="value">${ac.spectral_flatness?.toFixed(4)}</span>
                <span class="label">Onsets</span><span class="value">${ac.onset_rate?.toFixed(1)}/s</span>
            </div>
        </div>`;
    }

    // Color palette
    if (shot.color_palette?.length) {
        html += `<div style="margin-top:8px">
            <div class="detail-section-title">Color Palette</div>
            <div class="color-palette">`;
        for (const hex of shot.color_palette) {
            html += `<div class="color-swatch" style="background:${hex}" title="${hex}"></div>`;
        }
        html += `</div></div>`;
    }

    // Dialogue in shot
    if (shot.dialogue_in_shot?.length) {
        html += `<div style="margin-top:8px">
            <div class="detail-section-title">Dialogue IDs</div>
            <div style="font-size:11px;color:var(--text-secondary)">${shot.dialogue_in_shot.map(esc).join(', ')}</div>
        </div>`;
    }

    // Characters
    if (shot.characters_present?.length) {
        html += `<div style="margin-top:8px">
            <div class="detail-section-title">Characters</div>
            <div style="font-size:11px;color:var(--text-secondary)">${shot.characters_present.map(esc).join(', ')}</div>
        </div>`;
    }

    html += `</div>`;
    el.innerHTML = html;
}

// ========================================================================
// Keyframes strip
// ========================================================================
function renderKeyframesStrip(shot) {
    const el = document.getElementById('keyframes-strip');
    if (!shot) { el.innerHTML = ''; return; }

    let html = '';

    // Shot keyframes
    if (shot.keyframe_paths?.length) {
        html += `<div class="kf-group">
            <span class="kf-group-label">Keyframes (${shot.keyframe_paths.length})</span>
            <div class="kf-row">`;
        for (const kf of shot.keyframe_paths) {
            html += `<img class="kf-thumb" src="${kfUrl(kf)}" loading="lazy" title="${esc(kf.split(/[/\\]/).pop())}">`;
        }
        html += `</div></div>`;
    }

    // Background plate
    if (shot.background_plate_path) {
        html += `<div class="kf-divider"></div>`;
        html += `<div class="kf-group">
            <span class="kf-group-label">BG Plate</span>
            <div class="kf-row">
                <img class="kf-thumb" src="${kfUrl(shot.background_plate_path)}" loading="lazy" title="Background plate">
            </div>
        </div>`;
    }

    // Character crops
    if (shot.character_crops?.length) {
        html += `<div class="kf-divider"></div>`;
        html += `<div class="kf-group">
            <span class="kf-group-label">Crops (${shot.character_crops.length})</span>
            <div class="kf-row">`;
        for (const crop of shot.character_crops) {
            html += `<div class="crop-container">
                <img class="crop-thumb" src="${kfUrl(crop.crop_image_path)}" loading="lazy"
                     title="Conf: ${(crop.confidence * 100).toFixed(0)}% | Area: ${(crop.bbox_area_fraction * 100).toFixed(1)}% | KF: ${crop.keyframe_index}">
                <span class="crop-conf">${(crop.confidence * 100).toFixed(0)}%</span>
            </div>`;
        }
        html += `</div></div>`;
    }

    if (!html) {
        html = '<span style="color:var(--text-muted);font-size:12px">No keyframes for this shot</span>';
    }

    el.innerHTML = html;
}

// ========================================================================
// Video time tracking
// ========================================================================
function findSceneAt(time) {
    if (!state.manifest?.scenes) return null;
    for (const s of state.manifest.scenes) {
        if (time >= s.start_time && time < s.end_time) return s;
    }
    // Edge: at very end
    const last = state.manifest.scenes[state.manifest.scenes.length - 1];
    if (last && time >= last.start_time) return last;
    return null;
}

function findShotAt(time, sceneNum) {
    const shots = state.shotsByScene[sceneNum];
    if (!shots) return null;
    for (const s of shots) {
        if (time >= s.start_time_seconds && time < s.end_time_seconds) return s;
    }
    const last = shots[shots.length - 1];
    if (last && time >= last.start_time_seconds) return last;
    return null;
}

function onTimeUpdate() {
    const video = document.getElementById('video');
    const time = video.currentTime;
    const total = state.manifest?.total_duration_seconds || 1;

    // Scene playhead
    const scenePh = document.getElementById('scene-playhead');
    if (scenePh) scenePh.style.left = (time / total * 100) + '%';

    // Find current scene
    const scene = findSceneAt(time);
    if (scene) {
        if (scene.scene_number !== state.currentSceneNum) {
            onSceneChange(scene.scene_number);
        }

        // Shot playhead
        const shotPh = document.getElementById('shot-playhead');
        if (shotPh && scene.duration > 0) {
            shotPh.style.left = ((time - scene.start_time) / scene.duration * 100) + '%';
        }

        // Find current shot
        const shot = findShotAt(time, scene.scene_number);
        if (shot && shot.shot_id !== state.currentShotId) {
            onShotChange(shot.shot_id);
        }
    }

    // Highlight active dialogue
    highlightDialogue(time);
}

function highlightDialogue(time) {
    const items = document.querySelectorAll('#dialogue-list .dialogue-item');
    for (const item of items) {
        const start = parseFloat(item.dataset.start);
        const end = parseFloat(item.dataset.end);
        const active = time >= start && time < end;
        item.classList.toggle('active', active);
        if (active && !isElementVisible(item)) {
            item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
}

function isElementVisible(el) {
    const parent = el.closest('.detail-panel') || el.parentElement;
    const rect = el.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    return rect.top >= parentRect.top && rect.bottom <= parentRect.bottom;
}

// ========================================================================
// Seeking
// ========================================================================
function seekToTime(sec) {
    const video = document.getElementById('video');
    video.currentTime = sec;
}

function seekToScene(sceneNum) {
    const scene = state.manifest?.scenes?.find(s => s.scene_number === sceneNum);
    if (scene) seekToTime(scene.start_time);
}

function seekToNextScene(dir) {
    if (!state.currentSceneNum || !state.manifest?.scenes) return;
    const idx = state.manifest.scenes.findIndex(s => s.scene_number === state.currentSceneNum);
    const next = state.manifest.scenes[idx + dir];
    if (next) seekToTime(next.start_time);
}

function seekToNextShot(dir) {
    if (!state.currentShotId || !state.currentSceneNum) return;
    const shots = state.shotsByScene[state.currentSceneNum] || [];
    const idx = shots.findIndex(s => s.shot_id === state.currentShotId);
    const next = shots[idx + dir];
    if (next) {
        seekToTime(next.start_time_seconds);
    } else if (dir > 0) {
        // Jump to first shot of next scene
        seekToNextScene(1);
    }
}

// ========================================================================
// Keyboard shortcuts
// ========================================================================
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    const video = document.getElementById('video');
    switch (e.key) {
        case ' ':
            e.preventDefault();
            video.paused ? video.play() : video.pause();
            break;
        case 'ArrowLeft':
            e.preventDefault();
            video.currentTime -= (e.shiftKey ? 1 : 5);
            break;
        case 'ArrowRight':
            e.preventDefault();
            video.currentTime += (e.shiftKey ? 1 : 5);
            break;
        case '[':
            e.preventDefault();
            seekToNextScene(-1);
            break;
        case ']':
            e.preventDefault();
            seekToNextScene(1);
            break;
        case ',':
            e.preventDefault();
            seekToNextShot(-1);
            break;
        case '.':
            e.preventDefault();
            seekToNextShot(1);
            break;
        case 'o':
        case 'O':
            e.preventDefault();
            state.overviewVisible = !state.overviewVisible;
            document.getElementById('overview-panel').classList.toggle('hidden', !state.overviewVisible);
            break;
    }
});

// ========================================================================
// Video event binding
// ========================================================================
const video = document.getElementById('video');
video.addEventListener('timeupdate', onTimeUpdate);
video.addEventListener('seeked', onTimeUpdate);

// Start
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anime Metadata Viewer")
    parser.add_argument("--episode", default="0001", help="Episode ID to load (default: 0001)")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    startup_episode = args.episode
    PORT = args.port

    print(f"Anime Metadata Viewer starting on http://localhost:{PORT}")
    print(f"Default episode: {startup_episode}")

    if not args.no_browser:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    app.run(host="0.0.0.0", port=PORT, debug=False)
