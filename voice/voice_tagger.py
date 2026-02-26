"""
Voice Tagger — Interactive UI for tagging anime character voice samples.

Flask + HTML5 web app. Run with: python voice_tagger.py
Opens at http://localhost:7861
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import mimetypes
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from voice import voice_tagger_backend as backend

app = Flask(__name__)


# ============================================================
# VIDEO SERVING
# ============================================================
@app.route("/video/<episode_id>")
def serve_video(episode_id):
    """Serve episode MP4 with range request support for seeking."""
    mp4_path = backend.ensure_episode_mp4(episode_id)
    if not mp4_path or not mp4_path.exists():
        return "Video not found", 404

    file_size = mp4_path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        # Parse byte range
        byte_start = 0
        byte_end = file_size - 1
        match = __import__("re").search(r"bytes=(\d+)-(\d*)", range_header)
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


# ============================================================
# FACE THUMBNAIL SERVING
# ============================================================
@app.route("/face_thumb/<episode_id>/<face_id>")
def serve_face_thumb(episode_id, face_id):
    thumb_path = backend.VOICE_TAGS_DIR / episode_id / "face_thumbs" / f"{face_id}.jpg"
    if thumb_path.exists():
        return send_file(str(thumb_path), mimetype="image/jpeg")
    return "Not found", 404


@app.route("/reference/<character>/<filename>")
def serve_reference(character, filename):
    ref_path = backend.REFERENCES_DIR / character / filename
    if ref_path.exists():
        mime = mimetypes.guess_type(str(ref_path))[0] or "image/jpeg"
        return send_file(str(ref_path), mimetype=mime)
    return "Not found", 404


# ============================================================
# AUDIO SAMPLE SERVING
# ============================================================
@app.route("/sample/<episode_id>/<character>/<filename>")
def serve_sample(episode_id, character, filename):
    sample_path = backend.VOICE_TAGS_DIR / episode_id / "voice_samples" / character / filename
    if sample_path.exists():
        return send_file(str(sample_path), mimetype="audio/wav")
    return "Not found", 404


@app.route("/evaluation/<character>/<filename>")
def serve_evaluation(character, filename):
    eval_path = backend.VOICE_TAGS_DIR / "_evaluations" / character / filename
    if eval_path.exists():
        return send_file(str(eval_path), mimetype="audio/wav")
    return "Not found", 404


# ============================================================
# API ROUTES
# ============================================================
@app.route("/api/episodes")
def api_episodes():
    eps = []
    for ep_id in backend.EPISODE_MKVS:
        info = backend.get_episode_info(ep_id)
        if info:
            # Check if VLM scene data is available
            scenes = backend.load_vlm_scenes(ep_id)
            info["has_vlm_data"] = len(scenes) > 0
            info["scene_count"] = len(scenes)
            eps.append(info)
    return jsonify(eps)


@app.route("/api/episode/<episode_id>/subtitles")
def api_subtitles(episode_id):
    subs = backend.extract_subtitles(episode_id)
    return jsonify(subs)


@app.route("/api/episode/<episode_id>/info")
def api_episode_info(episode_id):
    info = backend.get_episode_info(episode_id)
    return jsonify(info or {})


# ---- Frame analysis (VLM scene lookup) ----
@app.route("/api/episode/<episode_id>/analyze_frame", methods=["POST"])
def api_analyze_frame(episode_id):
    body = request.get_json()
    time_sec = body.get("time_sec", 0)
    result = backend.analyze_frame(episode_id, time_sec)
    return jsonify(result)


# ---- Tags ----
@app.route("/api/episode/<episode_id>/tags")
def api_get_tags(episode_id):
    return jsonify(backend.load_tags(episode_id))


@app.route("/api/episode/<episode_id>/tags", methods=["POST"])
def api_create_tag(episode_id):
    body = request.get_json()
    tag = backend.create_tag(
        episode_id,
        character=body["character"],
        start_sec=body["start_sec"],
        end_sec=body["end_sec"],
        subtitle_text=body.get("subtitle_text", ""),
        face_cluster_id=body.get("face_cluster_id"),
    )
    return jsonify(tag)


@app.route("/api/episode/<episode_id>/tags/<tag_id>", methods=["DELETE"])
def api_delete_tag(episode_id, tag_id):
    backend.delete_tag(episode_id, tag_id)
    return jsonify({"ok": True})


@app.route("/api/episode/<episode_id>/tags/<tag_id>", methods=["PUT"])
def api_update_tag(episode_id, tag_id):
    body = request.get_json()
    backend.update_tag(episode_id, tag_id, body)
    return jsonify({"ok": True})


@app.route("/api/episode/<episode_id>/rename_speaker", methods=["POST"])
def api_rename_speaker(episode_id):
    body = request.get_json()
    old_name = body["old_name"]
    new_name = body["new_name"]
    data = backend.load_tags(episode_id)
    count = 0
    for tag in data["tags"]:
        if tag["character"] == old_name:
            tag["character"] = new_name
            count += 1
    backend.save_tags(episode_id, data)
    return jsonify({"ok": True, "renamed": count})


@app.route("/api/episode/<episode_id>/delete_speaker", methods=["POST"])
def api_delete_speaker(episode_id):
    body = request.get_json()
    speaker = body["speaker"]
    data = backend.load_tags(episode_id)
    before = len(data["tags"])
    data["tags"] = [t for t in data["tags"] if t["character"] != speaker]
    after = len(data["tags"])
    backend.save_tags(episode_id, data)
    return jsonify({"ok": True, "deleted": before - after})


@app.route("/api/episode/<episode_id>/speaker_recommendation", methods=["POST"])
def api_speaker_recommendation(episode_id):
    """Recommend a character for a speaker based on VLM scene overlap."""
    body = request.get_json()
    speaker = body["speaker"]
    data = backend.load_tags(episode_id)
    speaker_tags = [t for t in data["tags"] if t["character"] == speaker]
    if not speaker_tags:
        return jsonify({"character": None, "reason": "no tags"})

    scenes = backend.load_vlm_scenes(episode_id)
    if not scenes:
        return jsonify({"character": None, "reason": "no VLM data"})

    # For each speaker tag, find which VLM characters are present at that time
    from collections import Counter
    char_overlap = Counter()
    for tag in speaker_tags:
        mid = (tag["start_sec"] + tag["end_sec"]) / 2
        scene = backend.get_scene_at_time(episode_id, mid)
        if not scene:
            continue
        # Check text_elements for speaker attribution
        for te in scene.get("text_elements", []):
            te_start = te.get("start_time_seconds", 0)
            te_end = te.get("end_time_seconds", 0)
            if abs(te_start - tag["start_sec"]) < 2.0 or (te_start <= mid <= te_end):
                spk = te.get("speaker", "")
                if spk and backend._is_real_character(spk):
                    char_overlap[spk] += tag.get("duration_sec", 0)
                    break
        else:
            # Fallback: count characters_present
            for char in scene.get("characters_present", []):
                name = char.get("name", char.get("character_id", ""))
                if backend._is_real_character(name):
                    char_overlap[name] += tag.get("duration_sec", 0) / max(len(scene.get("characters_present", [])), 1)

    if not char_overlap:
        return jsonify({"character": None, "reason": "no VLM overlap"})

    best = char_overlap.most_common(3)
    total = sum(v for _, v in best)
    return jsonify({
        "character": best[0][0],
        "confidence": round(best[0][1] / total, 2) if total > 0 else 0,
        "candidates": [{"name": n, "score": round(s, 1)} for n, s in best],
    })


# ---- Recommendations ----
@app.route("/api/episode/<episode_id>/recommendations")
def api_recommendations(episode_id):
    recs = backend.get_recommendations(episode_id)
    return jsonify(recs)


# ---- Audio extraction ----
@app.route("/api/episode/<episode_id>/extract_sample", methods=["POST"])
def api_extract_sample(episode_id):
    body = request.get_json()
    tag_id = body["tag_id"]
    use_demucs = body.get("use_demucs", False)
    result = backend.extract_voice_sample(episode_id, tag_id, use_demucs=use_demucs)
    return jsonify({"path": result})


# ---- Voice clone evaluation ----
@app.route("/api/character/<character>/evaluate", methods=["POST"])
def api_evaluate_clone(character):
    result = backend.evaluate_voice_clone(character)
    return jsonify(result)


# ---- Characters ----
@app.route("/api/characters")
def api_characters():
    names = backend.get_character_names()
    # Get per-character totals across episodes
    char_data = {}
    for name in names:
        total_sec = 0
        total_tags = 0
        for ep_id in backend.EPISODE_MKVS:
            tags = backend.load_tags(ep_id)
            ct = tags.get("character_totals", {}).get(name, {})
            total_sec += ct.get("total_seconds", 0)
            total_tags += ct.get("tag_count", 0)

        # Get reference images
        ref_dir = backend.REFERENCES_DIR / name
        ref_imgs = []
        if ref_dir.exists():
            ref_imgs = [f.name for f in sorted(ref_dir.glob("*"))
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

        char_data[name] = {
            "name": name,
            "total_seconds": round(total_sec, 1),
            "total_tags": total_tags,
            "status": "ready" if total_sec >= backend.SCAN_CONFIG["target_seconds"] else "in_progress",
            "reference_images": ref_imgs[:3],
        }
    return jsonify(char_data)


# ============================================================
# MAIN HTML PAGE
# ============================================================
@app.route("/")
def index():
    return HTML_TEMPLATE


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Tagger</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; }

/* Top bar */
.topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px; background: #1a1a2e; border-bottom: 1px solid #333;
}
.topbar select { background: #252540; color: #fff; border: 1px solid #444; padding: 6px 10px; border-radius: 4px; }
.topbar button { background: #3a3a5c; color: #fff; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; }
.topbar button:hover { background: #4a4a7c; }
.topbar .status { color: #888; font-size: 13px; margin-left: auto; }

/* Main layout */
.main { display: flex; height: calc(100vh - 42px - 200px - 240px); min-height: 300px; }

/* Video panel */
.video-panel { flex: 1; position: relative; background: #000; display: flex; align-items: center; justify-content: center; }
.video-panel video { max-width: 100%; max-height: 100%; }
.video-panel canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
.video-controls {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: rgba(0,0,0,0.8); padding: 8px 12px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.video-controls button {
    background: #3a3a5c; color: #fff; border: none; padding: 4px 10px;
    border-radius: 3px; cursor: pointer; font-size: 12px;
}
.video-controls button:hover { background: #5a5a8c; }
.video-controls button.active { background: #6a4aac; }
.video-controls .time-display { color: #aaa; font-size: 12px; font-family: monospace; }
.selection-display { color: #ffcc00; font-size: 12px; font-family: monospace; }

/* Character panel */
.char-panel {
    width: 280px; background: #1a1a1a; border-left: 1px solid #333;
    overflow-y: auto; padding: 8px;
}
.char-panel h3 { font-size: 13px; color: #888; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
.recent-chars { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; min-height: 26px; }
.recent-chip {
    padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;
    cursor: pointer; border: 1px solid #444; background: #1a1a2e; white-space: nowrap;
    transition: background 0.15s;
}
.recent-chip:hover { background: #2a2a4e; }
.recent-chip.selected { border-color: #6a4aac; background: #2a1a4e; }
.char-search {
    width: 100%; box-sizing: border-box; padding: 6px 8px; margin-bottom: 8px;
    background: #222; color: #eee; border: 1px solid #444; border-radius: 4px;
    font-size: 13px; outline: none;
}
.char-search:focus { border-color: #6a4aac; }
.char-item {
    padding: 6px 8px; margin-bottom: 4px; border-radius: 4px;
    cursor: pointer; display: flex; align-items: center; gap: 8px;
    border: 1px solid transparent;
}
.char-item:hover { background: #252525; }
.char-item.selected { border-color: #6a4aac; background: #1a1a2e; }
.char-item.ready { border-color: #2a6a2a; }
.char-item .char-thumb { width: 32px; height: 32px; border-radius: 4px; object-fit: cover; background: #333; }
.char-item .char-info { flex: 1; min-width: 0; }
.char-item .char-name { font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.char-item .char-progress { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
.char-item .progress-bar { flex: 1; height: 4px; background: #333; border-radius: 2px; overflow: hidden; }
.char-item .progress-fill { height: 100%; border-radius: 2px; transition: width 0.3s; }
.char-item .progress-text { font-size: 10px; color: #888; white-space: nowrap; }
.char-actions { margin-top: 12px; }
.char-actions button {
    width: 100%; padding: 8px; margin-bottom: 6px;
    background: #3a3a5c; color: #fff; border: none; border-radius: 4px; cursor: pointer;
}
.char-actions button:hover { background: #5a5a8c; }
.char-actions button.clone-btn { background: #2a5a2a; }
.char-actions button.clone-btn:hover { background: #3a7a3a; }

/* Subtitle strip */
.sub-strip {
    height: 200px; background: #111; border-top: 1px solid #333;
    overflow-x: auto; overflow-y: hidden; position: relative;
    white-space: nowrap;
}
.sub-strip-inner { position: relative; height: 100%; padding: 8px 0; }
.sub-strip-header {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 12px; background: #1a1a1a; border-top: 1px solid #333;
    font-size: 12px; color: #888;
}
.sub-block {
    display: inline-block; vertical-align: top;
    margin: 4px 2px; padding: 6px 8px;
    border-radius: 4px; cursor: pointer;
    font-size: 11px; line-height: 1.3;
    max-width: 200px; white-space: normal;
    border: 2px solid transparent;
    min-height: 50px;
    overflow: hidden;
}
.sub-block:hover { opacity: 0.9; }
.sub-block.untagged { background: #252525; color: #999; }
.sub-block.tagged { color: #fff; }
.sub-block.suggested { border-style: dashed; }
.sub-block.current { box-shadow: 0 0 0 2px #fff; }
.sub-block .sub-time { font-size: 9px; color: rgba(255,255,255,0.5); display: block; margin-bottom: 2px; }
.sub-block .sub-char { font-size: 9px; font-weight: 700; display: block; margin-bottom: 2px; }
.sub-block .sub-text { display: block; word-break: break-word; }

/* Bottom panel — tags + recommendations */
.bottom-panel {
    height: 300px; display: flex; border-top: 1px solid #333;
}
.tags-panel {
    flex: 1; overflow-y: auto; padding: 8px 12px; background: #0f0f0f;
}
.tags-panel h3 { font-size: 13px; color: #888; margin-bottom: 8px; }
.spk-group { margin-bottom: 8px; border: 1px solid #2a2a2a; border-radius: 6px; overflow: hidden; }
.spk-header {
    display: flex; align-items: center; gap: 8px; padding: 6px 10px;
    background: #1a1a1a; cursor: pointer; font-size: 12px; user-select: none;
}
.spk-header:hover { background: #222; }
.spk-name { font-weight: 700; font-size: 13px; min-width: 120px; }
.spk-stats { color: #888; font-size: 11px; flex: 1; }
.spk-rec { font-size: 11px; color: #6a8; margin-right: 4px; }
.spk-assign {
    font-size: 11px; padding: 2px 8px; border: 1px solid #555; border-radius: 3px;
    background: #2a2a4e; color: #ccc; cursor: pointer;
}
.spk-assign:hover { background: #3a3a6e; border-color: #6a4aac; }
.spk-del-all {
    font-size: 11px; padding: 2px 8px; border: 1px solid #533; border-radius: 3px;
    background: #2a1a1a; color: #c88; cursor: pointer;
}
.spk-del-all:hover { background: #3a2a2a; border-color: #c44; }
.spk-segments { display: none; padding: 2px 0; background: #111; }
.spk-segments.open { display: block; }
.spk-seg {
    display: flex; align-items: center; gap: 6px; padding: 3px 10px;
    font-size: 11px; cursor: pointer; color: #aaa;
}
.spk-seg:hover { background: #1a1a2a; }
.spk-seg .seg-time { font-family: monospace; color: #888; min-width: 100px; }
.spk-seg .seg-dur { color: #666; min-width: 40px; }
.spk-seg .seg-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #777; }
.spk-seg .seg-del { color: #c44; cursor: pointer; padding: 0 4px; }
.spk-seg .seg-del:hover { color: #f66; }

.recs-panel {
    width: 320px; overflow-y: auto; padding: 8px 12px;
    background: #111; border-left: 1px solid #333;
}
.recs-panel h3 { font-size: 13px; color: #888; margin-bottom: 8px; }
.rec-item {
    padding: 6px 8px; margin-bottom: 4px; border-radius: 4px;
    background: #1a1a2e; font-size: 12px; cursor: pointer;
    border: 1px dashed #3a3a5c;
}
.rec-item:hover { border-color: #6a4aac; }
.rec-item .rec-char { font-weight: 700; }
.rec-item .rec-score { color: #888; float: right; }
.rec-item .rec-text { color: #999; font-size: 11px; margin-top: 2px; display: block; }
.rec-item .rec-actions { margin-top: 4px; }
.rec-item .rec-actions button {
    font-size: 11px; padding: 2px 8px; border: none; border-radius: 3px; cursor: pointer; margin-right: 4px;
}
.rec-accept { background: #2a5a2a; color: #fff; }
.rec-accept:hover { background: #3a7a3a; }
.rec-dismiss { background: #5a2a2a; color: #fff; }
.rec-dismiss:hover { background: #7a3a3a; }

/* Quality score card */
.quality-card {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    background: #1a1a2e; border: 1px solid #444; border-radius: 8px;
    padding: 24px; z-index: 1000; min-width: 350px; display: none;
}
.quality-card h3 { margin-bottom: 12px; }
.quality-card .close-btn {
    position: absolute; top: 8px; right: 12px; cursor: pointer; color: #888;
    font-size: 18px; background: none; border: none;
}
.quality-card .score { font-size: 36px; font-weight: 700; text-align: center; margin: 12px 0; }
.quality-card .score.excellent { color: #4caf50; }
.quality-card .score.good { color: #8bc34a; }
.quality-card .score.acceptable { color: #ffc107; }
.quality-card .score.poor { color: #f44336; }
.quality-card .details { font-size: 13px; color: #aaa; line-height: 1.8; }
.overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 999; display: none; }

/* Face analysis popup */
.face-popup {
    position: absolute; background: #1a1a2e; border: 1px solid #444;
    border-radius: 6px; padding: 8px; z-index: 100; font-size: 12px; min-width: 180px;
    display: none;
}
.face-popup .face-match { padding: 4px 0; cursor: pointer; }
.face-popup .face-match:hover { color: #aaf; }
.face-popup .face-match .sim { color: #888; float: right; }

/* Keyboard shortcuts help */
.shortcuts { position: fixed; bottom: 8px; right: 8px; font-size: 10px; color: #444; }
</style>
</head>
<body>

<!-- Top Bar -->
<div class="topbar">
    <select id="episodeSelect" onchange="loadEpisode(this.value)">
        <option value="">Select Episode...</option>
    </select>
    <button onclick="refreshRecommendations()">Refresh Recs</button>
    <span class="status" id="statusText">Ready</span>
</div>

<!-- Main Layout -->
<div class="main">
    <!-- Video Player -->
    <div class="video-panel" id="videoPanel">
        <video id="videoPlayer" preload="metadata"></video>
        <div class="video-controls">
            <button onclick="videoPlayer.paused ? videoPlayer.play() : videoPlayer.pause()" title="Space">Play/Pause</button>
            <button onclick="seekRelative(-5)" title="Left">-5s</button>
            <button onclick="seekRelative(-1)">-1s</button>
            <button onclick="seekRelative(1)">+1s</button>
            <button onclick="seekRelative(5)" title="Right">+5s</button>
            <span class="time-display" id="timeDisplay">0:00 / 0:00</span>
            <span style="width:16px"></span>
            <button onclick="markStart()" id="markStartBtn" title="[">Mark Start</button>
            <button onclick="markEnd()" id="markEndBtn" title="]">Mark End</button>
            <span class="selection-display" id="selectionDisplay"></span>
            <button onclick="analyzeFrame()" title="A">Analyze</button>
            <button onclick="createTagFromSelection()" title="Enter">Tag</button>
        </div>
    </div>

    <!-- Character Panel -->
    <div class="char-panel" id="charPanel">
        <h3>Characters</h3>
        <div id="recentChars" class="recent-chars"></div>
        <input type="text" id="charSearch" class="char-search" placeholder="Search..." oninput="renderCharacterPanel()">
        <div id="charList"></div>
        <div class="char-actions">
            <button onclick="extractAllSamples()">Extract All Audio</button>
            <button class="clone-btn" onclick="evaluateClone()">Clone Voice</button>
        </div>
    </div>
</div>

<!-- Subtitle Strip -->
<div class="sub-strip-header">
    <span>Subtitles</span>
    <span id="subInfo"></span>
</div>
<div class="sub-strip" id="subStrip">
    <div class="sub-strip-inner" id="subStripInner"></div>
</div>

<!-- Bottom Panel -->
<div class="bottom-panel">
    <div class="tags-panel">
        <h3>Tags (<span id="tagCount">0</span>)</h3>
        <div id="speakerGroups"></div>
    </div>
    <div class="recs-panel">
        <h3>Recommendations</h3>
        <div id="recsList"></div>
    </div>
</div>

<!-- Face Analysis Popup -->
<div class="face-popup" id="facePopup"></div>

<!-- Quality Score Card -->
<div class="overlay" id="qualityOverlay" onclick="closeQualityCard()"></div>
<div class="quality-card" id="qualityCard">
    <button class="close-btn" onclick="closeQualityCard()">&times;</button>
    <h3>Voice Clone Quality</h3>
    <div id="qualityContent"></div>
</div>

<!-- Keyboard shortcuts hint -->
<div class="shortcuts">
    [ Mark Start &nbsp; ] Mark End &nbsp; Enter Tag &nbsp; A Analyze &nbsp; Space Play/Pause &nbsp; Arrows Seek
</div>

<script>
// ============================================================
// STATE
// ============================================================
let currentEpisode = null;
let subtitles = [];
let tags = [];
let characterTotals = {};
let characters = {};
let selectedCharacter = null;
let selectionStart = null;
let selectionEnd = null;
let recommendations = [];
let recentCharacters = [];  // last 10 used characters
const MAX_RECENT = 10;

const videoPlayer = document.getElementById('videoPlayer');

// Character colors — deterministic from name hash
function charColor(name) {
    if (!name) return '#555';
    let h = 0;
    for (let i = 0; i < name.length; i++) h = name.charCodeAt(i) + ((h << 5) - h);
    h = ((h % 360) + 360) % 360;
    return `hsl(${h}, 65%, 45%)`;
}

function charColorBg(name) {
    if (!name) return '#252525';
    let h = 0;
    for (let i = 0; i < name.length; i++) h = name.charCodeAt(i) + ((h << 5) - h);
    h = ((h % 360) + 360) % 360;
    return `hsl(${h}, 40%, 20%)`;
}

// ============================================================
// INITIALIZATION
// ============================================================
async function init() {
    // Load episodes
    const eps = await fetchJson('/api/episodes');
    const sel = document.getElementById('episodeSelect');
    eps.forEach(ep => {
        const opt = document.createElement('option');
        opt.value = ep.episode_id;
        opt.textContent = `${ep.episode_id} (${formatTime(ep.duration_sec)})`;
        if (ep.has_scan) opt.textContent += ` [${ep.face_count} faces]`;
        sel.appendChild(opt);
    });

    // Load characters
    await loadCharacters();

    // Auto-load first episode
    if (eps.length > 0) {
        sel.value = eps[0].episode_id;
        loadEpisode(eps[0].episode_id);
    }
}

async function loadCharacters() {
    characters = await fetchJson('/api/characters');
    renderCharacterPanel();
}

async function loadEpisode(epId) {
    if (!epId) return;
    currentEpisode = epId;
    setStatus(`Loading ${epId}...`);

    // Load video
    videoPlayer.src = `/video/${epId}`;

    // Load subtitles
    subtitles = await fetchJson(`/api/episode/${epId}/subtitles`);
    renderSubtitleStrip();
    document.getElementById('subInfo').textContent = `${subtitles.length} subtitles`;

    // Load tags
    await refreshTags();

    // Load recommendations
    await refreshRecommendations();

    setStatus(`Loaded ${epId}`);
}

// ============================================================
// VIDEO CONTROLS
// ============================================================
function seekRelative(sec) {
    videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime + sec);
}

videoPlayer.addEventListener('timeupdate', () => {
    const cur = videoPlayer.currentTime;
    const dur = videoPlayer.duration || 0;
    document.getElementById('timeDisplay').textContent = `${formatTime(cur)} / ${formatTime(dur)}`;

    // Highlight current subtitle
    document.querySelectorAll('.sub-block.current').forEach(el => el.classList.remove('current'));
    for (let i = 0; i < subtitles.length; i++) {
        if (cur >= subtitles[i].start && cur <= subtitles[i].end) {
            const el = document.getElementById(`sub_${i}`);
            if (el) {
                el.classList.add('current');
                // Auto-scroll subtitle strip
                el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
            }
            break;
        }
    }

    hideFacePopup();
});

// ============================================================
// MARK START / END SELECTION
// ============================================================
function markStart() {
    selectionStart = videoPlayer.currentTime;
    updateSelectionDisplay();
    document.getElementById('markStartBtn').classList.add('active');
}

function markEnd() {
    selectionEnd = videoPlayer.currentTime;
    if (selectionStart !== null && selectionEnd < selectionStart) {
        [selectionStart, selectionEnd] = [selectionEnd, selectionStart];
    }
    updateSelectionDisplay();
    document.getElementById('markEndBtn').classList.add('active');
}

function updateSelectionDisplay() {
    const el = document.getElementById('selectionDisplay');
    if (selectionStart !== null && selectionEnd !== null) {
        const dur = selectionEnd - selectionStart;
        el.textContent = `Selected: ${formatTime(selectionStart)} - ${formatTime(selectionEnd)} (${dur.toFixed(1)}s)`;
    } else if (selectionStart !== null) {
        el.textContent = `Start: ${formatTime(selectionStart)} (press ] for end)`;
    } else {
        el.textContent = '';
    }
}

function clearSelection() {
    selectionStart = null;
    selectionEnd = null;
    updateSelectionDisplay();
    document.getElementById('markStartBtn').classList.remove('active');
    document.getElementById('markEndBtn').classList.remove('active');
}

// ============================================================
// TAG MANAGEMENT
// ============================================================
async function createTagFromSelection() {
    if (!currentEpisode || selectionStart === null || selectionEnd === null) {
        setStatus('Select a time range first ([ and ])');
        return;
    }
    if (!selectedCharacter) {
        setStatus('Select a character from the panel first');
        return;
    }

    // Find subtitle text in the selection range
    let subText = '';
    for (const sub of subtitles) {
        if (sub.start >= selectionStart && sub.end <= selectionEnd) {
            subText += sub.text + ' ';
        }
    }

    const tag = await fetchJson(`/api/episode/${currentEpisode}/tags`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            character: selectedCharacter,
            start_sec: selectionStart,
            end_sec: selectionEnd,
            subtitle_text: subText.trim(),
        }),
    });

    addToRecent(selectedCharacter);
    clearSelection();
    await refreshTags();
    await loadCharacters();
    await refreshRecommendations();
    setStatus(`Tagged ${selectedCharacter}: ${tag.duration_sec}s`);
}

async function deleteTag(tagId) {
    await fetchJson(`/api/episode/${currentEpisode}/tags/${tagId}`, { method: 'DELETE' });
    await refreshTags();
    await loadCharacters();
    setStatus('Tag deleted');
}

async function refreshTags() {
    if (!currentEpisode) return;
    const data = await fetchJson(`/api/episode/${currentEpisode}/tags`);
    tags = data.tags || [];
    characterTotals = data.character_totals || {};
    renderTagTable();
    renderSubtitleStrip();
}

// Find English subtitle text overlapping a time range
function getEnglishText(startSec, endSec) {
    const parts = [];
    for (const sub of subtitles) {
        // Subtitle overlaps if it starts before tag ends and ends after tag starts
        if (sub.start < endSec && sub.end > startSec) {
            parts.push(sub.text);
        }
    }
    return parts.join(' | ') || '';
}

function renderTagTable() {
    const container = document.getElementById('speakerGroups');
    document.getElementById('tagCount').textContent = tags.length;
    container.innerHTML = '';

    // Group tags by speaker (character field)
    const groups = {};
    const groupOrder = [];
    for (const tag of tags) {
        const spk = tag.character || 'Unknown';
        if (!groups[spk]) {
            groups[spk] = [];
            groupOrder.push(spk);
        }
        groups[spk].push(tag);
    }

    // Sort groups by total duration descending
    groupOrder.sort((a, b) => {
        const durA = groups[a].reduce((s, t) => s + (t.duration_sec || 0), 0);
        const durB = groups[b].reduce((s, t) => s + (t.duration_sec || 0), 0);
        return durB - durA;
    });

    for (const spk of groupOrder) {
        const spkTags = groups[spk];
        const totalDur = spkTags.reduce((s, t) => s + (t.duration_sec || 0), 0).toFixed(1);

        const group = document.createElement('div');
        group.className = 'spk-group';

        // Header row
        const header = document.createElement('div');
        header.className = 'spk-header';
        header.innerHTML = `
            <span class="spk-name" style="color:${charColor(spk)}">${spk}</span>
            <span class="spk-stats">${spkTags.length} segments, ${totalDur}s</span>
            <span class="spk-rec" id="spkRec_${spk.replace(/[^a-zA-Z0-9]/g, '_')}" title="VLM recommendation"></span>
            <button class="spk-assign" onclick="event.stopPropagation(); assignSpeaker('${escapeJs(spk)}')" title="Assign a character to all segments">Assign</button>
            <button class="spk-del-all" onclick="event.stopPropagation(); deleteAllSpeaker('${escapeJs(spk)}')" title="Delete all segments for this speaker">Del All</button>
        `;
        // Toggle collapse on header click
        header.addEventListener('click', () => {
            const segs = group.querySelector('.spk-segments');
            segs.classList.toggle('open');
        });
        group.appendChild(header);

        // Segments list (collapsed by default)
        const segsDiv = document.createElement('div');
        segsDiv.className = 'spk-segments';
        for (const tag of spkTags) {
            const seg = document.createElement('div');
            seg.className = 'spk-seg';
            seg.innerHTML = `
                <span class="seg-time">${formatTime(tag.start_sec)} - ${formatTime(tag.end_sec)}</span>
                <span class="seg-dur">${tag.duration_sec}s</span>
                <span class="seg-text">${getEnglishText(tag.start_sec, tag.end_sec)}</span>
                <span class="seg-del" onclick="event.stopPropagation(); deleteTag('${tag.tag_id}')" title="Delete">&times;</span>
            `;
            seg.addEventListener('click', () => {
                videoPlayer.currentTime = tag.start_sec;
                videoPlayer.play();
            });
            segsDiv.appendChild(seg);
        }
        group.appendChild(segsDiv);
        container.appendChild(group);

        // Fire off VLM recommendation for this speaker
        fetchSpeakerRec(spk);
    }
}

async function fetchSpeakerRec(speaker) {
    if (!currentEpisode) return;
    const el = document.getElementById('spkRec_' + speaker.replace(/[^a-zA-Z0-9]/g, '_'));
    if (!el) return;
    el.textContent = '...';
    const result = await fetchJson(`/api/episode/${currentEpisode}/speaker_recommendation`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ speaker }),
    });
    if (result.character) {
        const cands = (result.candidates || []).map(c => `${c.name}: ${c.score}s`).join(', ');
        el.innerHTML = `<span title="${cands}" style="cursor:help">Likely: <b style="color:${charColor(result.character)}">${result.character}</b> (${(result.confidence * 100).toFixed(0)}%)</span>`;
    } else {
        el.textContent = result.reason || '';
    }
}

async function assignSpeaker(speaker) {
    if (!currentEpisode) return;
    const name = prompt(`Assign all "${speaker}" segments to which character?`);
    if (!name || name.trim() === '') return;
    setStatus(`Renaming ${speaker} -> ${name.trim()}...`);
    const result = await fetchJson(`/api/episode/${currentEpisode}/rename_speaker`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ old_name: speaker, new_name: name.trim() }),
    });
    addToRecent(name.trim());
    await refreshTags();
    await loadCharacters();
    setStatus(`Renamed ${result.renamed} tags: ${speaker} -> ${name.trim()}`);
}

async function deleteAllSpeaker(speaker) {
    if (!currentEpisode) return;
    if (!confirm(`Delete ALL segments for "${speaker}"?`)) return;
    setStatus(`Deleting all ${speaker} tags...`);
    const result = await fetchJson(`/api/episode/${currentEpisode}/delete_speaker`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ speaker }),
    });
    await refreshTags();
    await loadCharacters();
    setStatus(`Deleted ${result.deleted} tags for ${speaker}`);
}

// ============================================================
// SUBTITLE STRIP
// ============================================================
function renderSubtitleStrip() {
    const container = document.getElementById('subStripInner');
    container.innerHTML = '';

    // Build tag lookup: find which subs are tagged
    const tagLookup = {};
    for (const tag of tags) {
        for (let i = 0; i < subtitles.length; i++) {
            const sub = subtitles[i];
            if (sub.start >= tag.start_sec - 0.5 && sub.end <= tag.end_sec + 0.5) {
                tagLookup[i] = tag;
            }
        }
    }

    // Build recommendation lookup
    const recLookup = {};
    for (const rec of recommendations) {
        for (let i = 0; i < subtitles.length; i++) {
            const sub = subtitles[i];
            if (Math.abs(sub.start - rec.start_sec) < 0.5 && Math.abs(sub.end - rec.end_sec) < 0.5) {
                recLookup[i] = rec;
            }
        }
    }

    for (let i = 0; i < subtitles.length; i++) {
        const sub = subtitles[i];
        const block = document.createElement('div');
        block.className = 'sub-block';
        block.id = `sub_${i}`;

        const tag = tagLookup[i];
        const rec = recLookup[i];

        if (tag) {
            block.classList.add('tagged');
            block.style.background = charColorBg(tag.character);
            block.style.borderColor = charColor(tag.character);
            block.innerHTML = `
                <span class="sub-time">${formatTime(sub.start)}</span>
                <span class="sub-char" style="color:${charColor(tag.character)}">${tag.character}</span>
                <span class="sub-text">${sub.text}</span>
            `;
        } else if (rec) {
            block.classList.add('suggested');
            block.style.borderColor = charColor(rec.character);
            block.style.background = charColorBg(rec.character);
            block.innerHTML = `
                <span class="sub-time">${formatTime(sub.start)}</span>
                <span class="sub-char" style="color:${charColor(rec.character)}">${rec.character}? (${(rec.score*100).toFixed(0)}%)</span>
                <span class="sub-text">${sub.text}</span>
            `;
        } else {
            block.classList.add('untagged');
            block.innerHTML = `
                <span class="sub-time">${formatTime(sub.start)}</span>
                <span class="sub-text">${sub.text}</span>
            `;
        }

        block.addEventListener('click', () => {
            videoPlayer.currentTime = sub.start;
            videoPlayer.play();
        });

        // Right-click: if tagged -> delete tag, if untagged -> quick-tag
        block.addEventListener('contextmenu', ((tag, sub) => (e) => {
            e.preventDefault();
            if (tag) {
                deleteTag(tag.tag_id);
            } else if (selectedCharacter) {
                quickTag(sub.start, sub.end, sub.text);
            }
        })(tag, sub));

        container.appendChild(block);
    }
}

async function quickTag(start, end, text) {
    if (!selectedCharacter || !currentEpisode) return;
    await fetchJson(`/api/episode/${currentEpisode}/tags`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            character: selectedCharacter,
            start_sec: start,
            end_sec: end,
            subtitle_text: text,
        }),
    });
    addToRecent(selectedCharacter);
    await refreshTags();
    await loadCharacters();
    setStatus(`Quick-tagged ${selectedCharacter}`);
}

// ============================================================
// CHARACTER PANEL
// ============================================================
function addToRecent(name) {
    if (!name) return;
    recentCharacters = recentCharacters.filter(n => n !== name);
    recentCharacters.unshift(name);
    if (recentCharacters.length > MAX_RECENT) recentCharacters.pop();
    renderRecentChars();
}

function renderRecentChars() {
    const container = document.getElementById('recentChars');
    container.innerHTML = '';
    if (recentCharacters.length === 0) {
        container.innerHTML = '<span style="font-size:10px;color:#555">Recent picks appear here</span>';
        return;
    }
    for (const name of recentCharacters) {
        const chip = document.createElement('span');
        chip.className = 'recent-chip';
        chip.textContent = name;
        chip.style.color = charColor(name);
        chip.style.borderColor = charColor(name) + '88';
        if (name === selectedCharacter) chip.classList.add('selected');
        chip.addEventListener('click', () => {
            selectedCharacter = name;
            renderCharacterPanel();
            renderRecentChars();
        });
        container.appendChild(chip);
    }
}

function renderCharacterPanel() {
    const list = document.getElementById('charList');
    list.innerHTML = '';

    const query = (document.getElementById('charSearch').value || '').toLowerCase();

    // Sort: characters with tags first, then alphabetical
    const names = Object.keys(characters).sort((a, b) => {
        const aSec = characters[a].total_seconds;
        const bSec = characters[b].total_seconds;
        if (aSec > 0 && bSec === 0) return -1;
        if (aSec === 0 && bSec > 0) return 1;
        return a.localeCompare(b);
    }).filter(n => !query || n.toLowerCase().includes(query));

    for (const name of names) {
        const ch = characters[name];
        const item = document.createElement('div');
        item.className = 'char-item';
        if (name === selectedCharacter) item.classList.add('selected');
        if (ch.status === 'ready') item.classList.add('ready');

        const pct = Math.min(100, (ch.total_seconds / 90) * 100);
        const fillColor = ch.status === 'ready' ? '#4caf50' : charColor(name);

        // Reference thumbnail
        const thumbSrc = ch.reference_images.length > 0
            ? `/reference/${name}/${ch.reference_images[0]}`
            : '';

        item.innerHTML = `
            ${thumbSrc ? `<img class="char-thumb" src="${thumbSrc}" alt="${name}">` : '<div class="char-thumb"></div>'}
            <div class="char-info">
                <div class="char-name">${name}</div>
                <div class="char-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width:${pct}%;background:${fillColor}"></div>
                    </div>
                    <span class="progress-text">${ch.total_seconds}s${ch.status === 'ready' ? ' OK' : ''}</span>
                </div>
            </div>
        `;

        item.addEventListener('click', () => {
            selectedCharacter = name;
            addToRecent(name);
            renderCharacterPanel();
        });

        list.appendChild(item);
    }

    renderRecentChars();
}

// ============================================================
// FACE SCANNING
// ============================================================
// ============================================================
// FRAME ANALYSIS — VLM scene lookup
// ============================================================
async function analyzeFrame() {
    if (!currentEpisode) return;
    const time = videoPlayer.currentTime;
    setStatus('Looking up scene...');

    const result = await fetchJson(`/api/episode/${currentEpisode}/analyze_frame`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ time_sec: time }),
    });

    showScenePopup(result);
    const charNames = (result.characters || []).map(c => c.name).join(', ');
    setStatus(result.characters.length > 0 ? `Characters: ${charNames}` : 'No characters in this scene');
}

function showScenePopup(result) {
    const popup = document.getElementById('facePopup');
    const videoRect = videoPlayer.getBoundingClientRect();
    popup.style.left = (videoRect.right - 260) + 'px';
    popup.style.top = (videoRect.top + 10) + 'px';
    popup.style.display = 'block';

    let html = '';
    if (result.scene) {
        html += `<div style="font-size:11px;color:#888;margin-bottom:6px">Scene ${result.scene.scene_number}</div>`;
        html += `<div style="font-size:11px;color:#aaa;margin-bottom:8px">${result.scene.narrative}</div>`;
    }
    html += '<div style="font-weight:700;margin-bottom:4px">Characters:</div>';
    if (result.characters && result.characters.length > 0) {
        for (const c of result.characters) {
            const escapedName = c.name.replace(/'/g, "\\'");
            html += `<div class="face-match" onclick="selectCharacterByName('${escapedName}');hideFacePopup()">
                <span style="color:${charColor(c.name)}">${c.name}</span>
                <span class="sim" style="font-size:10px">${c.action}</span>
            </div>`;
        }
    } else {
        html += '<div style="color:#888">No characters identified</div>';
    }
    popup.innerHTML = html;
}

function hideFacePopup() {
    document.getElementById('facePopup').style.display = 'none';
}

function selectCharacterByName(name) {
    selectedCharacter = name;
    renderCharacterPanel();
}

// ============================================================
// RECOMMENDATIONS
// ============================================================
async function refreshRecommendations() {
    if (!currentEpisode) return;
    recommendations = await fetchJson(`/api/episode/${currentEpisode}/recommendations`);
    renderRecommendations();
    renderSubtitleStrip();
}

function renderRecommendations() {
    const container = document.getElementById('recsList');
    container.innerHTML = '';

    if (recommendations.length === 0) {
        container.innerHTML = '<div style="color:#666;font-size:12px">No recommendations. Click "Refresh Recs" after loading an episode.</div>';
        return;
    }

    for (const rec of recommendations.slice(0, 50)) {
        const item = document.createElement('div');
        item.className = 'rec-item';
        item.innerHTML = `
            <span class="rec-char" style="color:${charColor(rec.character)}">${rec.character}</span>
            <span class="rec-score">${(rec.score*100).toFixed(0)}%</span>
            <span class="rec-text">${formatTime(rec.start_sec)} ${rec.subtitle_text.substring(0, 60)}</span>
            <div class="rec-actions">
                <button class="rec-accept" onclick="acceptRec(${rec.start_sec}, ${rec.end_sec}, '${escapeJs(rec.subtitle_text)}', '${escapeJs(rec.character)}')">Accept</button>
                <button class="rec-dismiss">Dismiss</button>
            </div>
        `;
        item.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') return;
            videoPlayer.currentTime = rec.start_sec;
        });
        container.appendChild(item);
    }
}

async function acceptRec(start, end, text, character) {
    if (!currentEpisode) return;
    await fetchJson(`/api/episode/${currentEpisode}/tags`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            character: character,
            start_sec: start,
            end_sec: end,
            subtitle_text: text,
        }),
    });
    addToRecent(character);
    await refreshTags();
    await loadCharacters();
    await refreshRecommendations();
    setStatus(`Accepted: ${character}`);
}

// ============================================================
// AUDIO EXTRACTION
// ============================================================
async function extractSample(tagId) {
    if (!currentEpisode) return;
    setStatus('Extracting audio...');
    const result = await fetchJson(`/api/episode/${currentEpisode}/extract_sample`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ tag_id: tagId, use_demucs: false }),
    });
    setStatus(result.path ? 'Audio extracted!' : 'Extraction failed');
}

async function extractAllSamples() {
    if (!currentEpisode || tags.length === 0) return;
    setStatus(`Extracting audio for ${tags.length} tags...`);
    for (const tag of tags) {
        await fetchJson(`/api/episode/${currentEpisode}/extract_sample`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ tag_id: tag.tag_id, use_demucs: false }),
        });
    }
    setStatus('All audio extracted!');
}

// ============================================================
// VOICE CLONE EVALUATION
// ============================================================
async function evaluateClone() {
    if (!selectedCharacter) {
        setStatus('Select a character first');
        return;
    }
    setStatus(`Evaluating voice clone for ${selectedCharacter}...`);
    document.getElementById('qualityOverlay').style.display = 'block';
    document.getElementById('qualityCard').style.display = 'block';
    document.getElementById('qualityContent').innerHTML = '<div style="text-align:center;color:#888">Synthesizing and scoring...</div>';

    const result = await fetchJson(`/api/character/${selectedCharacter}/evaluate`, { method: 'POST' });
    renderQualityCard(result);
    setStatus('Evaluation complete');
}

function renderQualityCard(result) {
    const content = document.getElementById('qualityContent');
    if (result.status === 'no_samples' || result.status === 'insufficient') {
        content.innerHTML = `
            <div class="score poor">${result.status === 'no_samples' ? 'N/A' : 'Insufficient'}</div>
            <div class="details">${result.message}</div>
        `;
        return;
    }
    if (result.status === 'error') {
        content.innerHTML = `<div class="score poor">Error</div><div class="details">${result.message}</div>`;
        return;
    }

    const q = result.quality || 'unknown';
    content.innerHTML = `
        <div class="score ${q}">${result.speaker_similarity ? (result.speaker_similarity * 100).toFixed(0) + '%' : '?'}</div>
        <div style="text-align:center;margin-bottom:12px;color:${q === 'excellent' ? '#4caf50' : q === 'good' ? '#8bc34a' : q === 'acceptable' ? '#ffc107' : '#f44336'}">${q.toUpperCase()}</div>
        <div class="details">
            <b>Character:</b> ${result.character}<br>
            <b>Samples:</b> ${result.total_samples} clips, ${result.total_duration}s total<br>
            <b>References used:</b> ${result.ref_clips_used} clips, ${result.ref_duration}s<br>
            <b>Speaker similarity:</b> ${result.speaker_similarity}<br>
            <b>Status:</b> ${result.status}<br>
            ${result.synth_path ? `<br><audio controls src="/evaluation/${result.character}/test_synthesis.wav" style="width:100%"></audio>` : ''}
        </div>
    `;
}

function closeQualityCard() {
    document.getElementById('qualityOverlay').style.display = 'none';
    document.getElementById('qualityCard').style.display = 'none';
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================
document.addEventListener('keydown', (e) => {
    // Don't capture if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
        case ' ':
            e.preventDefault();
            videoPlayer.paused ? videoPlayer.play() : videoPlayer.pause();
            break;
        case 'ArrowLeft':
            e.preventDefault();
            seekRelative(e.shiftKey ? -1 : -5);
            break;
        case 'ArrowRight':
            e.preventDefault();
            seekRelative(e.shiftKey ? 1 : 5);
            break;
        case '[':
            e.preventDefault();
            markStart();
            break;
        case ']':
            e.preventDefault();
            markEnd();
            break;
        case 'Enter':
            e.preventDefault();
            createTagFromSelection();
            break;
        case 'a':
        case 'A':
            if (!e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                analyzeFrame();
            }
            break;
        case 'Escape':
            clearSelection();
            hideFacePopup();
            closeQualityCard();
            break;
    }
});

// ============================================================
// UTILITY FUNCTIONS
// ============================================================
function formatTime(sec) {
    if (sec == null || isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function setStatus(msg) {
    document.getElementById('statusText').textContent = msg;
}

async function fetchJson(url, opts = {}) {
    try {
        const resp = await fetch(url, opts);
        return await resp.json();
    } catch (e) {
        console.error('Fetch error:', e);
        return {};
    }
}

function escapeJs(str) {
    return (str || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
}

// Start
init();
</script>
</body>
</html>
"""


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Voice Tagger")
    print("  http://localhost:7861")
    print("=" * 60)
    print()
    print("Workflow:")
    print("  1. Select an episode (VLM scene data auto-loads)")
    print("  2. Click 'Refresh Recs' for character recommendations")
    print("  3. Accept recs or select a character + right-click subtitles to quick-tag")
    print("  4. Press 'A' to analyze current frame (shows VLM characters)")
    print("  5. Click 'Extract All Audio' when done tagging")
    print("  6. Click 'Clone Voice' to test quality")
    print()

    app.run(host="0.0.0.0", port=7861, debug=False, threaded=True)
