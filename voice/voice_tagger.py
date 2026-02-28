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


@app.route("/api/episode/<episode_id>/scene_at_time", methods=["POST"])
def api_scene_at_time(episode_id):
    """Return full VLM scene data at a given timestamp."""
    body = request.get_json()
    time_sec = body.get("time_sec", 0)
    scene = backend.get_full_scene_at_time(episode_id, time_sec)
    if scene:
        return jsonify(scene)
    return jsonify({})


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
        confidence=body.get("confidence", "manual"),
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
        "total_overlap_sec": round(total, 1),
        "candidates": [{"name": n, "score": round(s, 1), "pct": round(s / total * 100, 1) if total > 0 else 0} for n, s in best],
    })


@app.route("/api/episode/<episode_id>/enriched_subtitles")
def api_enriched_subtitles(episode_id):
    enriched = backend.get_enriched_subtitles(episode_id)
    return jsonify(enriched)


@app.route("/api/episode/<episode_id>/bulk_accept", methods=["POST"])
def api_bulk_accept(episode_id):
    body = request.get_json()
    accepts = body.get("accepts", [])
    count = backend.bulk_accept_tags(episode_id, accepts)
    return jsonify({"ok": True, "accepted": count})


@app.route("/api/episode/<episode_id>/dismiss_recommendation", methods=["POST"])
def api_dismiss_recommendation(episode_id):
    body = request.get_json()
    backend.dismiss_recommendation(
        episode_id, body["start_sec"], body["end_sec"], body["character"]
    )
    return jsonify({"ok": True})


# ---- Diarization ----
@app.route("/api/episode/<episode_id>/diarization_groups")
def api_diarization_groups(episode_id):
    data = backend.get_diarization_status(episode_id)
    return jsonify(data)


@app.route("/api/episode/<episode_id>/diarization_run", methods=["POST"])
def api_diarization_run(episode_id):
    try:
        from voice.diarize_segments import diarize_episode
    except ImportError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    job_id = backend.start_job(diarize_episode, episode_id)
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/episode/<episode_id>/diarization_status")
def api_diarization_job_status(episode_id):
    job_id = request.args.get("job_id")
    if job_id:
        return jsonify(backend.get_job_status(job_id))
    data = backend.get_diarization_status(episode_id)
    return jsonify({"status": data.get("status", "not_found")})


@app.route("/api/episode/<episode_id>/diarization_accept", methods=["POST"])
def api_diarization_accept(episode_id):
    body = request.get_json()
    accepts = body.get("accepts", [])
    count = backend.bulk_accept_diarization(episode_id, accepts)
    return jsonify({"ok": True, "tagged": count})


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
<title>Voice Tagger — Triage Mode</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

/* ── Top bar ── */
.topbar {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 16px; background: #1a1a2e; border-bottom: 1px solid #333;
    flex-shrink: 0; flex-wrap: wrap;
}
.topbar select { background: #252540; color: #fff; border: 1px solid #444; padding: 5px 10px; border-radius: 4px; font-size: 13px; }
.topbar .status { color: #888; font-size: 12px; margin-left: auto; }

/* ── Progress bar ── */
.progress-row {
    display: flex; align-items: center; gap: 8px; flex: 1; min-width: 200px;
}
.progress-outer {
    flex: 1; height: 8px; background: #333; border-radius: 4px; overflow: hidden;
}
.progress-inner { height: 100%; border-radius: 4px; transition: width 0.3s, background 0.3s; }
.progress-text { font-size: 11px; color: #aaa; white-space: nowrap; }
.tier-counts { font-size: 11px; color: #888; white-space: nowrap; }
.tier-counts .t1 { color: #4caf50; } .tier-counts .t2 { color: #ffc107; } .tier-counts .t3 { color: #f44; }

/* ── Toolbar buttons ── */
.tb-btn {
    background: #3a3a5c; color: #fff; border: none; padding: 5px 12px;
    border-radius: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;
}
.tb-btn:hover { background: #4a4a7c; }
.tb-btn.active { background: #6a4aac; }
.tb-btn.accept-all { background: #2a6a2a; font-weight: 700; }
.tb-btn.accept-all:hover { background: #3a8a3a; }
.tb-btn.accept-scene { background: #2a5a3a; }
.tb-btn.accept-scene:hover { background: #3a7a4a; }
.filter-group { display: flex; gap: 2px; }
.filter-group .tb-btn { border-radius: 0; }
.filter-group .tb-btn:first-child { border-radius: 4px 0 0 4px; }
.filter-group .tb-btn:last-child { border-radius: 0 4px 4px 0; }

/* ── Main layout ── */
.main { display: flex; flex: 1; min-height: 0; overflow: hidden; }

/* ── Scene panel (toggleable) ── */
.scene-panel {
    width: 280px; background: #0d0d1a; border-right: 1px solid #333;
    overflow-y: auto; padding: 10px 12px; font-size: 12px;
    flex-shrink: 0; display: none;
}
.scene-panel.visible { display: block; }
.scene-panel h3 { font-size: 11px; color: #666; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
.scene-badge { padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
.scene-badge.story { background: #2a3a5a; color: #8af; }
.scene-badge.flashback { background: #5a3a2a; color: #fa8; }
.scene-badge.opening { background: #2a5a3a; color: #8fa; }
.scene-badge.ending { background: #5a2a5a; color: #f8f; }
.scene-badge.recap { background: #5a5a2a; color: #cc8; }
.scene-badge.transition { background: #3a3a3a; color: #aaa; }
.scene-badge.eyecatch { background: #5a3a5a; color: #c8f; }
.scene-narrative { color: #aaa; font-style: italic; line-height: 1.4; margin: 6px 0; border-left: 2px solid #444; padding-left: 8px; font-size: 11px; }
.scene-char { padding: 3px 6px; margin-bottom: 2px; border-radius: 3px; background: #1a1a2a; font-size: 11px; }
.scene-char-name { font-weight: 700; }
.te-item { padding: 4px 8px; margin-bottom: 2px; border-radius: 3px; background: #111122; border-left: 3px solid #444; cursor: pointer; font-size: 11px; }
.te-item:hover { background: #1a1a2e; }
.te-item.te-active { background: #1a1a3e; }

/* ── Video panel ── */
.video-panel {
    position: relative; background: #000; display: flex; flex-direction: column;
    align-items: center; justify-content: center; min-width: 280px;
    width: 30%; flex-shrink: 0;
}
.video-panel video { max-width: 100%; max-height: calc(100% - 44px); }
.video-controls {
    width: 100%; background: rgba(0,0,0,0.85); padding: 6px 10px;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap; flex-shrink: 0;
}
.video-controls button {
    background: #3a3a5c; color: #fff; border: none; padding: 3px 8px;
    border-radius: 3px; cursor: pointer; font-size: 11px;
}
.video-controls button:hover { background: #5a5a8c; }
.video-controls .time-display { color: #aaa; font-size: 11px; font-family: monospace; }

/* ── Subtitle list (main work area) ── */
.sub-list-panel {
    flex: 1; overflow-y: auto; background: #111; min-width: 300px;
}
.scene-group { border-bottom: 1px solid #222; }
.scene-group-header {
    display: flex; align-items: center; gap: 8px; padding: 6px 12px;
    background: #1a1a1a; cursor: pointer; user-select: none;
    position: sticky; top: 0; z-index: 5; flex-wrap: wrap;
}
.scene-group-header:hover { background: #222; }
.sg-title { font-weight: 700; font-size: 12px; }
.sg-badge { font-size: 10px; padding: 1px 6px; border-radius: 3px; font-weight: 600; }
.sg-location { color: #8af; font-size: 11px; }
.sg-chars { display: flex; gap: 4px; flex-wrap: wrap; }
.sg-char-chip { font-size: 10px; padding: 1px 6px; border-radius: 10px; border: 1px solid #444; }
.sg-progress { margin-left: auto; font-size: 11px; color: #888; white-space: nowrap; }
.sg-done { color: #4caf50; }
.scene-group-body { }
.scene-group.collapsed .scene-group-body { display: none; }

/* ── Subtitle row ── */
.sub-row {
    display: flex; align-items: flex-start; gap: 8px; padding: 5px 12px;
    border-left: 3px solid #333; cursor: pointer; transition: background 0.1s;
    min-height: 32px;
}
.sub-row:hover { background: #1a1a1a; }
.sub-row.focused { background: #1a1a2e; outline: 1px solid #6a4aac; outline-offset: -1px; }
.sub-row.tagged { border-left-color: #4caf50; }
.sub-row.tier1 { border-left-color: #ffc107; }
.sub-row.tier2 { border-left-color: #ff9800; }
.sub-row.tier3 { border-left-color: #666; }
.sub-row .sr-time { font-family: monospace; font-size: 11px; color: #888; min-width: 42px; flex-shrink: 0; padding-top: 1px; }
.sub-row .sr-text { flex: 1; font-size: 12px; line-height: 1.4; min-width: 0; }
.sub-row .sr-signal { display: flex; align-items: center; gap: 6px; flex-shrink: 0; flex-wrap: wrap; }
.sr-suggestion {
    display: flex; align-items: center; gap: 4px; font-size: 11px;
    padding: 2px 8px; border-radius: 3px; background: #1a1a2e;
}
.sr-char-name { font-weight: 700; }
.sr-score { color: #888; font-size: 10px; }
.sr-reason { color: #666; font-size: 10px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sr-actions { display: flex; gap: 3px; }
.sr-btn {
    font-size: 10px; padding: 2px 7px; border: none; border-radius: 3px;
    cursor: pointer; font-weight: 600;
}
.sr-accept { background: #2a5a2a; color: #cfc; }
.sr-accept:hover { background: #3a7a3a; }
.sr-change { background: #3a3a5c; color: #ccf; }
.sr-change:hover { background: #4a4a7c; }
.sr-tag-badge {
    display: flex; align-items: center; gap: 4px; font-size: 11px;
    padding: 2px 8px; border-radius: 3px; background: #1a3a1a;
}
.sr-tag-badge .sr-del { color: #c44; cursor: pointer; font-size: 13px; }
.sr-tag-badge .sr-del:hover { color: #f66; }
.sr-unresolved-chips { display: flex; gap: 3px; flex-wrap: wrap; }
.sr-chip {
    font-size: 10px; padding: 2px 8px; border-radius: 10px;
    border: 1px solid #444; cursor: pointer; font-weight: 600; white-space: nowrap;
}
.sr-chip:hover { background: #2a2a4e; border-color: #6a4aac; }

/* ── Inline character picker ── */
.char-picker {
    padding: 6px 12px 8px 62px; background: #151520; border-left: 3px solid #6a4aac;
    display: none;
}
.char-picker.open { display: block; }
.cp-section-label { font-size: 10px; color: #555; margin: 4px 0 2px; text-transform: uppercase; }
.cp-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
.cp-chip {
    font-size: 11px; padding: 3px 10px; border-radius: 12px;
    border: 1px solid #444; cursor: pointer; font-weight: 600;
}
.cp-chip:hover { background: #2a2a4e; border-color: #6a4aac; }
.cp-search {
    width: 200px; padding: 4px 8px; background: #222; color: #eee;
    border: 1px solid #444; border-radius: 4px; font-size: 12px; outline: none;
    margin-top: 4px;
}
.cp-search:focus { border-color: #6a4aac; }
.cp-all-chars { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; max-height: 80px; overflow-y: auto; }

/* ── Character panel (toggleable) ── */
.char-panel {
    width: 260px; background: #1a1a1a; border-left: 1px solid #333;
    overflow-y: auto; padding: 8px; flex-shrink: 0; display: none;
}
.char-panel.visible { display: block; }
.char-panel h3 { font-size: 11px; color: #666; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
.char-item {
    padding: 5px 8px; margin-bottom: 3px; border-radius: 4px;
    cursor: pointer; display: flex; align-items: center; gap: 6px;
    border: 1px solid transparent; font-size: 12px;
}
.char-item:hover { background: #252525; }
.char-item .char-thumb { width: 28px; height: 28px; border-radius: 3px; object-fit: cover; background: #333; }
.char-item .char-info { flex: 1; min-width: 0; }
.char-item .char-name { font-size: 12px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.char-item .progress-bar { height: 3px; background: #333; border-radius: 2px; overflow: hidden; margin-top: 2px; }
.char-item .progress-fill { height: 100%; border-radius: 2px; }
.char-item .progress-text { font-size: 9px; color: #888; }

/* ── Footer ── */
.footer {
    display: flex; align-items: center; gap: 16px;
    padding: 4px 16px; background: #1a1a2e; border-top: 1px solid #333;
    font-size: 11px; color: #888; flex-shrink: 0;
}
.footer .shortcuts { color: #555; margin-left: auto; }
.footer kbd { background: #333; padding: 1px 5px; border-radius: 2px; font-size: 10px; color: #aaa; }

/* ── Quality card (modal) ── */
.overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 999; display: none; }
.quality-card {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    background: #1a1a2e; border: 1px solid #444; border-radius: 8px;
    padding: 24px; z-index: 1000; min-width: 350px; display: none;
}
.quality-card .close-btn { position: absolute; top: 8px; right: 12px; cursor: pointer; color: #888; font-size: 18px; background: none; border: none; }

/* ── Diarization modal ── */
.diar-modal {
    position: fixed; inset: 20px; background: #0f0f1a;
    border: 1px solid #444; border-radius: 8px; z-index: 1001;
    display: none; flex-direction: column; overflow: hidden;
}
.diar-modal.open { display: flex; }
.diar-header {
    display: flex; align-items: center; gap: 10px; padding: 10px 16px;
    background: #1a1a2e; border-bottom: 1px solid #333; flex-shrink: 0; flex-wrap: wrap;
}
.diar-header h3 { margin: 0; font-size: 14px; }
.diar-header .dh-status { color: #888; font-size: 12px; flex: 1; min-width: 0; }
.diar-body { flex: 1; overflow-y: auto; padding: 12px 16px; }
.diar-scene-section { margin-bottom: 12px; }
.diar-scene-label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; padding: 0 4px; }
.diar-group {
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    margin-bottom: 3px; background: #111; border-radius: 4px;
    border-left: 4px solid #444; flex-wrap: wrap; min-height: 46px;
}
.diar-group.tier-clean  { border-left-color: #4caf50; }
.diar-group.tier-likely { border-left-color: #ffc107; }
.diar-group.tier-impure { border-left-color: #f44336; }
.diar-group.dg-accepted { opacity: 0.45; }
.dg-meta { min-width: 130px; flex-shrink: 0; }
.dg-seg   { font-weight: 700; font-size: 12px; color: #ccc; }
.dg-spk   { font-size: 11px; color: #666; font-family: monospace; }
.dg-stats { font-size: 11px; color: #888; }
.dg-suggestion { flex: 1; display: flex; align-items: center; gap: 6px; min-width: 0; flex-wrap: wrap; }
.dg-char { font-weight: 700; font-size: 13px; }
.dg-conf { font-size: 11px; color: #888; }
.dg-breakdown { font-size: 10px; color: #555; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 200px; }
.dg-tier-badge {
    font-size: 10px; padding: 1px 6px; border-radius: 3px;
    font-weight: 700; text-transform: uppercase; flex-shrink: 0;
}
.dg-tier-badge.clean  { background: #1a3a1a; color: #4caf50; }
.dg-tier-badge.likely { background: #3a3000; color: #ffc107; }
.dg-tier-badge.impure { background: #3a0a0a; color: #f66; }
.dg-actions { display: flex; gap: 4px; flex-shrink: 0; }
.dg-btn { font-size: 11px; padding: 3px 9px; border: none; border-radius: 3px; cursor: pointer; font-weight: 600; }
.dg-btn.dg-seek    { background: #252540; color: #ccf; }
.dg-btn.dg-seek:hover { background: #3a3a6a; }
.dg-btn.dg-confirm { background: #2a5a2a; color: #cfc; }
.dg-btn.dg-confirm:hover { background: #3a7a3a; }
.dg-btn.dg-change  { background: #3a3a5c; color: #ccf; }
.dg-btn.dg-change:hover { background: #4a4a7c; }
.dg-btn.dg-done    { background: #1a3a1a; color: #6c6; cursor: default; }
.diar-run-section { text-align: center; padding: 60px 20px; }
.diar-run-section p { color: #888; margin-bottom: 20px; line-height: 1.6; }
.diar-progress-wrap { padding: 40px 20px; text-align: center; }
.diar-progress-bar { height: 8px; background: #222; border-radius: 4px; overflow: hidden; max-width: 400px; margin: 10px auto; }
.diar-progress-fill { height: 100%; background: #4caf50; border-radius: 4px; transition: width 0.5s; }
</style>
</head>
<body>

<!-- ═══ Top Bar ═══ -->
<div class="topbar">
    <select id="episodeSelect" onchange="loadEpisode(this.value)">
        <option value="">Select Episode...</option>
    </select>
    <div class="progress-row">
        <div class="progress-outer"><div class="progress-inner" id="progressBar" style="width:0;background:#f44"></div></div>
        <span class="progress-text" id="progressText">0/0</span>
    </div>
    <span class="tier-counts" id="tierCounts"></span>
    <button class="tb-btn accept-all" onclick="acceptAllTier1()" title="Ctrl+Shift+Y">Accept All Tier 1</button>
    <div class="filter-group">
        <button class="tb-btn active" id="filterAll" onclick="setFilter('all')">All</button>
        <button class="tb-btn" id="filterUntagged" onclick="setFilter('untagged')">Untagged</button>
    </div>
    <button class="tb-btn" id="btnScene" onclick="toggleScenePanel()" title="S">S Scene</button>
    <button class="tb-btn" id="btnChars" onclick="toggleCharPanel()" title="C">C Chars</button>
    <button class="tb-btn" id="btnDiarize" onclick="openDiarModal()" title="Speaker group resolution">Diarize</button>
    <span class="status" id="statusText">Ready</span>
</div>

<!-- ═══ Main Layout ═══ -->
<div class="main">
    <!-- Scene Panel (hidden by default, toggle with S) -->
    <div class="scene-panel" id="scenePanel">
        <h3>Scene Data</h3>
        <div id="sceneContent"><div style="color:#555;text-align:center;padding:20px">Scene data loads during playback</div></div>
    </div>

    <!-- Video Player -->
    <div class="video-panel" id="videoPanel">
        <video id="videoPlayer" preload="metadata"></video>
        <div class="video-controls">
            <button onclick="videoPlayer.paused ? videoPlayer.play() : videoPlayer.pause()" title="Space">Play</button>
            <button onclick="seekRelative(-5)">-5s</button>
            <button onclick="seekRelative(-1)">-1s</button>
            <button onclick="seekRelative(1)">+1s</button>
            <button onclick="seekRelative(5)">+5s</button>
            <span class="time-display" id="timeDisplay">0:00 / 0:00</span>
        </div>
    </div>

    <!-- Subtitle List (main work area) -->
    <div class="sub-list-panel" id="subListPanel">
        <div id="subListContent"></div>
    </div>

    <!-- Character Panel (hidden by default, toggle with C) -->
    <div class="char-panel" id="charPanel">
        <h3>Characters</h3>
        <input type="text" class="cp-search" id="charPanelSearch" placeholder="Search..." oninput="renderCharPanel()">
        <div id="charPanelList"></div>
    </div>
</div>

<!-- ═══ Footer ═══ -->
<div class="footer">
    <span id="footerTagged">0 tagged</span>
    <span id="footerRemaining">0 remaining</span>
    <span id="footerSpeed"></span>
    <span id="footerEta"></span>
    <span class="shortcuts">
        <kbd>Y</kbd> Accept <kbd>N</kbd> Change <kbd>J</kbd>/<kbd>K</kbd> Nav
        <kbd>1</kbd>-<kbd>9</kbd> Quick-assign <kbd>Shift+Y</kbd> Scene T1
        <kbd>Space</kbd> Play <kbd>F</kbd> Filter <kbd>S</kbd> Scene <kbd>C</kbd> Chars
    </span>
</div>

<!-- Quality Card Modal -->
<div class="overlay" id="qualityOverlay" onclick="closeQualityCard()"></div>
<div class="quality-card" id="qualityCard">
    <button class="close-btn" onclick="closeQualityCard()">&times;</button>
    <h3>Voice Clone Quality</h3>
    <div id="qualityContent"></div>
</div>

<!-- Diarization Modal -->
<div class="overlay" id="diarOverlay" onclick="closeDiarModal()"></div>
<div class="diar-modal" id="diarModal">
    <div class="diar-header">
        <h3>Speaker Group Resolution</h3>
        <span class="dh-status" id="diarHeaderStatus"></span>
        <button class="tb-btn accept-all" onclick="acceptAllCleanGroups()">Accept All Clean</button>
        <button class="tb-btn" onclick="closeDiarModal()" style="background:#333">Done</button>
    </div>
    <div class="diar-body" id="diarBody">
        <div style="text-align:center;padding:60px;color:#555">Select an episode first.</div>
    </div>
</div>

<script>
// ============================================================
// STATE
// ============================================================
let currentEpisode = null;
let enrichedSubs = [];       // from /enriched_subtitles
let characters = {};         // from /characters
let allCharNames = [];       // sorted character name list
let focusedIndex = -1;       // index in enrichedSubs of focused row
let pickerOpenIndex = -1;    // index where character picker is open
let filterMode = 'all';      // 'all' or 'untagged'
let tagStartTime = null;     // for speed tracking
let tagCount = 0;            // tags made this session

// Scene panel state
let currentSceneNumber = null;
let currentSceneRange = null;
let currentSceneData = null;
let lastSceneUpdateTime = 0;

// Recent characters for picker
let recentCharacters = [];
const MAX_RECENT = 8;

const videoPlayer = document.getElementById('videoPlayer');

// ── Color helpers ──
function charColor(name) {
    if (!name) return '#555';
    let h = 0;
    for (let i = 0; i < name.length; i++) h = name.charCodeAt(i) + ((h << 5) - h);
    h = ((h % 360) + 360) % 360;
    return `hsl(${h}, 65%, 45%)`;
}
function charColorLight(name) {
    if (!name) return '#888';
    let h = 0;
    for (let i = 0; i < name.length; i++) h = name.charCodeAt(i) + ((h << 5) - h);
    h = ((h % 360) + 360) % 360;
    return `hsl(${h}, 60%, 70%)`;
}

// ============================================================
// INIT
// ============================================================
async function init() {
    const eps = await fetchJson('/api/episodes');
    const sel = document.getElementById('episodeSelect');
    eps.forEach(ep => {
        const opt = document.createElement('option');
        opt.value = ep.episode_id;
        opt.textContent = `${ep.episode_id} (${formatTime(ep.duration_sec)})`;
        sel.appendChild(opt);
    });
    characters = await fetchJson('/api/characters');
    allCharNames = Object.keys(characters).sort();

    // Restore panel visibility from localStorage
    if (localStorage.getItem('vt_scene_panel') === '1') toggleScenePanel();
    if (localStorage.getItem('vt_char_panel') === '1') toggleCharPanel();

    if (eps.length > 0) {
        sel.value = eps[0].episode_id;
        loadEpisode(eps[0].episode_id);
    }
}

async function loadEpisode(epId) {
    if (!epId) return;
    currentEpisode = epId;
    currentSceneNumber = null;
    currentSceneRange = null;
    currentSceneData = null;
    focusedIndex = -1;
    pickerOpenIndex = -1;
    tagStartTime = null;
    tagCount = 0;
    setStatus(`Loading ${epId}...`);

    videoPlayer.src = `/video/${epId}`;
    await refreshEnriched();
    setStatus(`Loaded ${epId} — ${enrichedSubs.length} subtitles`);
}

async function refreshEnriched() {
    if (!currentEpisode) return;
    enrichedSubs = await fetchJson(`/api/episode/${currentEpisode}/enriched_subtitles`);
    renderSubList();
    updateProgress();
    renderCharPanel();
}

// ============================================================
// PROGRESS TRACKING
// ============================================================
function updateProgress() {
    const total = enrichedSubs.length;
    const tagged = enrichedSubs.filter(s => s.tag).length;
    const remaining = total - tagged;
    const pct = total > 0 ? Math.round(tagged / total * 100) : 0;

    // Tier counts (among untagged)
    const untagged = enrichedSubs.filter(s => !s.tag);
    const t1 = untagged.filter(s => s.tier === 1).length;
    const t2 = untagged.filter(s => s.tier === 2).length;
    const t3 = untagged.filter(s => s.tier === 3).length;

    // Progress bar color
    const bar = document.getElementById('progressBar');
    bar.style.width = pct + '%';
    bar.style.background = pct < 30 ? '#f44' : pct < 70 ? '#ffc107' : '#4caf50';

    document.getElementById('progressText').textContent = `${tagged}/${total} (${pct}%)`;
    document.getElementById('tierCounts').innerHTML =
        `<span class="t1">T1:${t1}</span> <span class="t2">T2:${t2}</span> <span class="t3">T3:${t3}</span>`;

    // Footer
    document.getElementById('footerTagged').textContent = `${tagged} tagged`;
    document.getElementById('footerRemaining').textContent = `${remaining} remaining`;

    // Speed + ETA
    if (tagStartTime && tagCount > 0) {
        const elapsed = (Date.now() - tagStartTime) / 1000 / 60; // minutes
        const speed = tagCount / elapsed;
        document.getElementById('footerSpeed').textContent = `~${speed.toFixed(1)} tags/min`;
        if (remaining > 0 && speed > 0) {
            const eta = remaining / speed;
            document.getElementById('footerEta').textContent = `ETA ~${eta.toFixed(0)} min`;
        } else {
            document.getElementById('footerEta').textContent = '';
        }
    }
}

// ============================================================
// RENDER SUBTITLE LIST (grouped by scene)
// ============================================================
function renderSubList() {
    const container = document.getElementById('subListContent');
    container.innerHTML = '';

    // Group by scene_number
    const groups = [];
    let curGroup = null;
    for (let i = 0; i < enrichedSubs.length; i++) {
        const sub = enrichedSubs[i];
        const sn = sub.scene_number;
        if (!curGroup || curGroup.scene_number !== sn) {
            curGroup = {
                scene_number: sn,
                scene_type: sub.scene_type,
                scene_location: sub.scene_location,
                scene_chars: sub.scene_chars || [],
                items: [],
            };
            groups.push(curGroup);
        }
        curGroup.items.push({ ...sub, _idx: i });
    }

    for (const group of groups) {
        const totalInGroup = group.items.length;
        const taggedInGroup = group.items.filter(s => s.tag).length;
        const allDone = taggedInGroup === totalInGroup;

        // Filter: in 'untagged' mode, skip fully-tagged scenes
        const visibleItems = filterMode === 'untagged'
            ? group.items.filter(s => !s.tag)
            : group.items;
        if (filterMode === 'untagged' && visibleItems.length === 0) continue;

        const gDiv = document.createElement('div');
        gDiv.className = 'scene-group' + (allDone ? ' collapsed' : '');
        gDiv.dataset.sceneNumber = group.scene_number || '';

        // Header
        const header = document.createElement('div');
        header.className = 'scene-group-header';
        let headerHtml = `<span class="sg-title">Scene ${group.scene_number || '?'}</span>`;
        if (group.scene_type) {
            const badgeColor = {'story':'#2a3a5a','flashback':'#5a3a2a','opening':'#2a5a3a','ending':'#5a2a5a','recap':'#5a5a2a','eyecatch':'#5a3a5a'}[group.scene_type] || '#333';
            headerHtml += `<span class="sg-badge" style="background:${badgeColor}">${escapeHtml(group.scene_type)}</span>`;
        }
        if (group.scene_location) headerHtml += `<span class="sg-location">${escapeHtml(group.scene_location)}</span>`;

        // Character chips
        headerHtml += '<span class="sg-chars">';
        for (const c of (group.scene_chars || []).slice(0, 5)) {
            headerHtml += `<span class="sg-char-chip" style="color:${charColorLight(c.name)};border-color:${charColor(c.name)}44">${escapeHtml(c.name)}</span>`;
        }
        headerHtml += '</span>';

        // Per-scene progress
        headerHtml += `<span class="sg-progress${allDone ? ' sg-done' : ''}">${taggedInGroup}/${totalInGroup}</span>`;
        header.innerHTML = headerHtml;
        header.addEventListener('click', () => gDiv.classList.toggle('collapsed'));
        gDiv.appendChild(header);

        // Body (subtitle rows)
        const body = document.createElement('div');
        body.className = 'scene-group-body';
        for (const item of visibleItems) {
            body.appendChild(createSubRow(item));
        }
        gDiv.appendChild(body);
        container.appendChild(gDiv);
    }
}

function createSubRow(item) {
    const row = document.createElement('div');
    row.className = 'sub-row';
    row.dataset.idx = item._idx;
    const idx = item._idx;

    if (item.tag) {
        row.classList.add('tagged');
    } else if (item.tier === 1) {
        row.classList.add('tier1');
    } else if (item.tier === 2) {
        row.classList.add('tier2');
    } else {
        row.classList.add('tier3');
    }
    if (idx === focusedIndex) row.classList.add('focused');

    // Time
    let html = `<span class="sr-time">${formatTime(item.start)}</span>`;
    // Text
    html += `<span class="sr-text">${escapeHtml(item.text)}</span>`;
    // Signal area
    html += '<span class="sr-signal">';

    if (item.tag) {
        // Tagged — show tag badge
        const ch = item.tag.character;
        const confLabel = item.tag.confidence === 'auto_accepted' ? 'A' : item.tag.confidence === 'accepted_rec' ? 'R' : 'M';
        html += `<span class="sr-tag-badge" style="border-left:3px solid ${charColor(ch)}">`;
        html += `<span class="sr-char-name" style="color:${charColorLight(ch)}">${escapeHtml(ch)}</span>`;
        html += `<span style="font-size:9px;color:#888">${confLabel}</span>`;
        html += `<span class="sr-del" onclick="event.stopPropagation();deleteTagByIndex(${idx})">&times;</span>`;
        html += '</span>';
    } else if (item.suggested_character) {
        // Has suggestion — show accept/change
        const ch = item.suggested_character;
        html += `<span class="sr-suggestion" style="border-left:3px solid ${charColor(ch)}">`;
        html += `<span class="sr-char-name" style="color:${charColorLight(ch)}">${escapeHtml(ch)}</span>`;
        html += `<span class="sr-score">${Math.round(item.suggestion_score * 100)}%</span>`;
        html += '</span>';
        html += '<span class="sr-actions">';
        html += `<button class="sr-btn sr-accept" onclick="event.stopPropagation();acceptSuggestion(${idx})" title="Y">Accept</button>`;
        html += `<button class="sr-btn sr-change" onclick="event.stopPropagation();openPicker(${idx})" title="N">Chg</button>`;
        html += '</span>';
    } else {
        // No suggestion — show scene char chips for quick assign
        const chars = item.scene_chars || [];
        if (chars.length > 0) {
            html += '<span class="sr-unresolved-chips">';
            for (let ci = 0; ci < chars.length && ci < 5; ci++) {
                const c = chars[ci];
                html += `<span class="sr-chip" style="color:${charColorLight(c.name)};border-color:${charColor(c.name)}44" onclick="event.stopPropagation();assignCharacter(${idx},'${escapeJs(c.name)}')">${escapeHtml(c.name)}</span>`;
            }
            html += `<span class="sr-chip" style="color:#888" onclick="event.stopPropagation();openPicker(${idx})">...</span>`;
            html += '</span>';
        } else {
            html += `<button class="sr-btn sr-change" onclick="event.stopPropagation();openPicker(${idx})">Pick</button>`;
        }
    }

    html += '</span>';
    row.innerHTML = html;

    // Click row -> seek + focus
    row.addEventListener('click', () => {
        setFocus(idx);
        videoPlayer.currentTime = item.start;
    });

    // Wrap: row + optional picker
    const wrapper = document.createDocumentFragment();
    wrapper.appendChild(row);

    // Picker div (hidden by default)
    const picker = document.createElement('div');
    picker.className = 'char-picker' + (pickerOpenIndex === idx ? ' open' : '');
    picker.id = `picker_${idx}`;
    picker.addEventListener('click', (e) => e.stopPropagation());
    if (pickerOpenIndex === idx) {
        picker.innerHTML = buildPickerHtml(item);
    }
    wrapper.appendChild(picker);

    return wrapper;
}

function buildPickerHtml(item) {
    const sceneChars = (item.scene_chars || []).map(c => c.name);
    let html = '';

    // Scene characters
    if (sceneChars.length > 0) {
        html += '<div class="cp-section-label">Scene characters</div><div class="cp-chips">';
        for (const name of sceneChars) {
            html += `<span class="cp-chip" style="color:${charColorLight(name)};border-color:${charColor(name)}" onclick="assignCharacter(${item._idx},'${escapeJs(name)}')">${escapeHtml(name)}</span>`;
        }
        html += '</div>';
    }

    // Recent
    const recents = recentCharacters.filter(n => !sceneChars.includes(n));
    if (recents.length > 0) {
        html += '<div class="cp-section-label">Recent</div><div class="cp-chips">';
        for (const name of recents) {
            html += `<span class="cp-chip" style="color:${charColorLight(name)};border-color:${charColor(name)}44" onclick="assignCharacter(${item._idx},'${escapeJs(name)}')">${escapeHtml(name)}</span>`;
        }
        html += '</div>';
    }

    // Search + all chars
    html += `<input type="text" class="cp-search" id="pickerSearch" placeholder="Search all characters..." oninput="filterPickerChars(${item._idx})" onkeydown="if(event.key==='Escape'){closePicker();event.stopPropagation();}">`;
    html += `<div class="cp-all-chars" id="pickerAllChars">`;
    const exclude = new Set([...sceneChars, ...recents]);
    for (const name of allCharNames) {
        if (exclude.has(name)) continue;
        html += `<span class="cp-chip" style="color:${charColorLight(name)};border-color:${charColor(name)}44" onclick="assignCharacter(${item._idx},'${escapeJs(name)}')">${escapeHtml(name)}</span>`;
    }
    html += '</div>';
    return html;
}

function filterPickerChars(idx) {
    const search = document.getElementById('pickerSearch');
    if (!search) return;
    const q = search.value.toLowerCase();
    const container = document.getElementById('pickerAllChars');
    if (!container) return;
    for (const chip of container.children) {
        chip.style.display = chip.textContent.toLowerCase().includes(q) ? '' : 'none';
    }
}

// ============================================================
// ACTIONS
// ============================================================
async function acceptSuggestion(idx) {
    const sub = enrichedSubs[idx];
    if (!sub || !sub.suggested_character || sub.tag) return;
    await createTag(sub.start, sub.end, sub.text, sub.suggested_character, 'accepted_rec');
}

async function assignCharacter(idx, character) {
    const sub = enrichedSubs[idx];
    if (!sub) return;
    closePicker();
    if (sub.tag) {
        // Re-assign: delete old, create new
        await fetchJson(`/api/episode/${currentEpisode}/tags/${sub.tag.tag_id}`, { method: 'DELETE' });
    }
    await createTag(sub.start, sub.end, sub.text, character, 'manual');
}

async function createTag(start, end, text, character, confidence) {
    if (!currentEpisode) return;
    if (!tagStartTime) tagStartTime = Date.now();
    tagCount++;
    addToRecent(character);

    await fetchJson(`/api/episode/${currentEpisode}/tags`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            character: character,
            start_sec: start,
            end_sec: end,
            subtitle_text: text,
            confidence: confidence,
        }),
    });
    await refreshEnriched();
    setStatus(`Tagged: ${character}`);
}

async function deleteTagByIndex(idx) {
    const sub = enrichedSubs[idx];
    if (!sub || !sub.tag) return;
    await fetchJson(`/api/episode/${currentEpisode}/tags/${sub.tag.tag_id}`, { method: 'DELETE' });
    await refreshEnriched();
    setStatus('Tag removed');
}

async function acceptAllTier1() {
    const t1 = enrichedSubs.filter(s => !s.tag && s.tier === 1 && s.suggested_character);
    if (t1.length === 0) { setStatus('No Tier 1 to accept'); return; }
    setStatus(`Accepting ${t1.length} Tier 1 tags...`);
    if (!tagStartTime) tagStartTime = Date.now();

    const accepts = t1.map(s => ({
        start_sec: s.start, end_sec: s.end,
        character: s.suggested_character, subtitle_text: s.text,
    }));

    await fetchJson(`/api/episode/${currentEpisode}/bulk_accept`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ accepts }),
    });
    tagCount += t1.length;
    await refreshEnriched();
    setStatus(`Accepted ${t1.length} Tier 1 tags!`);
}

async function acceptSceneTier1() {
    // Accept Tier 1 in the scene of the currently focused row
    const sub = enrichedSubs[focusedIndex];
    if (!sub) return;
    const sn = sub.scene_number;
    const t1 = enrichedSubs.filter(s => !s.tag && s.tier === 1 && s.suggested_character && s.scene_number === sn);
    if (t1.length === 0) { setStatus('No Tier 1 in this scene'); return; }

    if (!tagStartTime) tagStartTime = Date.now();
    const accepts = t1.map(s => ({
        start_sec: s.start, end_sec: s.end,
        character: s.suggested_character, subtitle_text: s.text,
    }));
    await fetchJson(`/api/episode/${currentEpisode}/bulk_accept`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ accepts }),
    });
    tagCount += t1.length;
    await refreshEnriched();
    setStatus(`Accepted ${t1.length} tags in scene ${sn}`);
}

// ============================================================
// FOCUS / NAVIGATION
// ============================================================
function setFocus(idx) {
    focusedIndex = idx;
    // Update visual focus
    document.querySelectorAll('.sub-row.focused').forEach(el => el.classList.remove('focused'));
    const row = document.querySelector(`.sub-row[data-idx="${idx}"]`);
    if (row) {
        row.classList.add('focused');
        row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function focusNextUntagged(direction) {
    if (enrichedSubs.length === 0) return;
    const start = focusedIndex >= 0 ? focusedIndex : -1;
    if (direction > 0) {
        for (let i = start + 1; i < enrichedSubs.length; i++) {
            if (!enrichedSubs[i].tag) { setFocus(i); return; }
        }
    } else {
        for (let i = start - 1; i >= 0; i--) {
            if (!enrichedSubs[i].tag) { setFocus(i); return; }
        }
    }
}

function jumpScene(direction) {
    if (enrichedSubs.length === 0 || focusedIndex < 0) return;
    const curScene = enrichedSubs[focusedIndex].scene_number;
    if (direction > 0) {
        for (let i = focusedIndex + 1; i < enrichedSubs.length; i++) {
            if (enrichedSubs[i].scene_number !== curScene && !enrichedSubs[i].tag) {
                setFocus(i);
                // Expand that scene group if collapsed
                const group = document.querySelector(`.sub-row[data-idx="${i}"]`)?.closest('.scene-group');
                if (group && group.classList.contains('collapsed')) group.classList.remove('collapsed');
                return;
            }
        }
    } else {
        let prevScene = null;
        for (let i = focusedIndex - 1; i >= 0; i--) {
            if (enrichedSubs[i].scene_number !== curScene) { prevScene = enrichedSubs[i].scene_number; break; }
        }
        if (prevScene !== null) {
            for (let i = 0; i < enrichedSubs.length; i++) {
                if (enrichedSubs[i].scene_number === prevScene && !enrichedSubs[i].tag) {
                    setFocus(i);
                    const group = document.querySelector(`.sub-row[data-idx="${i}"]`)?.closest('.scene-group');
                    if (group && group.classList.contains('collapsed')) group.classList.remove('collapsed');
                    return;
                }
            }
        }
    }
}

// ============================================================
// PICKER
// ============================================================
function openPicker(idx) {
    closePicker();
    pickerOpenIndex = idx;
    const el = document.getElementById(`picker_${idx}`);
    if (el) {
        el.classList.add('open');
        const item = enrichedSubs[idx];
        item._idx = idx;
        el.innerHTML = buildPickerHtml(item);
        setTimeout(() => {
            const search = document.getElementById('pickerSearch');
            if (search) search.focus();
        }, 50);
    }
}

function closePicker() {
    if (pickerOpenIndex >= 0) {
        const el = document.getElementById(`picker_${pickerOpenIndex}`);
        if (el) { el.classList.remove('open'); el.innerHTML = ''; }
    }
    pickerOpenIndex = -1;
}

// ============================================================
// PANEL TOGGLES
// ============================================================
function toggleScenePanel() {
    const p = document.getElementById('scenePanel');
    p.classList.toggle('visible');
    document.getElementById('btnScene').classList.toggle('active', p.classList.contains('visible'));
    localStorage.setItem('vt_scene_panel', p.classList.contains('visible') ? '1' : '0');
}

function toggleCharPanel() {
    const p = document.getElementById('charPanel');
    p.classList.toggle('visible');
    document.getElementById('btnChars').classList.toggle('active', p.classList.contains('visible'));
    localStorage.setItem('vt_char_panel', p.classList.contains('visible') ? '1' : '0');
}

function setFilter(mode) {
    filterMode = mode;
    document.getElementById('filterAll').classList.toggle('active', mode === 'all');
    document.getElementById('filterUntagged').classList.toggle('active', mode === 'untagged');
    renderSubList();
}

// ============================================================
// SCENE PANEL (reused from old UI, simplified)
// ============================================================
async function updateScenePanel(timeSec) {
    if (!currentEpisode) return;
    if (!document.getElementById('scenePanel').classList.contains('visible')) return;
    if (currentSceneRange && timeSec >= currentSceneRange.start && timeSec <= currentSceneRange.end) return;

    const scene = await fetchJson(`/api/episode/${currentEpisode}/scene_at_time`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ time_sec: timeSec }),
    });
    if (!scene || !scene.scene_number) {
        currentSceneNumber = null; currentSceneRange = null; currentSceneData = null;
        document.getElementById('sceneContent').innerHTML = '<div style="color:#555;text-align:center">No scene data</div>';
        return;
    }
    currentSceneNumber = scene.scene_number;
    currentSceneRange = { start: scene.start_time_seconds, end: scene.end_time_seconds };
    currentSceneData = scene;

    let html = `<div style="display:flex;gap:6px;align-items:center;margin-bottom:6px;flex-wrap:wrap">`;
    html += `<span style="font-weight:700">Scene ${scene.scene_number}</span>`;
    html += `<span class="scene-badge ${scene.scene_type || ''}">${scene.scene_type || '?'}</span>`;
    if (scene.emotional_tone) html += `<span style="color:#888;font-size:11px">${escapeHtml(scene.emotional_tone)}</span>`;
    html += '</div>';
    html += `<div style="font-family:monospace;font-size:10px;color:#888;margin-bottom:4px">${formatTime(scene.start_time_seconds)} - ${formatTime(scene.end_time_seconds)}</div>`;
    if (scene.location) html += `<div style="color:#8af;font-size:11px;margin-bottom:4px">${escapeHtml(scene.location)}</div>`;
    if (scene.scene_narrative) html += `<div class="scene-narrative">${escapeHtml(scene.scene_narrative)}</div>`;

    for (const c of (scene.characters_present || [])) {
        const name = c.name || c.character_id || '';
        html += `<div class="scene-char"><span class="scene-char-name" style="color:${charColor(name)}">${escapeHtml(name)}</span>`;
        if (c.action) html += ` <span style="color:#888;font-size:10px">${escapeHtml(c.action)}</span>`;
        html += '</div>';
    }

    const tes = scene.text_elements || [];
    if (tes.length > 0) {
        html += `<div style="margin-top:6px;font-size:10px;color:#555;text-transform:uppercase">Dialogue (${tes.length})</div>`;
        for (const te of tes) {
            const conf = te.speaker_confidence || 0;
            const borderColor = te.speaker && te.speaker !== 'Unknown' ? charColor(te.speaker) : '#444';
            html += `<div class="te-item" style="border-left-color:${borderColor}" onclick="videoPlayer.currentTime=${te.start_time_seconds||0}">`;
            html += `<span style="font-weight:700;color:${te.speaker ? charColor(te.speaker) : '#888'};font-size:11px">${escapeHtml(te.speaker||'?')}</span>`;
            html += ` <span style="color:#666;font-size:9px">${Math.round(conf*100)}%</span>`;
            html += `<div style="color:#bbb;font-size:11px">${escapeHtml(te.content||'')}</div>`;
            html += '</div>';
        }
    }
    document.getElementById('sceneContent').innerHTML = html;
}

// ============================================================
// CHARACTER PANEL
// ============================================================
function renderCharPanel() {
    const list = document.getElementById('charPanelList');
    if (!list) return;
    list.innerHTML = '';
    const query = (document.getElementById('charPanelSearch')?.value || '').toLowerCase();

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
        const pct = Math.min(100, (ch.total_seconds / 90) * 100);
        const fillColor = ch.status === 'ready' ? '#4caf50' : charColor(name);
        const thumbSrc = ch.reference_images?.length > 0 ? `/reference/${name}/${ch.reference_images[0]}` : '';
        item.innerHTML = `
            ${thumbSrc ? `<img class="char-thumb" src="${thumbSrc}" alt="">` : '<div class="char-thumb"></div>'}
            <div class="char-info">
                <div class="char-name" style="color:${charColorLight(name)}">${escapeHtml(name)}</div>
                <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${fillColor}"></div></div>
                <span class="progress-text">${ch.total_seconds}s / ${ch.total_tags} tags</span>
            </div>`;
        list.appendChild(item);
    }
}

// ============================================================
// VIDEO
// ============================================================
function seekRelative(sec) {
    videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime + sec);
}

videoPlayer.addEventListener('timeupdate', () => {
    const cur = videoPlayer.currentTime;
    const dur = videoPlayer.duration || 0;
    document.getElementById('timeDisplay').textContent = `${formatTime(cur)} / ${formatTime(dur)}`;

    const now = Date.now();
    if (now - lastSceneUpdateTime > 1000) {
        lastSceneUpdateTime = now;
        updateScenePanel(cur);
    }
});

videoPlayer.addEventListener('seeked', () => {
    currentSceneRange = null;
    updateScenePanel(videoPlayer.currentTime);
});

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        if (e.key === 'Escape') { closePicker(); e.target.blur(); }
        return;
    }

    const focused = focusedIndex >= 0 ? enrichedSubs[focusedIndex] : null;

    switch (e.key) {
        case ' ':
            e.preventDefault();
            videoPlayer.paused ? videoPlayer.play() : videoPlayer.pause();
            break;
        case 'j': case 'ArrowDown':
            e.preventDefault();
            focusNextUntagged(1);
            if (focusedIndex >= 0) videoPlayer.currentTime = enrichedSubs[focusedIndex].start;
            break;
        case 'k': case 'ArrowUp':
            e.preventDefault();
            focusNextUntagged(-1);
            if (focusedIndex >= 0) videoPlayer.currentTime = enrichedSubs[focusedIndex].start;
            break;
        case 'y': case 'Y':
            e.preventDefault();
            if (e.ctrlKey && e.shiftKey) { acceptAllTier1(); }
            else if (e.shiftKey) { acceptSceneTier1(); }
            else if (focused && !focused.tag && focused.suggested_character) { acceptSuggestion(focusedIndex); }
            break;
        case 'n': case 'N':
            e.preventDefault();
            if (focused && !focused.tag) openPicker(focusedIndex);
            break;
        case '[':
            e.preventDefault();
            jumpScene(-1);
            break;
        case ']':
            e.preventDefault();
            jumpScene(1);
            break;
        case 'f': case 'F':
            e.preventDefault();
            setFilter(filterMode === 'all' ? 'untagged' : 'all');
            break;
        case 's':
            if (!e.ctrlKey) { e.preventDefault(); toggleScenePanel(); }
            break;
        case 'c':
            if (!e.ctrlKey) { e.preventDefault(); toggleCharPanel(); }
            break;
        case 'Escape':
            closePicker();
            break;
        default:
            // 1-9: quick assign scene characters
            if (e.key >= '1' && e.key <= '9' && focused && !focused.tag) {
                const ci = parseInt(e.key) - 1;
                const chars = focused.scene_chars || [];
                if (ci < chars.length) {
                    e.preventDefault();
                    assignCharacter(focusedIndex, chars[ci].name);
                }
            }
            break;
    }
});

// ============================================================
// RECENT CHARACTERS
// ============================================================
function addToRecent(name) {
    if (!name) return;
    recentCharacters = recentCharacters.filter(n => n !== name);
    recentCharacters.unshift(name);
    if (recentCharacters.length > MAX_RECENT) recentCharacters.pop();
}

// ============================================================
// QUALITY CARD (kept from old UI)
// ============================================================
function closeQualityCard() {
    document.getElementById('qualityOverlay').style.display = 'none';
    document.getElementById('qualityCard').style.display = 'none';
}

// ============================================================
// UTILITY
// ============================================================
function formatTime(sec) {
    if (sec == null || isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function setStatus(msg) { document.getElementById('statusText').textContent = msg; }

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escapeJs(str) {
    return (str || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
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

// ============================================================
// DIARIZATION UI
// ============================================================
let diarGroups = [];
let diarAccepted = new Set();   // cluster_ids confirmed this session
let diarJobId = null;
let diarPollTimer = null;

async function openDiarModal() {
    document.getElementById('diarOverlay').style.display = 'block';
    document.getElementById('diarModal').classList.add('open');
    diarAccepted = new Set();
    await loadDiarGroups();
}

function closeDiarModal() {
    document.getElementById('diarOverlay').style.display = 'none';
    document.getElementById('diarModal').classList.remove('open');
    if (diarPollTimer) { clearInterval(diarPollTimer); diarPollTimer = null; }
    if (diarAccepted.size > 0) {
        refreshEnriched();
        setStatus(`Applied diarization: ${diarAccepted.size} groups confirmed`);
    }
}

async function loadDiarGroups() {
    if (!currentEpisode) {
        document.getElementById('diarBody').innerHTML =
            '<div style="text-align:center;padding:60px;color:#555">Select an episode first.</div>';
        return;
    }
    document.getElementById('diarBody').innerHTML =
        '<div style="text-align:center;padding:40px;color:#888">Loading...</div>';

    const data = await fetchJson(`/api/episode/${currentEpisode}/diarization_groups`);
    if (data.status === 'not_found') {
        renderDiarRunPrompt();
    } else if (data.status === 'done') {
        diarGroups = data.groups || [];
        renderDiarGroups();
    } else {
        document.getElementById('diarBody').innerHTML =
            `<div style="text-align:center;padding:40px;color:#888">Status: ${escapeHtml(data.status || 'unknown')}</div>`;
    }
}

function renderDiarRunPrompt() {
    document.getElementById('diarHeaderStatus').textContent = 'No diarization data';
    document.getElementById('diarBody').innerHTML = `
        <div class="diar-run-section">
            <p>
                Run VLM-guided per-segment speaker diarization.<br>
                Uses pyannote 3.1 + Demucs for vocal isolation.<br>
                <strong>One-time preprocessing: ~5-8 min on GPU. Results cached.</strong>
            </p>
            <p style="font-size:12px;color:#666">
                Requires <code>HF_TOKEN</code> env variable and pyannote model access.<br>
                See <code>voice/DIARIZATION_PLAN.md</code> for setup details.
            </p>
            <button class="tb-btn accept-all" onclick="triggerDiarization()"
                    style="padding:10px 28px;font-size:14px">
                Run Diarization
            </button>
        </div>`;
}

async function triggerDiarization() {
    if (!currentEpisode) return;
    document.getElementById('diarBody').innerHTML =
        '<div class="diar-progress-wrap">Starting diarization pipeline...</div>';

    const resp = await fetchJson(`/api/episode/${currentEpisode}/diarization_run`, { method: 'POST' });
    if (!resp.ok || !resp.job_id) {
        document.getElementById('diarBody').innerHTML =
            `<div class="diar-progress-wrap" style="color:#f66">Failed to start: ${escapeHtml(resp.error || 'unknown error')}</div>`;
        return;
    }
    diarJobId = resp.job_id;
    diarPollTimer = setInterval(pollDiarStatus, 2000);
}

async function pollDiarStatus() {
    if (!diarJobId || !currentEpisode) return;
    const status = await fetchJson(
        `/api/episode/${currentEpisode}/diarization_status?job_id=${diarJobId}`
    );
    const pct = status.progress || 0;
    const msg = status.message || 'Running...';

    if (status.status === 'done') {
        clearInterval(diarPollTimer); diarPollTimer = null; diarJobId = null;
        await loadDiarGroups();
    } else if (status.status === 'error') {
        clearInterval(diarPollTimer); diarPollTimer = null;
        document.getElementById('diarBody').innerHTML =
            `<div class="diar-progress-wrap" style="color:#f66">
                <strong>Error:</strong> ${escapeHtml(msg)}
             </div>`;
    } else {
        document.getElementById('diarBody').innerHTML = `
            <div class="diar-progress-wrap">
                <div style="color:#aaa;margin-bottom:12px">${escapeHtml(msg)}</div>
                <div class="diar-progress-bar">
                    <div class="diar-progress-fill" style="width:${pct}%"></div>
                </div>
                <div style="margin-top:8px;color:#555;font-size:12px">${pct}%</div>
            </div>`;
    }
}

function renderDiarGroups() {
    const clean   = diarGroups.filter(g => g.tier === 'clean').length;
    const likely  = diarGroups.filter(g => g.tier === 'likely').length;
    const impure  = diarGroups.filter(g => g.tier === 'impure').length;
    document.getElementById('diarHeaderStatus').innerHTML =
        `${diarGroups.length} groups &mdash; `
        + `<span style="color:#4caf50">clean:${clean}</span> `
        + `<span style="color:#ffc107">likely:${likely}</span> `
        + `<span style="color:#f66">impure:${impure}</span>`;

    const body = document.getElementById('diarBody');
    body.innerHTML = '';

    if (diarGroups.length === 0) {
        body.innerHTML = '<div style="text-align:center;padding:60px;color:#555">No speaker groups found.</div>';
        return;
    }

    // Group by scene number
    const byScene = {};
    for (const g of diarGroups) {
        const sn = g.scene_number;
        if (!byScene[sn]) byScene[sn] = [];
        byScene[sn].push(g);
    }

    for (const sn of Object.keys(byScene).sort((a, b) => +a - +b)) {
        const section = document.createElement('div');
        section.className = 'diar-scene-section';
        section.innerHTML = `<div class="diar-scene-label">Scene ${sn}</div>`;
        for (const group of byScene[sn]) {
            section.appendChild(createDiarRow(group));
        }
        body.appendChild(section);
    }
}

function createDiarRow(group) {
    const div = document.createElement('div');
    div.className = `diar-group tier-${group.tier}${diarAccepted.has(group.cluster_id) ? ' dg-accepted' : ''}`;
    div.id = `dg_${group.cluster_id}`;

    const char = group.character;
    const pct  = Math.round(group.confidence * 100);
    const breakdown = Object.entries(group.vlm_breakdown || {})
        .map(([c, n]) => `${c}\u00d7${n}`).join('  ');

    div.innerHTML = `
        <div class="dg-meta">
            <div class="dg-seg">${escapeHtml(group.segment_id)}</div>
            <div class="dg-spk">${escapeHtml(group.speaker_label)}</div>
            <div class="dg-stats">${group.subtitle_count} lines &middot; ${group.total_duration_sec.toFixed(1)}s</div>
        </div>
        <div class="dg-suggestion">
            ${char
                ? `<span class="dg-char" style="color:${charColorLight(char)}">${escapeHtml(char)}</span>
                   <span class="dg-conf">${pct}%</span>`
                : `<span style="color:#555">No VLM match</span>`}
            <span class="dg-breakdown" title="${escapeHtml(breakdown)}">${escapeHtml(breakdown)}</span>
            <span class="dg-tier-badge ${group.tier}">${group.tier}</span>
        </div>
        <div class="dg-actions" id="dgact_${group.cluster_id}">
            <button class="dg-btn dg-seek"
                    onclick="seekDiarSample(${group.sample_start_sec})">&#9654;</button>
            ${char
                ? `<button class="dg-btn dg-confirm" id="dgbtn_${group.cluster_id}"
                           data-cluster="${escapeHtml(group.cluster_id)}"
                           data-char="${escapeHtml(char)}"
                           onclick="confirmDiarGroup(this)">
                       ${escapeHtml(char)} &#10003;
                   </button>`
                : ''}
            <button class="dg-btn dg-change"
                    data-cluster="${escapeHtml(group.cluster_id)}"
                    onclick="changeDiarChar(this)">&#8230;</button>
        </div>`;

    return div;
}

function seekDiarSample(startSec) {
    videoPlayer.currentTime = startSec;
    videoPlayer.play();
}

async function confirmDiarGroup(btn) {
    const clusterId = btn.dataset.cluster;
    const character = btn.dataset.char;
    if (!clusterId || !character) return;
    btn.textContent = '...';
    btn.disabled = true;

    const resp = await fetchJson(
        `/api/episode/${currentEpisode}/diarization_accept`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ accepts: [{ cluster_id: clusterId, character }] }),
        }
    );
    diarAccepted.add(clusterId);
    const row = document.getElementById(`dg_${clusterId}`);
    if (row) row.classList.add('dg-accepted');
    btn.textContent = '\u2713 Done';
    btn.className = 'dg-btn dg-done';
    btn.disabled = true;
}

function changeDiarChar(changeBtn) {
    const clusterId = changeBtn.dataset.cluster;
    const char = prompt('Enter character name:', '');
    if (!char || !char.trim()) return;
    const character = char.trim();

    // Update the confirm button and trigger confirmation
    const confirmBtn = document.getElementById(`dgbtn_${clusterId}`);
    if (confirmBtn) {
        confirmBtn.dataset.char = character;
        confirmBtn.textContent = character + ' \u2713';
        confirmDiarGroup(confirmBtn);
    } else {
        // No confirm button existed (no VLM match) — create the tag directly
        fetchJson(`/api/episode/${currentEpisode}/diarization_accept`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ accepts: [{ cluster_id: clusterId, character }] }),
        }).then(() => {
            diarAccepted.add(clusterId);
            const row = document.getElementById(`dg_${clusterId}`);
            if (row) row.classList.add('dg-accepted');
        });
    }
}

async function acceptAllCleanGroups() {
    const pending = diarGroups.filter(
        g => g.tier === 'clean' && g.character && !diarAccepted.has(g.cluster_id)
    );
    if (pending.length === 0) { setStatus('No clean groups remaining to accept.'); return; }

    const accepts = pending.map(g => ({ cluster_id: g.cluster_id, character: g.character }));
    const resp = await fetchJson(
        `/api/episode/${currentEpisode}/diarization_accept`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ accepts }),
        }
    );
    for (const g of pending) {
        diarAccepted.add(g.cluster_id);
        const row = document.getElementById(`dg_${g.cluster_id}`);
        if (row) row.classList.add('dg-accepted');
        const btn = document.getElementById(`dgbtn_${g.cluster_id}`);
        if (btn) { btn.textContent = '\u2713 Done'; btn.className = 'dg-btn dg-done'; btn.disabled = true; }
    }
    setStatus(`Diarization: tagged ${resp.tagged || 0} lines from ${pending.length} clean groups`);
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
    print("Triage workflow:")
    print("  1. Select an episode (enriched subtitles load automatically)")
    print("  2. Click 'Accept All Tier 1' to tag ~40-60% instantly")
    print("  3. Press J/K to navigate untagged, Y to accept, 1-9 to quick-assign")
    print("  4. Press N to open character picker for tricky lines")
    print("  5. Press F to filter to untagged only, S/C to toggle panels")
    print()

    app.run(host="0.0.0.0", port=7861, debug=False, threaded=True)
