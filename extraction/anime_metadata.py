"""
anime_metadata.py — Main pipeline orchestrator for anime episode metadata extraction.

Extracts structured metadata from anime episodes, designed to mirror the manga
extraction schema so a third alignment stage can correlate them. Captures both
what is in the episode and the production decisions made (additions, expansions,
reinterpretations of manga source material).

Pipeline stages:
  [1/9] Extract subtitles from embedded tracks (ASS/SRT — no Whisper API needed)
  [2/9] Run TransNetV2 for shot boundary detection
  [3/9] Extract per-shot features (color, motion, audio — lightweight OpenCV only)
  [4/9] Detect scenes (group shots into scenes via multi-signal scoring)
  [5/9] VLM analysis per scene (narrative, characters, location, emotion)
  [6/9] Content origin classification per shot
  [7/9] Speaker attribution and dialogue index construction
  [8/9] Build per-scene JSONs, shot index, character registry
  [9/9] Build episode manifest

Usage:
    python anime_metadata.py <video_path> --episode-id 0001
                             [--chapter-manifest path/to/chapter_manifest.json]
                             [--resume]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
import re
import argparse
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path

import logging
import numpy as np
import cv2

try:
    from wiki.one_piece_wiki_api import (
        get_episode_characters, get_episode_summary, get_episode_infobox,
        get_arc_for_episode, normalize_arc_id,
    )
    HAS_WIKI_API = True
except ImportError:
    HAS_WIKI_API = False

    def normalize_arc_id(arc_name):
        """Fallback when wiki API is unavailable."""
        if not arc_name:
            return ""
        name = re.sub(r"\s+(Arc|Saga)$", "", arc_name.strip(), flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

# ---------------------------------------------------------------------------
# API cost tracking
# ---------------------------------------------------------------------------

API_COSTS = {
    "gpt4o_input_tokens": 0,
    "gpt4o_output_tokens": 0,
    "vlm_calls": 0,
}

# Pricing (as of 2025 — adjust if OpenAI changes rates)
# GPT-4o: $2.50 / 1M input tokens, $10.00 / 1M output tokens
PRICING = {
    "gpt4o_input_per_1m": 2.50,
    "gpt4o_output_per_1m": 10.00,
}


def get_cost_summary():
    """Calculate and return a cost summary string."""
    gpt4o_input_cost = (API_COSTS["gpt4o_input_tokens"] / 1_000_000) * PRICING["gpt4o_input_per_1m"]
    gpt4o_output_cost = (API_COSTS["gpt4o_output_tokens"] / 1_000_000) * PRICING["gpt4o_output_per_1m"]
    total = gpt4o_input_cost + gpt4o_output_cost

    return (
        f"\n{'=' * 70}\n"
        f"API COST BREAKDOWN\n"
        f"{'=' * 70}\n"
        f"GPT-4o Vision API:\n"
        f"  Calls:  {API_COSTS['vlm_calls']}\n"
        f"  Input:  {API_COSTS['gpt4o_input_tokens']:,} tokens (${gpt4o_input_cost:.4f})\n"
        f"  Output: {API_COSTS['gpt4o_output_tokens']:,} tokens (${gpt4o_output_cost:.4f})\n"
        f"\n"
        f"TOTAL COST: ${total:.4f}\n"
        f"{'=' * 70}"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Paths
    "output_base": "output/anime_metadata",
    "keyframes_subdir": "keyframes",

    # API
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "openai_model": "gpt-4o",
    "openai_max_tokens": 6144,
    "openai_temperature": 0.2,

    # VLM
    "vlm_max_retries": 3,
    "vlm_retry_delay": 5,
    "vlm_calls_per_minute": 100,  # self-imposed rate limit (OpenAI allows 500+ RPM at Tier 1)
    "vlm_short_scene_threshold": 15.0,   # scenes < 15s: send all shots' keyframes
    "vlm_medium_scene_interval": 5.0,    # scenes 15-60s: 1 keyframe per ~5s of duration
    "vlm_long_scene_threshold": 60.0,    # scenes > 60s: switch to longer interval
    "vlm_long_scene_interval": 8.0,      # scenes > 60s: 1 keyframe per ~8s of duration

    # Recap detection
    "recap_max_duration": 90.0,  # first N seconds to check for recap
    "recap_similarity_threshold": 0.85,  # text similarity for duplicate detection

    # Content origin
    "manga_dialogue_match_threshold": 0.75,  # fuzzy match threshold

    # Resume
    "resume": False,

    # Character ID aliases (same as manga pipeline for consistency)
    "character_aliases": {
        "monkey d. luffy": "luffy",
        "monkey d luffy": "luffy",
        "straw hat luffy": "luffy",
        "roronoa zoro": "zoro",
        "pirate hunter zoro": "zoro",
        "vinsmoke sanji": "sanji",
        "black leg sanji": "sanji",
        "cat burglar nami": "nami",
        "tony tony chopper": "chopper",
        "cotton candy lover chopper": "chopper",
        "nico robin": "robin",
        "devil child robin": "robin",
        "cyborg franky": "franky",
        "cutty flam": "franky",
        "soul king brook": "brook",
        "dead bones brook": "brook",
        "god usopp": "usopp",
        "sogeking": "usopp",
        "jinbei": "jinbe",
        "jimbei": "jinbe",
        "knight of the sea jinbe": "jinbe",
        "gol d. roger": "gold_roger",
        "gol d roger": "gold_roger",
        "gold roger": "gold_roger",
        "pirate king roger": "gold_roger",
        "red-haired shanks": "shanks",
        "red hair shanks": "shanks",
        "shanks": "shanks",
        "dracule mihawk": "mihawk",
        "hawk eyes mihawk": "mihawk",
        "portgas d. ace": "ace",
        "portgas d ace": "ace",
        "fire fist ace": "ace",
        "trafalgar d. water law": "law",
        "trafalgar law": "law",
        "surgeon of death law": "law",
        "narrator": "narrator",
    },
}

# Valid values for VLM-classified shot types and content origins
VALID_SHOT_TYPES = {
    "wide", "medium", "close-up", "over-the-shoulder", "reaction", "aerial", "POV",
}
VALID_CONTENT_ORIGINS = {
    "manga_derived", "reaction_shot", "action_extension",
    "establishing", "transition", "anime_original",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_character_id(name):
    """Normalize a character name to a consistent snake_case ID."""
    if not name:
        return None
    name_lower = name.strip().lower()

    # Check aliases first
    if name_lower in CONFIG["character_aliases"]:
        return CONFIG["character_aliases"][name_lower]

    # Generic normalization: lowercase, replace spaces/hyphens with underscores
    normalized = re.sub(r"[^a-z0-9]+", "_", name_lower).strip("_")
    return normalized or None


def fuzzy_match_score(text_a, text_b):
    """
    Compute a simple fuzzy match score between two text strings.
    Returns float 0.0–1.0 based on word overlap (Jaccard similarity).
    """
    if not text_a or not text_b:
        return 0.0

    words_a = set(re.findall(r'\w+', text_a.lower()))
    words_b = set(re.findall(r'\w+', text_b.lower()))

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def load_json(path):
    """Load a JSON file, return None if not found."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types during JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    """Save data to a JSON file with pretty printing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

STAGE_NAMES = {
    1: "Subtitle extraction",
    2: "Shot boundary detection (TransNetV2)",
    3: "Per-shot feature extraction",
    4: "Scene detection",
    5: "VLM scene analysis (GPT-4o)",
    6: "Content origin classification",
    7: "Speaker attribution & dialogue index",
    8: "Output files (scenes, shots, characters)",
    9: "Episode manifest",
}


def _fmt_elapsed(seconds):
    """Format elapsed seconds as 'Xm Ys' or 'Ys'."""
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"


def _stage_start(stage_num):
    """Print stage banner and return start timestamp."""
    name = STAGE_NAMES.get(stage_num, "")
    print(f"\n{'=' * 70}")
    print(f"  [{stage_num}/9] {name}")
    print(f"{'=' * 70}")
    return time.time()


def _stage_end(stage_num, t_start):
    """Print stage completion with elapsed time, return elapsed seconds."""
    elapsed = time.time() - t_start
    name = STAGE_NAMES.get(stage_num, "")
    print(f"  OK [{stage_num}/9] {name} -- {_fmt_elapsed(elapsed)}")
    return elapsed


def _progress(current, total, prefix="", suffix="", every=1):
    """Print progress line if at the right interval."""
    if current % every != 0 and current != total:
        return
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "#" * filled + "." * (bar_len - filled)
    print(f"  {prefix}[{bar}] {current}/{total} ({pct:.0f}%) {suffix}", flush=True)


# ---------------------------------------------------------------------------
# [1/9] Subtitle extraction
# ---------------------------------------------------------------------------

def extract_subtitles(video_path, output_dir):
    """
    Extract subtitles from video file and parse into segment list.
    Uses embedded subtitle tracks (ASS/SRT) instead of Whisper API —
    free, faster, and typically more accurate for fansubbed releases.

    Returns list of segment dicts:
        [{"id": int, "start": float, "end": float, "text": str, "words": []}, ...]
    """
    subs_cache = os.path.join(output_dir, "subtitles_raw.json")

    if CONFIG["resume"] and os.path.exists(subs_cache):
        print("[1/9] Subtitles cached, loading...")
        return load_json(subs_cache)

    print("[1/9] Extracting subtitles from video...")

    # Probe subtitle tracks
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "s", video_path
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)

    sub_streams = []
    if probe_result.returncode == 0:
        probe_data = json.loads(probe_result.stdout)
        sub_streams = probe_data.get("streams", [])

    if not sub_streams:
        print("[1/9] Warning: No subtitle tracks found in video")
        return []

    # Pick subtitle track (prefer English, fall back to first)
    track_index = 0
    for i, stream in enumerate(sub_streams):
        tags = stream.get("tags", {})
        lang = tags.get("language", "").lower()
        title = tags.get("title", "").lower()
        if lang in ("eng", "en") or "english" in title:
            track_index = i
            break

    codec = sub_streams[track_index].get("codec_name", "ass")
    tags = sub_streams[track_index].get("tags", {})
    print(f"[1/9] Using subtitle track {track_index}: "
          f"{tags.get('language', '?')} / {tags.get('title', '?')} ({codec})")

    # Extract subtitle track
    sub_ext = "ass" if codec in ("ass", "ssa") else "srt"
    sub_path = os.path.join(output_dir, f"subtitles.{sub_ext}")

    extract_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-map", f"0:s:{track_index}", "-c:s", sub_ext, sub_path
    ]
    subprocess.run(extract_cmd, check=True, capture_output=True)

    # Parse subtitle file
    with open(sub_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    if sub_ext == "ass":
        segments = _parse_ass(content)
    else:
        segments = _parse_srt(content)

    save_json(segments, subs_cache)
    print(f"[1/9] Extracted {len(segments)} subtitle segments")
    return segments


def _parse_ass_time(time_str):
    """Parse ASS timestamp 'H:MM:SS.CC' to seconds."""
    h, m, rest = time_str.split(":")
    s, cs = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0


def _strip_ass_tags(text):
    """Strip ASS override tags and clean text."""
    text = re.sub(r'\{[^}]*\}', '', text)
    text = text.replace('\\N', ' ').replace('\\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _parse_ass(content):
    """Parse ASS subtitle content into segment list."""
    segments = []
    seg_id = 0

    for line in content.split('\n'):
        line = line.strip()
        if not line.startswith('Dialogue:'):
            continue

        # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        parts = line.split(',', 9)
        if len(parts) < 10:
            continue

        start_str = parts[1].strip()
        end_str = parts[2].strip()
        text_raw = parts[9]

        text = _strip_ass_tags(text_raw)
        if not text:
            continue

        start = round(_parse_ass_time(start_str), 3)
        end = round(_parse_ass_time(end_str), 3)

        segments.append({
            "id": seg_id,
            "start": start,
            "end": end,
            "text": text,
            "words": [],
        })
        seg_id += 1

    # Sort by start time and re-assign IDs
    segments.sort(key=lambda s: s["start"])
    for i, seg in enumerate(segments):
        seg["id"] = i

    return segments


def _parse_srt_time(time_str):
    """Parse SRT timestamp 'HH:MM:SS,mmm' to seconds."""
    h, m, rest = time_str.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_srt(content):
    """Parse SRT subtitle content into segment list."""
    segments = []
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        # Find the timestamp line
        time_line = None
        text_start = 0
        for j, ln in enumerate(lines):
            if '-->' in ln:
                time_line = ln
                text_start = j + 1
                break

        if not time_line:
            continue

        match = re.match(
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
            time_line
        )
        if not match:
            continue

        start = round(_parse_srt_time(match.group(1)), 3)
        end = round(_parse_srt_time(match.group(2)), 3)
        text = ' '.join(lines[text_start:]).strip()
        text = re.sub(r'<[^>]+>', '', text)  # Strip HTML tags
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            continue

        segments.append({
            "id": len(segments),
            "start": start,
            "end": end,
            "text": text,
            "words": [],
        })

    return segments


# ---------------------------------------------------------------------------
# [2/9] Shot boundary detection
# ---------------------------------------------------------------------------

def run_shot_detection(video_path, output_dir):
    """Run TransNetV2 shot detection via scene_detector module."""
    shots_cache = os.path.join(output_dir, "shots_raw.json")

    if CONFIG["resume"] and os.path.exists(shots_cache):
        print("[2/9] Shot detection cached, loading...")
        return load_json(shots_cache)

    print("[2/9] Running shot boundary detection...")
    from extraction.scene_detector import detect_shots
    shots = detect_shots(video_path)

    save_json(shots, shots_cache)
    print(f"[2/9] Detected {len(shots)} shots")
    return shots


# ---------------------------------------------------------------------------
# [3/9] Per-shot feature extraction
# ---------------------------------------------------------------------------

def run_feature_extraction(video_path, shots, subtitle_segments, output_dir,
                           gpu_models=None, audio_path=None):
    """Extract per-shot features via scene_detector module.

    Uses lightweight OpenCV features by default. When gpu_models is provided,
    also runs YOLOv8 character segmentation, CLIP background embeddings, and
    dominant hex color extraction. When audio_path is provided, runs librosa-based
    audio classification per shot (music, ambient, silence detection).

    Also saves a keyframe JPEG for every shot into the keyframes/ subdirectory.
    """
    features_cache = os.path.join(output_dir, "shots_features.json")

    if CONFIG["resume"] and os.path.exists(features_cache):
        print("[3/9] Feature extraction cached, loading...")
        shots = load_json(features_cache)
        # Migrate old keyframe_path (singular) to keyframe_paths (list)
        for shot in shots:
            if "keyframe_path" in shot and "keyframe_paths" not in shot:
                old = shot.pop("keyframe_path")
                shot["keyframe_paths"] = [old] if old else []
        return shots

    print("[3/9] Extracting per-shot features...")
    keyframes_dir = os.path.join(output_dir, CONFIG["keyframes_subdir"])
    from extraction.scene_detector import extract_shot_features
    shots = extract_shot_features(
        video_path, shots,
        audio_path=audio_path,
        subtitle_segments=subtitle_segments,
        keyframes_dir=keyframes_dir,
        gpu_models=gpu_models,
    )

    save_json(shots, features_cache)
    total_kf = sum(len(s.get('keyframe_paths', [])) for s in shots)
    shots_with_kf = sum(1 for s in shots if s.get('keyframe_paths'))
    bg_count = len([s for s in shots if s.get('background_plate_path')])
    ac_count = len([s for s in shots
                    if s.get('audio_classification', {}).get('audio_type') != 'unknown'])
    summary = f"({total_kf} keyframes across {shots_with_kf} shots)"
    if bg_count:
        summary = f"({total_kf} keyframes across {shots_with_kf} shots, {bg_count} background plates)"
    if ac_count:
        music_count = len([s for s in shots
                           if s.get('audio_classification', {}).get('has_music')])
        silence_count = len([s for s in shots
                             if s.get('audio_classification', {}).get('is_silence')])
        summary += f" (audio: {music_count} music, {silence_count} silence)"
    print(f"[3/9] Features extracted for {len(shots)} shots {summary}")
    return shots


# ---------------------------------------------------------------------------
# [4/9] Scene detection
# ---------------------------------------------------------------------------

def run_scene_detection(shots, output_dir):
    """Group shots into scenes via scene_detector module."""
    scenes_cache = os.path.join(output_dir, "scenes_raw.json")

    if CONFIG["resume"] and os.path.exists(scenes_cache):
        print("[4/9] Scene detection cached, loading...")
        return load_json(scenes_cache)

    print("[4/9] Detecting scenes...")
    from extraction.scene_detector import detect_scenes, detect_eyecatch

    # Detect eyecatch first to exclude from scene grouping
    eyecatch_ids = detect_eyecatch(shots)

    # Mark eyecatch shots
    for shot in shots:
        shot["_is_eyecatch"] = shot["shot_id"] in eyecatch_ids

    scenes = detect_scenes(shots)

    # Tag eyecatch scenes
    for scene in scenes:
        eyecatch_shots_in_scene = [
            sid for sid in scene["shot_ids"] if sid in eyecatch_ids
        ]
        scene["_is_eyecatch"] = len(eyecatch_shots_in_scene) > 0

    save_json(scenes, scenes_cache)
    print(f"[4/9] Grouped into {len(scenes)} scenes "
          f"({sum(1 for s in scenes if s.get('_is_eyecatch'))} eyecatch)")
    return scenes


# ---------------------------------------------------------------------------
# [5/9] VLM analysis per scene
# ---------------------------------------------------------------------------

def run_vlm_scene_analysis(video_path, shots, scenes, subtitle_segments,
                           chapter_manifest, output_dir, wiki_context=None):
    """
    Run GPT-4o vision analysis on each scene. One VLM call per scene using
    the establishing shot keyframe, the scene's transcribed dialogue,
    the chapter manifest as context, and wiki episode context for grounding.
    """
    print(f"[5/9] Running VLM scene analysis on {len(scenes)} scenes...")
    import base64
    from openai import OpenAI

    client = OpenAI(api_key=CONFIG["openai_api_key"])

    keyframes_dir = os.path.join(output_dir, CONFIG["keyframes_subdir"])
    os.makedirs(keyframes_dir, exist_ok=True)

    # Build shot lookup
    shot_map = {s["shot_id"]: s for s in shots}

    # Extract location vocabulary from chapter manifest if available
    location_vocab = _extract_location_vocabulary(chapter_manifest)

    # Build manga dialogue list for context
    manga_dialogue = _extract_manga_dialogue(chapter_manifest)

    # Inject total_scenes count for prompt context
    for s in scenes:
        s["total_scenes"] = len(scenes)

    # Rate limiting
    last_call_time = 0
    min_interval = 60.0 / CONFIG["vlm_calls_per_minute"]

    # Track previous scene result for continuity context
    prev_scene_result = None

    # VLM result cache directory (for accurate resume of per-shot analysis)
    vlm_cache_dir = os.path.join(output_dir, "vlm_cache")

    for scene_idx, scene in enumerate(scenes):
        vlm_cache_path = os.path.join(
            vlm_cache_dir, f"scene_{scene['scene_number']:03d}.json"
        )
        scene_file = os.path.join(output_dir, f"scene_{scene['scene_number']:03d}.json")

        # --- Resume: try VLM cache first, then legacy scene file ---
        if CONFIG["resume"]:
            cached_vlm = None
            if os.path.exists(vlm_cache_path):
                cached_vlm = load_json(vlm_cache_path)
            elif os.path.exists(scene_file):
                cached = load_json(scene_file)
                if cached and cached.get("scene_narrative"):
                    # Build partial VLM-like result from legacy cache
                    cached_vlm = {
                        "scene_narrative": cached.get("scene_narrative", ""),
                        "location": cached.get("location"),
                        "time_context": cached.get("time_context"),
                        "emotional_tone": cached.get("emotional_tone"),
                        "scene_type": cached.get("scene_type"),
                        "establishing_shot_type": cached.get("establishing_shot_type"),
                        "characters_present": [
                            {"name": c.get("name", ""), "action": c.get("action", ""),
                             "emotion": c.get("emotion", "")}
                            for c in cached.get("characters_present", [])
                        ],
                        "reconstruction_prompt": cached.get("reconstruction_prompt", ""),
                    }
            if cached_vlm:
                _apply_vlm_scene_result(cached_vlm, scene, shot_map)
                scene_dialogue = _get_scene_dialogue(scene, subtitle_segments)
                _apply_vlm_dialogue_attribution(cached_vlm, scene, scene_dialogue)
                # Set keyframe_paths from per-shot data (even on resume)
                scene_shots_r = [shot_map[sid] for sid in scene["shot_ids"] if sid in shot_map]
                vlm_kf_info_r = _select_vlm_keyframes(scene_shots_r, scene["duration"], CONFIG)
                scene["keyframe_paths"] = [path for _, path in vlm_kf_info_r]
                prev_scene_result = {
                    "scene_narrative": scene.get("scene_narrative", ""),
                    "location": scene.get("location"),
                    "time_context": scene.get("time_context"),
                    "characters_present": cached_vlm.get("characters_present", []),
                }
                continue

        # Skip eyecatch scenes
        if scene.get("_is_eyecatch"):
            scene["scene_type"] = "eyecatch"
            scene["scene_narrative"] = "Mid-episode eyecatch title card."
            scene["location"] = None
            scene["time_context"] = None
            scene["emotional_tone"] = "calm"
            scene["characters_present"] = []
            scene["establishing_shot_type"] = "none"
            scene["reconstruction_prompt"] = ""
            scene["keyframe_paths"] = []
            scene["_vlm_dialogue_attribution"] = {}
            prev_scene_result = None  # eyecatch breaks continuity
            continue

        # Gather dialogue and build shot list for this scene
        scene_dialogue = _get_scene_dialogue(scene, subtitle_segments)
        scene_shots = [shot_map[sid] for sid in scene["shot_ids"] if sid in shot_map]

        # Select keyframes to send to VLM (duration-based, from per-shot keyframes on disk)
        vlm_keyframe_info = _select_vlm_keyframes(scene_shots, scene["duration"], CONFIG)
        vlm_keyframe_paths = [path for _, path in vlm_keyframe_info]
        scene["keyframe_paths"] = vlm_keyframe_paths

        # Build VLM prompt with full context
        prompt = _build_scene_analysis_prompt(
            scene, scene_dialogue, location_vocab, manga_dialogue,
            wiki_context=wiki_context,
            prev_scene_result=prev_scene_result,
            scene_shots=scene_shots,
            vlm_keyframe_info=vlm_keyframe_info,
        )

        # Rate limit
        elapsed = time.time() - last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        # Call VLM
        vlm_result = _call_vlm_scene(client, vlm_keyframe_paths, prompt)
        last_call_time = time.time()

        # Cache raw VLM result for accurate resume
        if vlm_result:
            os.makedirs(vlm_cache_dir, exist_ok=True)
            save_json(vlm_result, vlm_cache_path)

        # Apply VLM result to scene and shots
        _apply_vlm_scene_result(vlm_result, scene, shot_map)
        _apply_vlm_dialogue_attribution(vlm_result, scene, scene_dialogue)

        if vlm_result:
            prev_scene_result = {
                "scene_narrative": scene["scene_narrative"],
                "location": scene["location"],
                "time_context": scene["time_context"],
                "characters_present": vlm_result.get("characters_present", []),
            }
        else:
            prev_scene_result = None

        _progress(scene_idx + 1, len(scenes), prefix="[5/9] ", every=3)

    return scenes


def _pick_establishing_shot(scene, shot_map):
    """Pick the best establishing shot for a scene — prefer first wide/aerial shot."""
    for sid in scene["shot_ids"]:
        shot = shot_map.get(sid, {})
        if shot.get("shot_type") in ("wide", "aerial"):
            return sid
    return scene["shot_ids"][0]


def _get_scene_dialogue(scene, subtitle_segments):
    """Get all Whisper dialogue that falls within a scene's time range."""
    lines = []
    for seg in subtitle_segments:
        if seg["end"] < scene["start_time"]:
            continue
        if seg["start"] > scene["end_time"]:
            break
        # Check overlap
        if seg["start"] < scene["end_time"] and seg["end"] > scene["start_time"]:
            lines.append({
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "subtitle_id": seg["id"],
            })
    return lines


def _extract_location_vocabulary(chapter_manifest):
    """Extract location strings from chapter manifest for controlled vocabulary.

    Reads from the dedicated `locations` field first (aggregated from all panels),
    then falls back to extracting from dialogue_index entries.
    """
    if not chapter_manifest:
        return []

    locations = set()

    # Prefer the dedicated locations field (added by manga pipeline)
    manifest_locations = chapter_manifest.get("locations", [])
    if manifest_locations:
        locations.update(manifest_locations)
    else:
        # Fallback: extract from dialogue_index entries
        for entry in chapter_manifest.get("dialogue_index", []):
            loc = entry.get("location")
            if loc:
                locations.add(loc)

    return sorted(locations)


def _extract_manga_dialogue(chapter_manifest):
    """Extract flat dialogue list from chapter manifest for context."""
    if not chapter_manifest:
        return []

    return [
        {
            "seq": entry.get("seq"),
            "content": entry.get("content", ""),
            "speaker": entry.get("speaker", ""),
            "location": entry.get("location", ""),
            "time_context": entry.get("time_context", ""),
        }
        for entry in chapter_manifest.get("dialogue_index", [])
    ]


def fetch_wiki_episode_context(episode_id):
    """
    Fetch structured episode context from the One Piece wiki.

    Returns a dict with character_list, synopsis, and arc/title info,
    or None if the wiki API is unavailable or the fetch fails.
    """
    if not HAS_WIKI_API:
        return None

    ep_num = int(episode_id) if episode_id.isdigit() else None
    if ep_num is None:
        return None

    try:
        characters = get_episode_characters(ep_num)
        synopsis = get_episode_summary(ep_num)
        infobox = get_episode_infobox(ep_num)
    except Exception as e:
        logger.warning(f"Wiki API fetch failed for episode {ep_num}: {e}")
        return None

    # Build character list with annotations (e.g. "flashback")
    char_entries = []
    for c in characters:
        entry = c["name"]
        if c.get("annotation"):
            entry += f" ({c['annotation']})"
        char_entries.append(entry)

    arc_lookup = get_arc_for_episode(ep_num)
    arc_name = arc_lookup["arc"] if arc_lookup else ""
    saga_name = arc_lookup.get("saga", "") if arc_lookup else ""
    context = {
        "episode_number": ep_num,
        "episode_title": infobox.get("english") or infobox.get("translation", ""),
        "synopsis": synopsis,
        "characters": char_entries,
        "arc": arc_name,
        "arc_id": normalize_arc_id(arc_name),
        "saga": saga_name,
    }

    # Extract location debuts if present
    loc_debut = infobox.get("locationdebut", "")
    if loc_debut:
        context["location_debuts"] = [l.strip() for l in loc_debut.split(",") if l.strip()]

    return context


def _select_vlm_keyframes(scene_shots, scene_duration, cfg):
    """Select which shots' primary keyframes to send to the VLM.

    Duration-based selection (no hard cap):
    - Scene < 15s: all shots' primary keyframes
    - Scene 15-60s: 1 image per ~5s of scene duration
    - Scene > 60s: 1 image per ~8s of scene duration

    When sampling, always includes first and last shot, fills evenly between.

    Returns:
        List of (shot_id, keyframe_path) tuples in scene order.
    """
    # Build candidates: (shot_id, primary_keyframe_path) for each shot
    candidates = []
    for shot in scene_shots:
        paths = shot.get("keyframe_paths", [])
        if not paths:
            continue
        # Primary is the middle path (convention from _compute_keyframe_positions)
        primary_idx = len(paths) // 2 if len(paths) > 1 else 0
        candidates.append((shot["shot_id"], paths[primary_idx]))

    if not candidates:
        return []

    short_thresh = cfg.get("vlm_short_scene_threshold", 15.0)
    medium_interval = cfg.get("vlm_medium_scene_interval", 5.0)
    long_thresh = cfg.get("vlm_long_scene_threshold", 60.0)
    long_interval = cfg.get("vlm_long_scene_interval", 8.0)

    # Determine how many keyframes to send
    if scene_duration < short_thresh:
        return candidates  # all shots

    if scene_duration < long_thresh:
        target_count = max(2, round(scene_duration / medium_interval))
    else:
        target_count = max(2, round(scene_duration / long_interval))

    if len(candidates) <= target_count:
        return candidates  # have fewer than target, send all

    # Uniform sample: always include first and last, fill evenly between
    indices = [0, len(candidates) - 1]
    remaining = target_count - 2
    if remaining > 0:
        step = (len(candidates) - 1) / (remaining + 1)
        for i in range(1, remaining + 1):
            idx = int(round(i * step))
            if idx not in indices:
                indices.append(idx)
    indices = sorted(set(indices))[:target_count]
    return [candidates[i] for i in indices]


def _build_scene_analysis_prompt(scene, scene_dialogue, location_vocab, manga_dialogue,
                                 wiki_context=None, prev_scene_result=None,
                                 scene_shots=None, vlm_keyframe_info=None):
    """Build the VLM prompt for scene analysis.

    Args:
        scene: Scene dict with timing, shot info, flashback flags
        scene_dialogue: List of subtitle segments in this scene
        location_vocab: Location strings from manga chapter manifest
        manga_dialogue: Flat dialogue list from manga chapter manifest
        wiki_context: Episode context from wiki API (characters, synopsis, etc.)
        prev_scene_result: VLM result dict from the previous scene (may be unrelated)
        scene_shots: List of shot dicts for shots in this scene (for per-shot analysis)
        vlm_keyframe_info: List of (shot_id, keyframe_path) tuples for images sent to VLM
    """
    dialogue_text = "\n".join(
        f"  [{i}] {d['start']:.1f}s-{d['end']:.1f}s: \"{d['text']}\""
        for i, d in enumerate(scene_dialogue)
    ) or "  (no dialogue in this scene)"

    # --- Episode context block (wiki data) ---
    episode_context = ""
    if wiki_context:
        parts = []
        if wiki_context.get("episode_title"):
            ep_num = wiki_context.get("episode_number", "?")
            parts.append(f"Episode {ep_num}: \"{wiki_context['episode_title']}\"")
        if wiki_context.get("arc"):
            parts.append(f"Arc: {wiki_context['arc']}")
        if wiki_context.get("synopsis"):
            parts.append(f"Episode synopsis: {wiki_context['synopsis']}")
        if wiki_context.get("characters"):
            char_list = ", ".join(wiki_context["characters"])
            parts.append(
                f"Characters in this episode (use ONLY these names): {char_list}"
            )
        if parts:
            episode_context = "\nEpisode context:\n" + "\n".join(f"  {p}" for p in parts)

    # --- Location vocabulary ---
    location_hint = ""
    all_locations = list(location_vocab) if location_vocab else []
    if wiki_context and wiki_context.get("location_debuts"):
        for loc in wiki_context["location_debuts"]:
            if loc not in all_locations:
                all_locations.append(loc)
    if all_locations:
        location_hint = (
            f"\nPreferred location vocabulary (use these exact strings when applicable): "
            f"{', '.join(all_locations)}"
        )

    # --- Previous scene context (careful framing) ---
    prev_context = ""
    if prev_scene_result and prev_scene_result.get("scene_narrative"):
        prev_chars = ""
        if prev_scene_result.get("characters_present"):
            names = [c.get("name", "") for c in prev_scene_result["characters_present"]
                     if c.get("name")]
            if names:
                prev_chars = f" (characters: {', '.join(names)})"
        prev_context = (
            f"\nPrevious scene (for continuity — this scene may be a DIFFERENT "
            f"location/time/storyline, so only use this if there is clear continuity):\n"
            f"  \"{prev_scene_result['scene_narrative']}\""
            f"{prev_chars}"
        )
        if prev_scene_result.get("location"):
            prev_context += f"\n  Location: {prev_scene_result['location']}"
        if prev_scene_result.get("time_context"):
            prev_context += f"\n  Time: {prev_scene_result['time_context']}"

    # --- Manga dialogue reference ---
    manga_context = ""
    if manga_dialogue:
        manga_lines = "\n".join(
            f"  [{d['seq']}] {d['speaker']}: {d['content']}"
            for d in manga_dialogue[:50]
        )
        manga_context = (
            f"\n\nReference manga dialogue (for content_origin — if a shot's dialogue "
            f"closely matches a line here, its content_origin is 'manga_derived'):\n{manga_lines}"
        )

    # --- Shot info for per-shot analysis ---
    shot_info = ""
    if scene_shots:
        # Build set of shot_ids that have keyframe images sent to VLM
        vlm_shot_ids = set()
        if vlm_keyframe_info:
            vlm_shot_ids = {sid for sid, _ in vlm_keyframe_info}

        shot_lines = []
        for s in scene_shots:
            has_dial = any(
                d["start"] < s["end_time"] and d["end"] > s["start_time"]
                for d in scene_dialogue
            )
            dial_str = "has dialogue" if has_dial else "no dialogue"
            motion = s.get("motion_intensity", 0)
            img_tag = " [has keyframe image]" if s["shot_id"] in vlm_shot_ids else ""
            shot_lines.append(
                f"  Shot {s['shot_id']}: {s['start_time']:.1f}s-{s['end_time']:.1f}s "
                f"({s['duration']:.1f}s) [{dial_str}, motion={motion:.2f}]{img_tag}"
            )
        shot_info = "\nShots in this scene:\n" + "\n".join(shot_lines)

    # --- Image-to-shot mapping ---
    image_mapping = ""
    if vlm_keyframe_info and len(vlm_keyframe_info) > 0:
        # Build shot timing lookup
        shot_time_map = {}
        if scene_shots:
            shot_time_map = {s["shot_id"]: s for s in scene_shots}
        mapping_lines = []
        for img_idx, (shot_id, _) in enumerate(vlm_keyframe_info):
            s = shot_time_map.get(shot_id, {})
            t_start = s.get("start_time", 0)
            t_end = s.get("end_time", 0)
            mapping_lines.append(
                f"  Image {img_idx + 1} -> Shot {shot_id} ({t_start:.1f}s-{t_end:.1f}s)"
            )
        image_mapping = (
            f"\nKeyframe images provided ({len(vlm_keyframe_info)} images, in order):\n"
            + "\n".join(mapping_lines)
        )

    # --- Character identification rules ---
    char_rules = ""
    if wiki_context and wiki_context.get("characters"):
        char_rules = """
- CRITICAL: For characters_present, use the character's actual name from the episode character list above.
  Do NOT use generic labels like "unknown", "Character 1", "pirate", "boy", "man", etc.
  If you can identify the character from the image or dialogue, use their name.
  If you truly cannot identify someone, omit them rather than using a generic label."""
    else:
        char_rules = """
- For characters_present, use proper character names (e.g. "Luffy", "Zoro", "Nami").
  Do NOT use generic labels like "unknown", "Character 1", "pirate", etc.
  If you truly cannot identify someone, omit them rather than using a generic label."""

    num_images = len(vlm_keyframe_info) if vlm_keyframe_info else 0
    image_intro = (
        f"Analyze these {num_images} anime scene keyframes and transcribed dialogue."
        if num_images > 1
        else "Analyze this anime scene keyframe and transcribed dialogue."
    )

    return f"""{image_intro}
{episode_context}
Scene info:
  Scene {scene['scene_number']}/{scene.get('total_scenes', '?')} in episode
  Timestamp: {scene.get('start_time', 0):.1f}s - {scene.get('end_time', 0):.1f}s
  Duration: {scene['duration']:.1f}s
  Shot count: {scene['shot_count']}
  Flashback candidate: {scene.get('has_flashback_candidate', False)}
{shot_info}{image_mapping}{prev_context}
Subtitle lines in this scene (with indices for speaker attribution):
{dialogue_text}
{location_hint}{manga_context}

Respond with a JSON object containing:
{{
  "scene_narrative": "1-2 sentence summary of what happens in this scene",
  "location": "specific location name (e.g. 'Partys Bar', 'execution platform at Loguetown')",
  "time_context": "present day | X years ago | flashback to [event]",
  "emotional_tone": "tense | comedic | melancholic | triumphant | dramatic | calm | action",
  "scene_type": "story | opening | ending | recap | filler | flashback | transition | eyecatch",
  "establishing_shot_type": "wide | medium | aerial | none",
  "reconstruction_prompt": "A director's description for recreating this scene from scratch...",
  "characters_present": [
    {{
      "name": "character name",
      "action": "what they are doing",
      "emotion": "their emotional state"
    }}
  ],
  "shot_analysis": [
    {{
      "shot_id": <number>,
      "shot_type": "wide | medium | close-up | over-the-shoulder | reaction | aerial | POV",
      "content_origin": "manga_derived | reaction_shot | action_extension | establishing | transition | anime_original"
    }}
  ],
  "dialogue_attribution": [
    {{
      "line_index": <subtitle_line_index_from_above>,
      "speaker": "character name",
      "speaker_confidence": <0.0-1.0>
    }}
  ]
}}

Rules:
- Use the preferred location vocabulary when the location matches.
- For time_context, if the scene appears to be a flashback (desaturated colors, past events), specify the time period.
- scene_type "recap" should only be used for content that repeats earlier episodes.
- scene_type "opening" for the OP theme sequence, "ending" for the ED theme sequence.
- Be specific about character actions and emotions.{char_rules}
- reconstruction_prompt: describe camera work, timing, framing, and tone as a director would for recreating this scene. Do not reference the manga source. Example: "Open on a wide shot of the execution platform, late afternoon light, thousands silent. Slow push-in on Roger's face -- defiant, almost amused. Hold 4s. Cut to wide crowd reaction. Return to close-up as he speaks. Tone: historic and mythic."
- shot_analysis: return one entry per shot listed above. For shots marked [has keyframe image], use the corresponding image to determine shot_type and content_origin. For shots without a keyframe image, infer from timing, dialogue, and context of surrounding shots. For content_origin: "manga_derived" if the shot's dialogue matches manga reference lines; "reaction_shot" for character reactions without manga source; "action_extension" for extended fight/action beyond manga; "establishing" for location-establishing shots; "transition" for brief transitions; "anime_original" for fully invented content.
- dialogue_attribution: for each subtitle line above, identify the most likely speaker from the characters present. speaker_confidence 0.8-1.0 when clearly identifiable, 0.4-0.7 when uncertain.
- Return ONLY the JSON object, no markdown formatting."""


def _call_vlm_scene(client, keyframe_paths, prompt):
    """Call GPT-4o with one or more keyframe images and a text prompt.

    Args:
        client: OpenAI client instance
        keyframe_paths: List of keyframe image paths to send (one per shot)
        prompt: Text prompt for scene analysis
    """
    import base64

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    # Add images (inserted before text, in order)
    if keyframe_paths:
        image_contents = []
        for kf_path in keyframe_paths:
            if kf_path and os.path.exists(kf_path):
                with open(kf_path, "rb") as img_file:
                    image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high",
                    }
                })
        # Insert all images before the text content
        for img in reversed(image_contents):
            messages[0]["content"].insert(0, img)

    for attempt in range(CONFIG["vlm_max_retries"]):
        try:
            response = client.chat.completions.create(
                model=CONFIG["openai_model"],
                messages=messages,
                max_tokens=CONFIG["openai_max_tokens"],
                temperature=CONFIG["openai_temperature"],
            )
            # Track API cost
            usage = response.usage
            if usage:
                API_COSTS["gpt4o_input_tokens"] += usage.prompt_tokens
                API_COSTS["gpt4o_output_tokens"] += usage.completion_tokens
            API_COSTS["vlm_calls"] += 1

            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"VLM JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < CONFIG["vlm_max_retries"] - 1:
                time.sleep(CONFIG["vlm_retry_delay"])
        except Exception as e:
            logger.warning(f"VLM API error (attempt {attempt + 1}): {e}")
            if attempt < CONFIG["vlm_max_retries"] - 1:
                time.sleep(CONFIG["vlm_retry_delay"])

    return None


def _normalize_vlm_characters(characters_list):
    """Normalize character data from VLM response."""
    normalized = []
    for char in characters_list:
        name = char.get("name", "")
        char_id = normalize_character_id(name)
        normalized.append({
            "character_id": char_id,
            "name": name,
            "action": char.get("action", ""),
            "emotion": char.get("emotion", ""),
        })
    return normalized


def _apply_vlm_scene_result(vlm_result, scene, shot_map):
    """Apply VLM scene analysis result to scene and shot data."""
    if not vlm_result:
        scene["scene_narrative"] = ""
        scene["location"] = None
        scene["time_context"] = "present day"
        scene["emotional_tone"] = "calm"
        scene["scene_type"] = "story"
        scene["establishing_shot_type"] = "medium"
        scene["characters_present"] = []
        scene["reconstruction_prompt"] = ""
        return

    scene["scene_narrative"] = vlm_result.get("scene_narrative", "")
    scene["location"] = vlm_result.get("location")
    scene["time_context"] = vlm_result.get("time_context", "present day")
    scene["emotional_tone"] = vlm_result.get("emotional_tone", "calm")
    scene["scene_type"] = vlm_result.get("scene_type", "story")
    scene["establishing_shot_type"] = vlm_result.get(
        "establishing_shot_type", "medium"
    )
    scene["characters_present"] = _normalize_vlm_characters(
        vlm_result.get("characters_present", [])
    )
    scene["reconstruction_prompt"] = vlm_result.get("reconstruction_prompt", "")

    # Apply per-shot analysis from VLM
    for sa in vlm_result.get("shot_analysis", []):
        sid = sa.get("shot_id")
        if sid is not None and sid in shot_map:
            vlm_shot_type = sa.get("shot_type")
            if vlm_shot_type in VALID_SHOT_TYPES:
                shot_map[sid]["shot_type"] = vlm_shot_type
            vlm_origin = sa.get("content_origin")
            if vlm_origin in VALID_CONTENT_ORIGINS:
                shot_map[sid]["vlm_content_origin"] = vlm_origin


def _apply_vlm_dialogue_attribution(vlm_result, scene, scene_dialogue):
    """Map VLM dialogue attribution back to subtitle IDs and store on the scene."""
    attribution_map = {}
    if vlm_result:
        for attr in vlm_result.get("dialogue_attribution", []):
            idx = attr.get("line_index")
            if idx is not None and 0 <= idx < len(scene_dialogue):
                sub_id = scene_dialogue[idx].get("subtitle_id")
                if sub_id is not None:
                    speaker = attr.get("speaker", "")
                    speaker_id = normalize_character_id(speaker) if speaker else None
                    attribution_map[sub_id] = {
                        "speaker": speaker,
                        "speaker_id": speaker_id,
                        "speaker_confidence": attr.get("speaker_confidence", 0.0),
                    }
    scene["_vlm_dialogue_attribution"] = attribution_map


# ---------------------------------------------------------------------------
# [6/9] Content origin classification per shot
# ---------------------------------------------------------------------------

def classify_content_origins(shots, scenes, subtitle_segments, chapter_manifest,
                             output_dir):
    """
    For every shot, classify whether it is manga-derived or a studio addition.

    Categories:
      manga_derived — shot corresponds to a manga panel
      reaction_shot — character reaction with no panel source
      action_extension — studio extended a fight/action beyond manga
      establishing — location-establishing shot studio added
      transition — brief transition with no narrative content
      anime_original — fully invented content with no manga basis
    """
    print(f"[6/9] Classifying content origins for {len(shots)} shots...")

    manga_dialogue = _extract_manga_dialogue(chapter_manifest)
    shot_map = {s["shot_id"]: s for s in shots}
    scene_map = {s["scene_number"]: s for s in scenes}

    # Assign dialogue lines to shots by timestamp
    shot_dialogue = _assign_dialogue_to_shots(shots, subtitle_segments)

    for shot_idx, shot in enumerate(shots):
        sid = shot["shot_id"]
        dialogue_in_shot = shot_dialogue.get(sid, [])

        # If shot is eyecatch, mark as transition
        if shot.get("_is_eyecatch"):
            shot["content_origin"] = "transition"
            shot["dialogue_in_shot"] = []
            continue

        shot["dialogue_in_shot"] = [d["text_id"] for d in dialogue_in_shot]

        # Check if any dialogue in this shot matches manga dialogue
        manga_match = False
        if dialogue_in_shot and manga_dialogue:
            for d_line in dialogue_in_shot:
                for m_line in manga_dialogue:
                    score = fuzzy_match_score(d_line["text"], m_line["content"])
                    if score >= CONFIG["manga_dialogue_match_threshold"]:
                        manga_match = True
                        break
                if manga_match:
                    break

        # Classification logic
        if manga_match:
            shot["content_origin"] = "manga_derived"
        elif shot.get("shot_type") in ("wide", "aerial") and not dialogue_in_shot:
            shot["content_origin"] = "establishing"
        elif shot.get("shot_type") == "reaction" and not dialogue_in_shot:
            shot["content_origin"] = "reaction_shot"
        elif shot.get("motion_intensity", 0) > 0.6 and not dialogue_in_shot:
            shot["content_origin"] = "action_extension"
        elif shot.get("duration", 0) < 1.0 and not dialogue_in_shot:
            shot["content_origin"] = "transition"
        else:
            # Check if scene has any manga-derived shots — if so, ambiguous shots
            # in the same scene are more likely reaction/extension than fully original
            scene_num = _find_scene_for_shot(shot, scenes)
            scene = scene_map.get(scene_num, {})

            scene_has_manga = False
            for s_id in scene.get("shot_ids", []):
                other = shot_map.get(s_id, {})
                if other.get("content_origin") == "manga_derived":
                    scene_has_manga = True
                    break

            if dialogue_in_shot and not manga_match:
                shot["content_origin"] = "anime_original"
            elif scene_has_manga and shot.get("shot_type") == "reaction":
                shot["content_origin"] = "reaction_shot"
            elif not dialogue_in_shot:
                shot["content_origin"] = "anime_original"
            else:
                shot["content_origin"] = "anime_original"

        # Use VLM content origin when available (overrides heuristic)
        vlm_origin = shot.get("vlm_content_origin")
        if vlm_origin and vlm_origin in VALID_CONTENT_ORIGINS:
            if shot["content_origin"] != vlm_origin:
                shot["_content_origin_heuristic"] = shot["content_origin"]
            shot["content_origin"] = vlm_origin

        _progress(shot_idx + 1, len(shots), prefix="[6/9] ", every=50)

    # Summary
    origin_counts = {}
    for shot in shots:
        origin = shot.get("content_origin", "unknown")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1

    print(f"  Content origins: {json.dumps(origin_counts)}")

    # Log VLM vs heuristic disagreements
    vlm_classified = sum(1 for s in shots if s.get("vlm_content_origin"))
    disagreements = sum(1 for s in shots if s.get("_content_origin_heuristic"))
    if vlm_classified > 0:
        print(f"  VLM content origin: {vlm_classified} shots classified, "
              f"{disagreements} disagreements with heuristic")

    return shots


def _assign_dialogue_to_shots(shots, subtitle_segments):
    """
    Assign Whisper dialogue segments to shots by timestamp overlap.
    Returns dict: {shot_id: [{"text_id": str, "text": str, ...}]}
    """
    dialogue_seq = 0
    shot_dialogue = {}

    for seg in subtitle_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        for shot in shots:
            # Check overlap
            if shot["end_time"] < seg_start:
                continue
            if shot["start_time"] > seg_end:
                break

            overlap_start = max(shot["start_time"], seg_start)
            overlap_end = min(shot["end_time"], seg_end)

            if overlap_end - overlap_start > 0.1:  # meaningful overlap
                dialogue_seq += 1
                text_id = f"s{shot['shot_id']}_d{dialogue_seq}"

                if shot["shot_id"] not in shot_dialogue:
                    shot_dialogue[shot["shot_id"]] = []

                shot_dialogue[shot["shot_id"]].append({
                    "text_id": text_id,
                    "text": seg["text"],
                    "start": seg_start,
                    "end": seg_end,
                })

    return shot_dialogue


def _find_scene_for_shot(shot, scenes):
    """Find which scene a shot belongs to."""
    for scene in scenes:
        if shot["shot_id"] in scene.get("shot_ids", []):
            return scene["scene_number"]
    return 0


# ---------------------------------------------------------------------------
# [7/9] Speaker attribution and dialogue index
# ---------------------------------------------------------------------------

def build_dialogue_index(shots, scenes, subtitle_segments, output_dir):
    """
    Build the flat dialogue index for the episode.
    Assigns speaker identity based on VLM character list per scene.

    Returns list of dialogue entries with sequence numbers, speaker info,
    and shot/scene references.
    """
    print(f"[7/9] Building dialogue index from {len(subtitle_segments)} subtitle segments...")

    scene_map = {s["scene_number"]: s for s in scenes}
    shot_scene_map = {}
    for scene in scenes:
        for sid in scene.get("shot_ids", []):
            shot_scene_map[sid] = scene["scene_number"]

    shot_map = {s["shot_id"]: s for s in shots}

    dialogue_index = []
    seq = 0

    for seg in subtitle_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        text = seg["text"].strip()
        if not text:
            continue

        # Find the primary shot this dialogue belongs to
        best_shot_id = None
        best_overlap = 0

        for shot in shots:
            if shot["end_time"] < seg_start:
                continue
            if shot["start_time"] > seg_end:
                break

            overlap_start = max(shot["start_time"], seg_start)
            overlap_end = min(shot["end_time"], seg_end)
            overlap = overlap_end - overlap_start

            if overlap > best_overlap:
                best_overlap = overlap
                best_shot_id = shot["shot_id"]

        if best_shot_id is None:
            continue

        scene_num = shot_scene_map.get(best_shot_id, 0)
        scene = scene_map.get(scene_num, {})
        shot = shot_map.get(best_shot_id, {})

        # Skip eyecatch and recap dialogue from the index
        if scene.get("_is_eyecatch"):
            continue
        if scene.get("scene_type") == "recap":
            continue

        seq += 1

        # Speaker attribution — prefer VLM attribution, fall back to heuristic
        vlm_attr = scene.get("_vlm_dialogue_attribution", {}).get(seg["id"])
        if vlm_attr and vlm_attr.get("speaker_id"):
            speaker = vlm_attr["speaker"]
            speaker_id = vlm_attr["speaker_id"]
            confidence = vlm_attr.get("speaker_confidence", 0.8)
        else:
            speaker, speaker_id, confidence = _attribute_speaker(
                seg, scene, shot
            )

        # Determine emotion from scene context
        emotion = _infer_emotion(text, scene)

        # Determine dialogue type
        dtype = _classify_dialogue_type(text, scene)

        dialogue_index.append({
            "seq": seq,
            "content": text,
            "type": dtype,
            "speaker_id": speaker_id,
            "speaker": speaker,
            "speaker_confidence": confidence,
            "emotion": emotion,
            "tone_direction": _compute_tone_direction(emotion, text),
            "scene_number": scene_num,
            "shot_id": best_shot_id,
            "start_time_seconds": round(seg_start, 3),
            "end_time_seconds": round(seg_end, 3),
            "location": scene.get("location"),
            "time_context": scene.get("time_context", "present day"),
        })

    speakers = set(d["speaker_id"] for d in dialogue_index if d.get("speaker_id"))
    print(f"  Dialogue index: {len(dialogue_index)} entries, {len(speakers)} unique speakers")
    return dialogue_index


def _attribute_speaker(segment, scene, shot):
    """
    Attribute a dialogue segment to a speaker based on scene context.

    Single visible character who is speaking → high confidence.
    Multiple characters → lower confidence with best guess.
    """
    characters = scene.get("characters_present", [])

    if not characters:
        return "Unknown", None, 0.0

    if len(characters) == 1:
        char = characters[0]
        return char.get("name", "Unknown"), char.get("character_id"), 0.9

    # Multiple characters — check if shot has character data
    shot_chars = shot.get("characters_present", [])
    if shot_chars and len(shot_chars) == 1:
        char_id = shot_chars[0]
        # Find matching character in scene
        for char in characters:
            if char.get("character_id") == char_id:
                return char.get("name", "Unknown"), char_id, 0.8
        return "Unknown", char_id, 0.6

    # Multiple characters, no clear attribution
    # Default to first character in scene (weak heuristic)
    char = characters[0]
    return char.get("name", "Unknown"), char.get("character_id"), 0.3


def _infer_emotion(text, scene):
    """Infer emotion from text content and scene context."""
    text_lower = text.lower()

    # Check text cues
    if text.endswith("!!!") or text.endswith("!!"):
        if any(w in text_lower for w in ["die", "kill", "damn", "bastard"]):
            return "angry"
        return "excited"
    if text.endswith("..."):
        return "melancholic"
    if "?" in text and "!" in text:
        return "surprised"
    if any(w in text_lower for w in ["haha", "hehe", "funny"]):
        return "happy"
    if any(w in text_lower for w in ["sorry", "forgive", "cry", "tears"]):
        return "sad"

    # Fall back to scene emotional tone
    scene_tone = scene.get("emotional_tone", "calm")
    tone_emotion_map = {
        "tense": "tense",
        "comedic": "happy",
        "melancholic": "sad",
        "triumphant": "determined",
        "dramatic": "dramatic",
        "calm": "calm",
        "action": "determined",
    }
    return tone_emotion_map.get(scene_tone, "neutral")


def _classify_dialogue_type(text, scene):
    """Classify whether text is dialog, narration, or internal monologue."""
    # Narration boxes typically appear as scene descriptions or episode context
    # In anime, narration is usually a distinct voice
    text_lower = text.lower()

    if any(text_lower.startswith(w) for w in [
        "previously", "last time", "in the", "the story", "and so",
        "this is the story", "wealth", "fame", "power",
    ]):
        return "narration"

    # Internal monologue often has thinking cues
    if any(w in text_lower for w in ["i think", "i must", "i have to", "could it be"]):
        return "internal_monologue"

    return "dialog"


def _compute_tone_direction(emotion, text):
    """Compute TTS voice direction string from emotion and text."""
    tone_map = {
        "angry": "loud and aggressive",
        "excited": "loud and excited",
        "happy": "warm and cheerful",
        "sad": "soft and somber",
        "tense": "low and tense",
        "determined": "firm and resolute",
        "dramatic": "intense and dramatic",
        "calm": "calm and steady",
        "surprised": "sharp and surprised",
        "melancholic": "slow and wistful",
        "neutral": "neutral and conversational",
    }

    direction = tone_map.get(emotion, "neutral and conversational")

    # Amplify for exclamations
    if text.count("!") >= 2:
        direction = "very " + direction

    return direction


# ---------------------------------------------------------------------------
# [5/9 continued] Recap detection
# ---------------------------------------------------------------------------

def detect_recaps(scenes, subtitle_segments, prev_episode_manifest=None):
    """
    Detect recap scenes at the start of the episode.
    Checks if opening dialogue duplicates content from the prior episode.
    """
    if not prev_episode_manifest:
        # Without prior episode data, use heuristic: first scene under 90s
        # with narration-style content is likely recap
        for scene in scenes:
            if scene["start_time"] > CONFIG["recap_max_duration"]:
                break
            # Check for typical recap narration cues
            scene_dialogue = _get_scene_dialogue(scene, subtitle_segments)
            for d in scene_dialogue:
                text_lower = d["text"].lower()
                if any(cue in text_lower for cue in [
                    "previously", "last time", "when we left off"
                ]):
                    scene["scene_type"] = "recap"
                    print(f"[Recap] Scene {scene['scene_number']} flagged as recap "
                          f"(narration cue)")
        return scenes

    # Compare against prior episode dialogue index
    prev_dialogue = prev_episode_manifest.get("dialogue_index", [])
    prev_texts = [d.get("content", "") for d in prev_dialogue]

    for scene in scenes:
        if scene["start_time"] > CONFIG["recap_max_duration"]:
            break

        scene_dialogue = _get_scene_dialogue(scene, subtitle_segments)
        match_count = 0
        total_lines = len(scene_dialogue)

        for d in scene_dialogue:
            for prev_text in prev_texts:
                if fuzzy_match_score(d["text"], prev_text) > CONFIG[
                    "recap_similarity_threshold"
                ]:
                    match_count += 1
                    break

        if total_lines > 0 and match_count / total_lines > 0.5:
            scene["scene_type"] = "recap"
            print(f"[Recap] Scene {scene['scene_number']} flagged as recap "
                  f"({match_count}/{total_lines} lines matched prior episode)")

    return scenes


# ---------------------------------------------------------------------------
# [8/9] Build output files
# ---------------------------------------------------------------------------

def build_scene_jsons(shots, scenes, dialogue_index, output_dir, arc_id=""):
    """Build per-scene JSON output files."""
    print(f"[8/9] Building {len(scenes)} per-scene JSON files...")

    shot_map = {s["shot_id"]: s for s in shots}

    # Build text_id → dialogue entry lookup
    dialogue_by_scene = {}
    for entry in dialogue_index:
        sn = entry["scene_number"]
        if sn not in dialogue_by_scene:
            dialogue_by_scene[sn] = []
        dialogue_by_scene[sn].append(entry)

    for scene in scenes:
        scene_num = scene["scene_number"]

        # Gather shots for this scene
        scene_shots = [shot_map[sid] for sid in scene["shot_ids"] if sid in shot_map]

        # Shot type sequence
        shot_type_sequence = [s.get("shot_type", "medium") for s in scene_shots]

        # Average motion intensity
        motion_values = [s.get("motion_intensity", 0) for s in scene_shots]
        avg_motion = round(sum(motion_values) / max(len(motion_values), 1), 3)

        # Text elements from dialogue index
        scene_dialogue = dialogue_by_scene.get(scene_num, [])
        text_elements = []
        for d in scene_dialogue:
            text_elements.append({
                "text_id": f"sc{scene_num}_d{d['seq']}",
                "type": d.get("type", "dialog"),
                "content": d["content"],
                "speaker": d.get("speaker"),
                "speaker_id": d.get("speaker_id"),
                "speaker_confidence": d.get("speaker_confidence", 0.0),
                "emotion": d.get("emotion", "neutral"),
                "tone_direction": d.get("tone_direction", ""),
                "start_time_seconds": d["start_time_seconds"],
                "end_time_seconds": d["end_time_seconds"],
            })

        # Sound effects (from high-motion, non-dialogue shots)
        sound_effects = _extract_scene_sfx(scene_shots, scene)

        # Determine if scene is anime-original
        origin_counts = {}
        for s in scene_shots:
            origin = s.get("content_origin", "unknown")
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
        anime_original = origin_counts.get("anime_original", 0) > len(scene_shots) * 0.7

        scene_data = {
            "scene_number": scene_num,
            "start_time_seconds": scene["start_time"],
            "end_time_seconds": scene["end_time"],
            "duration_seconds": scene["duration"],
            "shot_count": scene["shot_count"],
            "scene_type": scene.get("scene_type", "story"),
            "anime_original": anime_original,
            "keyframe_paths": scene.get("keyframe_paths", []),
            "scene_narrative": scene.get("scene_narrative", ""),
            "location": scene.get("location"),
            "time_context": scene.get("time_context", "present day"),
            "arc_id": arc_id,
            "establishing_shot_type": scene.get("establishing_shot_type", "none"),
            "emotional_tone": scene.get("emotional_tone", "calm"),
            "reconstruction_prompt": scene.get("reconstruction_prompt", ""),
            "shot_type_sequence": shot_type_sequence,
            "motion_intensity": avg_motion,
            "characters_present": scene.get("characters_present", []),
            "text_elements": text_elements,
            "sound_effects": sound_effects,
        }

        scene_path = os.path.join(output_dir, f"scene_{scene_num:03d}.json")
        save_json(scene_data, scene_path)

    print(f"[8/9] Saved {len(scenes)} scene files")


def _extract_scene_sfx(scene_shots, scene):
    """
    Infer sound effects from shot properties.
    High motion + no dialogue suggests action SFX.
    When audio classification is available, uses it to refine SFX category.
    """
    sfx = []
    sfx_seq = 0

    for shot in scene_shots:
        motion = shot.get("motion_intensity", 0)
        has_dialogue = shot.get("audio_features", {}).get("has_dialogue", False)
        audio_class = shot.get("audio_classification", {})
        audio_type = audio_class.get("audio_type", "unknown")

        if motion > 0.5 and not has_dialogue:
            sfx_seq += 1
            intensity = min(int(motion * 10), 10)

            # Refine category using audio classification when available
            if audio_type == "ambient":
                category = "environment"
                description = "ambient environment"
            elif motion > 0.7:
                category = "action"
                description = "action impact"
            else:
                category = "environment"
                description = "ambient movement"

            sfx.append({
                "sfx_id": f"sc{scene['scene_number']}_sfx_{sfx_seq}",
                "description": description,
                "intensity": intensity,
                "category": category,
                "timestamp_seconds": round(shot["start_time"], 3),
            })

    return sfx


def build_shot_index(shots, scenes, output_dir):
    """Build the shot_index.json — first-class output for alignment."""
    print("[8/9] Building shot index...")

    # Build shot-to-scene mapping
    shot_scene_map = {}
    shot_number_in_scene = {}
    for scene in scenes:
        for i, sid in enumerate(scene.get("shot_ids", [])):
            shot_scene_map[sid] = scene["scene_number"]
            shot_number_in_scene[sid] = i + 1

    shot_index = []
    for shot in shots:
        sid = shot["shot_id"]

        # Include audio_classification only if real data (not all defaults)
        audio_class = shot.get("audio_classification", {})
        include_audio_class = audio_class.get("audio_type") != "unknown"

        shot_entry = {
            "shot_id": sid,
            "scene_number": shot_scene_map.get(sid, 0),
            "shot_number_in_scene": shot_number_in_scene.get(sid, 0),
            "start_time_seconds": shot["start_time"],
            "end_time_seconds": shot["end_time"],
            "duration_seconds": shot["duration"],
            "shot_type": shot.get("shot_type", "medium"),
            "content_origin": shot.get("content_origin", "unknown"),
            "characters_present": shot.get("characters_present", []),
            "character_crops": shot.get("character_crops", []),
            "motion_intensity": shot.get("motion_intensity", 0.0),
            "background_embedding": shot.get("background_embedding", []),
            "background_plate_path": shot.get("background_plate_path"),
            "color_palette": shot.get("color_hex", []),
            "keyframe_paths": shot.get("keyframe_paths", []),
            "dialogue_in_shot": shot.get("dialogue_in_shot", []),
            "fade_type": shot.get("fade_type", "none"),
        }
        if include_audio_class:
            shot_entry["audio_classification"] = audio_class
        shot_index.append(shot_entry)

    save_json(shot_index, os.path.join(output_dir, "shot_index.json"))
    print(f"[8/9] Shot index: {len(shot_index)} entries")
    return shot_index


def build_character_registry(scenes, shots, dialogue_index, episode_id, output_dir):
    """Build character_registry.json tracking all characters across the episode."""
    print("[8/9] Building character registry...")

    characters = {}

    # From scene character lists
    shot_scene_map = {}
    for scene in scenes:
        for sid in scene.get("shot_ids", []):
            shot_scene_map[sid] = scene["scene_number"]

        for char in scene.get("characters_present", []):
            cid = char.get("character_id")
            if not cid:
                continue

            if cid not in characters:
                characters[cid] = {
                    "character_id": cid,
                    "name": char.get("name", cid),
                    "first_appearance_scene": scene["scene_number"],
                    "first_appearance_shot": scene["shot_ids"][0] if scene["shot_ids"] else None,
                    "appearances": [],
                    "shot_appearances": [],
                    "voice_profile": {
                        "observed_emotions": [],
                    },
                }

            if scene["scene_number"] not in characters[cid]["appearances"]:
                characters[cid]["appearances"].append(scene["scene_number"])

            # Add shot appearances
            for sid in scene.get("shot_ids", []):
                shot = next((s for s in shots if s["shot_id"] == sid), None)
                if shot and cid in shot.get("characters_present", []):
                    if sid not in characters[cid]["shot_appearances"]:
                        characters[cid]["shot_appearances"].append(sid)

    # Enrich with dialogue emotions
    for entry in dialogue_index:
        speaker_id = entry.get("speaker_id")
        emotion = entry.get("emotion")
        if speaker_id and speaker_id in characters and emotion:
            if emotion not in characters[speaker_id]["voice_profile"]["observed_emotions"]:
                characters[speaker_id]["voice_profile"]["observed_emotions"].append(emotion)

    registry = {
        "episode_id": episode_id,
        "characters": characters,
    }

    save_json(registry, os.path.join(output_dir, "character_registry.json"))
    print(f"[8/9] Character registry: {len(characters)} characters")
    return registry


# ---------------------------------------------------------------------------
# [9/9] Build episode manifest
# ---------------------------------------------------------------------------

def build_episode_manifest(episode_id, episode_title, scenes, shots, dialogue_index,
                           processing_start, output_dir, arc_id=""):
    """Build the episode_manifest.json — top-level episode summary."""
    print("[9/9] Building episode manifest...")

    total_duration = 0.0
    if shots:
        total_duration = shots[-1]["end_time"] - shots[0]["start_time"]

    # Scene summaries
    scene_summaries = []
    for scene in scenes:
        scene_shots = [s for s in shots if s["shot_id"] in scene.get("shot_ids", [])]
        motion_vals = [s.get("motion_intensity", 0) for s in scene_shots]
        avg_motion = round(sum(motion_vals) / max(len(motion_vals), 1), 3)

        scene_dialogue = [d for d in dialogue_index
                          if d["scene_number"] == scene["scene_number"]]

        scene_summaries.append({
            "scene_number": scene["scene_number"],
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "duration": scene["duration"],
            "scene_type": scene.get("scene_type", "story"),
            "shot_count": scene["shot_count"],
            "motion_intensity": avg_motion,
            "emotional_tone": scene.get("emotional_tone", "calm"),
            "character_count": len(scene.get("characters_present", [])),
            "dialog_line_count": len(scene_dialogue),
            "metadata_file": f"scene_{scene['scene_number']:03d}.json",
        })

    processing_time = time.time() - processing_start

    manifest = {
        "episode_id": episode_id,
        "episode_title": episode_title or f"Episode {episode_id}",
        "episode_number": int(episode_id) if episode_id.isdigit() else None,
        "arc_id": arc_id,
        "total_scenes": len(scenes),
        "total_shots": len(shots),
        "total_duration_seconds": round(total_duration, 3),
        "scenes": scene_summaries,
        "dialogue_index": dialogue_index,
        "processing_metadata": {
            "model_used": CONFIG["openai_model"],
            "subtitle_source": "embedded_track",
            "transnetv2_version": "2.0",
            "processing_date": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 1),
        },
    }

    save_json(manifest, os.path.join(output_dir, "episode_manifest.json"))
    print(f"[9/9] Episode manifest saved")
    print(f"       Scenes: {len(scenes)}, Shots: {len(shots)}, "
          f"Dialogue: {len(dialogue_index)} lines")
    print(f"       Duration: {total_duration:.1f}s, "
          f"Processing time: {processing_time:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(video_path, episode_id, episode_title=None,
                 chapter_manifest_path=None, prev_episode_manifest_path=None,
                 resume=False, use_yolo=False, use_clip=False, use_audio=False):
    """
    Run the full anime metadata extraction pipeline.

    Args:
        video_path: Path to episode video file (MKV or MP4)
        episode_id: Episode identifier (e.g. "0001")
        episode_title: Optional episode title
        chapter_manifest_path: Optional path to reference manga chapter manifest
                               for vocabulary grounding and content origin classification
        prev_episode_manifest_path: Optional path to previous episode manifest
                                    for recap detection
        resume: If True, skip stages whose output files already exist
        use_yolo: If True, run YOLOv8-seg for character crops and background plates
        use_clip: If True, run CLIP for background embeddings (scene grouping)
        use_audio: If True, extract audio and run librosa-based classification
                   per shot (music, ambient, silence detection for scene grouping)
    """
    CONFIG["resume"] = resume
    processing_start = time.time()

    output_dir = os.path.join(CONFIG["output_base"], episode_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  Anime Metadata Extraction Pipeline")
    print(f"{'=' * 70}")
    print(f"  Episode:   {episode_id}")
    print(f"  Video:     {os.path.basename(video_path)}")
    print(f"  Output:    {output_dir}")
    if chapter_manifest_path:
        print(f"  Manifest:  {chapter_manifest_path}")
    print(f"  Resume:    {'yes' if resume else 'no'}")
    print(f"{'=' * 70}")

    # Load reference data
    chapter_manifest = None
    if chapter_manifest_path:
        chapter_manifest = load_json(chapter_manifest_path)
        if chapter_manifest:
            print(f"Loaded chapter manifest: {chapter_manifest.get('chapter_id', '?')}, "
                  f"{len(chapter_manifest.get('dialogue_index', []))} dialogue lines")

    prev_manifest = None
    if prev_episode_manifest_path:
        prev_manifest = load_json(prev_episode_manifest_path)
        if prev_manifest:
            print(f"Loaded previous episode manifest: "
                  f"{prev_manifest.get('episode_id', '?')}")

    # Fetch wiki episode context for VLM grounding
    wiki_context = fetch_wiki_episode_context(episode_id)
    if wiki_context:
        print(f"Loaded wiki context: \"{wiki_context.get('episode_title', '?')}\", "
              f"{len(wiki_context.get('characters', []))} characters")
    elif HAS_WIKI_API:
        print("Wiki context: not available for this episode")

    # Compute arc_id: prefer wiki context, fall back to chapter manifest
    arc_id = ""
    if wiki_context and wiki_context.get("arc_id"):
        arc_id = wiki_context["arc_id"]
    elif chapter_manifest and chapter_manifest.get("arc_id"):
        arc_id = chapter_manifest["arc_id"]
    if arc_id:
        print(f"Arc ID: {arc_id}")

    stage_times = {}

    # --- Load GPU models if requested ---
    gpu_models = None
    if use_yolo or use_clip:
        from extraction.scene_detector import load_gpu_models
        gpu_models = load_gpu_models(use_yolo=use_yolo, use_clip=use_clip)
        if not gpu_models:
            print("Warning: GPU models requested but none loaded -- continuing without them")
            gpu_models = None

    # --- Stage 1: Subtitle extraction ---
    t = _stage_start(1)
    subtitle_segments = extract_subtitles(video_path, output_dir)
    stage_times[1] = _stage_end(1, t)

    # --- Stage 2: Shot detection ---
    t = _stage_start(2)
    shots = run_shot_detection(video_path, output_dir)
    stage_times[2] = _stage_end(2, t)

    # --- Extract audio track (if --use-audio) ---
    audio_path = None
    if use_audio:
        from extraction.scene_detector import extract_audio_track
        audio_path = extract_audio_track(video_path, output_dir)
        if audio_path:
            print(f"  Audio:     {os.path.basename(audio_path)}")
        else:
            print("  Audio:     extraction failed -- continuing without audio classification")

    # --- Stage 3: Per-shot feature extraction ---
    t = _stage_start(3)
    shots = run_feature_extraction(video_path, shots, subtitle_segments, output_dir,
                                   gpu_models=gpu_models, audio_path=audio_path)
    stage_times[3] = _stage_end(3, t)

    # --- Stage 4: Scene detection ---
    t = _stage_start(4)
    scenes = run_scene_detection(shots, output_dir)

    # Detect recaps
    scenes = detect_recaps(scenes, subtitle_segments, prev_manifest)

    # Tag flashback scenes
    for scene in scenes:
        if scene.get("has_flashback_candidate") and scene.get("scene_type") == "story":
            scene["scene_type"] = "flashback"
    stage_times[4] = _stage_end(4, t)

    # --- Stage 5: VLM scene analysis ---
    t = _stage_start(5)
    scenes = run_vlm_scene_analysis(
        video_path, shots, scenes, subtitle_segments, chapter_manifest, output_dir,
        wiki_context=wiki_context,
    )
    stage_times[5] = _stage_end(5, t)

    # --- Stage 6: Content origin classification ---
    t = _stage_start(6)
    shots = classify_content_origins(
        shots, scenes, subtitle_segments, chapter_manifest, output_dir
    )
    stage_times[6] = _stage_end(6, t)

    # --- Stage 7: Dialogue index ---
    t = _stage_start(7)
    dialogue_index = build_dialogue_index(shots, scenes, subtitle_segments, output_dir)
    stage_times[7] = _stage_end(7, t)

    # --- Stage 8: Output files ---
    t = _stage_start(8)
    build_scene_jsons(shots, scenes, dialogue_index, output_dir, arc_id=arc_id)
    build_shot_index(shots, scenes, output_dir)
    character_registry = build_character_registry(
        scenes, shots, dialogue_index, episode_id, output_dir
    )
    stage_times[8] = _stage_end(8, t)

    # --- Stage 9: Episode manifest ---
    t = _stage_start(9)
    manifest = build_episode_manifest(
        episode_id, episode_title, scenes, shots, dialogue_index,
        processing_start, output_dir, arc_id=arc_id
    )
    stage_times[9] = _stage_end(9, t)

    # --- Final summary ---
    total_elapsed = time.time() - processing_start
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE -- Episode {episode_id}")
    print(f"{'=' * 70}")
    print(f"  Output:    {output_dir}")
    print(f"  Scenes:    {len(scenes)}")
    print(f"  Shots:     {len(shots)}")
    print(f"  Dialogue:  {len(dialogue_index)} lines")
    print(f"{'-' * 70}")
    print(f"  Stage Timing:")
    for stage_num in sorted(stage_times):
        name = STAGE_NAMES.get(stage_num, "")
        elapsed = stage_times[stage_num]
        pct = elapsed / total_elapsed * 100 if total_elapsed > 0 else 0
        print(f"    [{stage_num}/9] {name:<42} {_fmt_elapsed(elapsed):>8}  ({pct:4.1f}%)")
    print(f"{'-' * 70}")
    print(f"    {'TOTAL':<47} {_fmt_elapsed(total_elapsed):>8}")
    print(f"{'=' * 70}")
    print(get_cost_summary())

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anime Metadata Extraction Pipeline"
    )
    parser.add_argument("video", help="Path to episode video file (MKV/MP4)")
    parser.add_argument("--episode-id", required=True,
                        help="Episode identifier (e.g. '0001')")
    parser.add_argument("--episode-title", default=None,
                        help="Episode title (optional)")
    parser.add_argument("--chapter-manifest", default=None,
                        help="Path to reference manga chapter_manifest.json "
                             "for vocabulary grounding")
    parser.add_argument("--prev-episode-manifest", default=None,
                        help="Path to previous episode's episode_manifest.json "
                             "for recap detection")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from cached intermediate files")
    parser.add_argument("--use-yolo", action="store_true",
                        help="Run YOLOv8-seg for character crops and background plates")
    parser.add_argument("--use-clip", action="store_true",
                        help="Run CLIP for background embeddings (improves scene grouping)")
    parser.add_argument("--use-audio", action="store_true",
                        help="Run audio classification per shot (music/ambient/silence "
                             "detection for improved scene grouping)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    run_pipeline(
        video_path=args.video,
        episode_id=args.episode_id,
        episode_title=args.episode_title,
        chapter_manifest_path=args.chapter_manifest,
        prev_episode_manifest_path=args.prev_episode_manifest,
        resume=args.resume,
        use_yolo=args.use_yolo,
        use_clip=args.use_clip,
        use_audio=args.use_audio,
    )
