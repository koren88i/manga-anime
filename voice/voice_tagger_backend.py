"""
Voice Tagger Backend — VLM-based recommendations, audio extraction, quality scoring.

Uses pre-computed GPT-4o Vision scene analysis from the anime_metadata pipeline
for character identification instead of CLIP/YOLO face matching.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import glob as glob_mod
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# PATHS & CONFIG
# ============================================================
PROJECT_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUTPUT_DIR = PROJECT_DIR / "output"
VOICE_TAGS_DIR = OUTPUT_DIR / "voice_tags"
ANIME_METADATA_DIR = OUTPUT_DIR / "anime_metadata"
REFERENCES_DIR = OUTPUT_DIR / "references"

MEDIA_DIR = PROJECT_DIR / "anime"

EPISODE_MKVS = {
    "ep0001": MEDIA_DIR / "[Anime Time] One Piece - 0001 - I'm Luffy! The Man Who's Gonna Be King of the Pirates!.mkv",
    "ep0002": MEDIA_DIR / "[Anime Time] One Piece - 0002 - Enter the Great Swordsman! Pirate Hunter Roronoa Zoro!.mkv",
    "ep0003": MEDIA_DIR / "[Anime Time] One Piece - 0003 - Morgan versus Luffy! Who's the Mysterious Pretty Girl.mkv",
}

SCAN_CONFIG = {
    "target_seconds": 90,          # 90s per character goal
    "audio_track": 0,              # Japanese audio (track 0)
}

# Background jobs tracker
_jobs = {}
_jobs_lock = threading.Lock()


# ============================================================
# VLM SCENE DATA — load pre-computed GPT-4o scene analysis
# ============================================================
_vlm_scene_cache = {}  # episode_id -> list of scene dicts


def _ep_num(episode_id):
    """Convert 'ep0001' -> '0001'."""
    return episode_id.replace("ep", "")


def load_vlm_scenes(episode_id):
    """Load scene_*.json files from the anime_metadata pipeline output."""
    if episode_id in _vlm_scene_cache:
        return _vlm_scene_cache[episode_id]

    ep_num = _ep_num(episode_id)
    scene_dir = ANIME_METADATA_DIR / ep_num
    if not scene_dir.exists():
        return []

    scenes = []
    for path in sorted(scene_dir.glob("scene_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                scenes.append(json.load(f))
        except Exception:
            continue

    _vlm_scene_cache[episode_id] = scenes
    return scenes


def get_scene_at_time(episode_id, time_sec):
    """Find the scene that contains the given timestamp."""
    scenes = load_vlm_scenes(episode_id)
    for scene in scenes:
        if scene["start_time_seconds"] <= time_sec <= scene["end_time_seconds"]:
            return scene
    return None


def get_full_scene_at_time(episode_id, time_sec):
    """Return the full scene dict at the given timestamp, with generic characters filtered."""
    scene = get_scene_at_time(episode_id, time_sec)
    if not scene:
        return None

    # Return a copy with characters filtered to real names
    result = dict(scene)
    result["characters_present"] = [
        c for c in scene.get("characters_present", [])
        if _is_real_character(c.get("name", c.get("character_id", "")))
    ]
    return result


# ============================================================
# DATA MODEL — tags, face scans, clusters
# ============================================================
def _ep_dir(episode_id):
    d = VOICE_TAGS_DIR / episode_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_tags(episode_id):
    path = _ep_dir(episode_id) / "tags.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tags": [], "character_totals": {}, "dismissed": []}


def save_tags(episode_id, data):
    # Recompute character_totals
    totals = defaultdict(lambda: {"total_seconds": 0, "tag_count": 0})
    for tag in data["tags"]:
        ch = tag["character"]
        dur = tag.get("end_sec", 0) - tag.get("start_sec", 0)
        totals[ch]["total_seconds"] += dur
        totals[ch]["tag_count"] += 1
    for ch, info in totals.items():
        info["total_seconds"] = round(info["total_seconds"], 1)
        info["status"] = "ready" if info["total_seconds"] >= SCAN_CONFIG["target_seconds"] else "in_progress"
    data["character_totals"] = dict(totals)
    path = _ep_dir(episode_id) / "tags.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_tag(episode_id, character, start_sec, end_sec, subtitle_text="", face_cluster_id=None, confidence="manual"):
    data = load_tags(episode_id)
    tag_id = f"tag_{uuid.uuid4().hex[:8]}"
    tag = {
        "tag_id": tag_id,
        "character": character,
        "start_sec": round(start_sec, 2),
        "end_sec": round(end_sec, 2),
        "duration_sec": round(end_sec - start_sec, 2),
        "subtitle_text": subtitle_text,
        "confidence": confidence,
        "face_cluster_id": face_cluster_id,
    }
    data["tags"].append(tag)
    save_tags(episode_id, data)
    return tag


def delete_tag(episode_id, tag_id):
    data = load_tags(episode_id)
    data["tags"] = [t for t in data["tags"] if t["tag_id"] != tag_id]
    save_tags(episode_id, data)


def update_tag(episode_id, tag_id, updates):
    data = load_tags(episode_id)
    for tag in data["tags"]:
        if tag["tag_id"] == tag_id:
            tag.update(updates)
            tag["duration_sec"] = round(tag["end_sec"] - tag["start_sec"], 2)
            break
    save_tags(episode_id, data)


def dismiss_recommendation(episode_id, start_sec, end_sec, character):
    """Add a subtitle+character pair to the dismissed list."""
    data = load_tags(episode_id)
    dismissed = data.setdefault("dismissed", [])
    dismissed.append({
        "start_sec": round(start_sec, 2),
        "end_sec": round(end_sec, 2),
        "character": character,
    })
    save_tags(episode_id, data)


# ============================================================
# SUBTITLE PARSING (reuse speaker_diarization pattern)
# ============================================================
def extract_subtitles(episode_id):
    """Extract ASS subtitles from MKV and parse them."""
    mkv_path = EPISODE_MKVS.get(episode_id)
    if not mkv_path or not mkv_path.exists():
        return []

    ep_dir = _ep_dir(episode_id)
    subs_path = ep_dir / "subtitles.ass"

    if not subs_path.exists():
        # Extract subtitle track from MKV (track 0 = Japanese subs typically)
        cmd = [
            "ffmpeg", "-y", "-i", str(mkv_path),
            "-map", "0:s:0", str(subs_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Subtitle extraction failed: {result.stderr[:200]}")
            return []

    return parse_ass_subtitles(subs_path)


def parse_ass_subtitles(subs_path):
    """Parse ASS subtitle file — reuses pattern from speaker_diarization.py."""
    if subs_path is None or not Path(subs_path).exists():
        return []

    import ass

    with open(subs_path, "r", encoding="utf-8-sig") as f:
        doc = ass.parse(f)

    events = []
    for event in doc.events:
        if event.TYPE == "Dialogue":
            # Strip ASS formatting tags like {\an8}, {\pos(...)}, etc.
            import re
            clean_text = re.sub(r"\{[^}]*\}", "", event.text)
            clean_text = clean_text.replace("\\N", " ").strip()
            if clean_text:
                events.append({
                    "start": round(event.start.total_seconds(), 3),
                    "end": round(event.end.total_seconds(), 3),
                    "text": clean_text,
                    "style": event.style,
                })

    return events


# ============================================================
# EPISODE PREP — remux MKV to MP4 for browser playback
# ============================================================
def ensure_episode_mp4(episode_id):
    """Remux MKV to browser-friendly MP4 (stream copy, one-time)."""
    mkv_path = EPISODE_MKVS.get(episode_id)
    if not mkv_path or not mkv_path.exists():
        return None

    mp4_path = _ep_dir(episode_id) / "episode.mp4"
    if mp4_path.exists():
        return mp4_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(mkv_path),
        "-map", "0:v:0",
        "-map", f"0:a:{SCAN_CONFIG['audio_track']}",
        "-c", "copy", "-sn",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Remux failed: {result.stderr[:300]}")
        return None
    return mp4_path


# ============================================================
# ON-DEMAND FRAME ANALYSIS — VLM scene lookup
# ============================================================
def analyze_frame(episode_id, time_sec):
    """Look up which characters are present at the given time using VLM scene data."""
    scene = get_scene_at_time(episode_id, time_sec)
    if not scene:
        return {"characters": [], "scene": None}

    characters = []
    for char in scene.get("characters_present", []):
        name = char.get("name", char.get("character_id", "Unknown"))
        if not _is_real_character(name):
            continue
        characters.append({
            "name": name,
            "action": char.get("action", ""),
            "emotion": char.get("emotion", ""),
        })

    return {
        "characters": characters,
        "scene": {
            "scene_number": scene.get("scene_number"),
            "narrative": scene.get("scene_narrative", ""),
            "location": scene.get("location"),
            "time_range": [scene["start_time_seconds"], scene["end_time_seconds"]],
        },
    }


# ============================================================
# RECOMMENDATION ENGINE — VLM character data per scene
# ============================================================
def get_recommendations(episode_id):
    """Generate tag suggestions from pre-computed GPT-4o scene analysis."""
    subtitles = extract_subtitles(episode_id)
    scenes = load_vlm_scenes(episode_id)
    tags_data = load_tags(episode_id)

    if not subtitles or not scenes:
        return []

    existing_tags = tags_data.get("tags", [])

    recommendations = []
    for sub in subtitles:
        sub_start = sub["start"]
        sub_end = sub["end"]
        sub_mid = (sub_start + sub_end) / 2

        # Skip if already tagged
        already_tagged = False
        for tag in existing_tags:
            if tag["start_sec"] <= sub_start and tag["end_sec"] >= sub_end:
                already_tagged = True
                break
        if already_tagged:
            continue

        # Find the scene containing this subtitle
        scene = None
        for s in scenes:
            if s["start_time_seconds"] <= sub_mid <= s["end_time_seconds"]:
                scene = s
                break
        if not scene:
            continue

        raw_chars = scene.get("characters_present", [])
        chars = [c for c in raw_chars
                 if _is_real_character(c.get("name", c.get("character_id", "")))]
        if not chars:
            continue

        # Single character in scene → high confidence recommendation
        if len(chars) == 1:
            char = chars[0]
            name = char.get("name", char.get("character_id", "Unknown"))
            recommendations.append({
                "start_sec": sub_start,
                "end_sec": sub_end,
                "subtitle_text": sub["text"],
                "character": name,
                "score": 0.9,
                "reason": "only character in scene",
            })
        else:
            # Multiple characters — check dialogue text_elements for speaker hints
            best_char = None
            best_score = 0

            # Check if the pipeline already assigned a speaker for this time range
            for te in scene.get("text_elements", []):
                if te.get("type") != "dialog":
                    continue
                te_start = te.get("start_time_seconds", 0)
                te_end = te.get("end_time_seconds", 0)
                # Match subtitle to text_element by time overlap
                if abs(te_start - sub_start) < 1.0 and abs(te_end - sub_end) < 1.0:
                    speaker = te.get("speaker")
                    conf = te.get("speaker_confidence", 0)
                    if speaker and speaker != "Unknown" and _is_real_character(speaker) and conf > 0:
                        best_char = speaker
                        best_score = min(conf, 0.85)
                    break

            # Fallback: if only 2 characters, lower-confidence guess
            if not best_char and len(chars) == 2:
                best_char = chars[0].get("name", chars[0].get("character_id"))
                best_score = 0.4

            if best_char and best_score >= 0.3:
                recommendations.append({
                    "start_sec": sub_start,
                    "end_sec": sub_end,
                    "subtitle_text": sub["text"],
                    "character": best_char,
                    "score": round(best_score, 3),
                    "reason": f"{len(chars)} characters in scene",
                })

    recommendations.sort(key=lambda r: r["score"], reverse=True)

    # Tag correction pass: use existing tags to improve recommendations
    if existing_tags:
        _apply_tag_corrections(recommendations, existing_tags, scenes)

    # Filter out dismissed recommendations
    dismissed = tags_data.get("dismissed", [])
    if dismissed:
        dismissed_set = {(d["start_sec"], d["end_sec"], d["character"]) for d in dismissed}
        recommendations = [r for r in recommendations
                           if (r["start_sec"], r["end_sec"], r["character"]) not in dismissed_set]

    return recommendations


def _apply_tag_corrections(recommendations, existing_tags, scenes):
    """Use existing manual tags to improve recommendations via process of elimination.

    Only applies to scenes with 2-3 characters. If user tagged lines as character A
    in a scene with characters A and B, remaining untagged lines probably belong to B.
    """
    # Build map: scene_number -> set of characters the user has tagged in that scene
    scene_char_tags = defaultdict(set)
    for tag in existing_tags:
        mid = (tag["start_sec"] + tag["end_sec"]) / 2
        for s in scenes:
            if s["start_time_seconds"] <= mid <= s["end_time_seconds"]:
                scene_char_tags[s["scene_number"]].add(tag["character"])
                break

    if not scene_char_tags:
        return

    # Build scene lookup by number
    scene_by_num = {s["scene_number"]: s for s in scenes}

    for rec in recommendations:
        mid = (rec["start_sec"] + rec["end_sec"]) / 2
        # Find which scene this rec is in
        rec_scene = None
        for s in scenes:
            if s["start_time_seconds"] <= mid <= s["end_time_seconds"]:
                rec_scene = s
                break
        if not rec_scene:
            continue

        scene_num = rec_scene["scene_number"]
        tagged_chars = scene_char_tags.get(scene_num)
        if not tagged_chars:
            continue  # No tags in this scene, nothing to propagate

        # Get real characters present in the scene
        chars_present = set()
        for c in rec_scene.get("characters_present", []):
            name = c.get("name", c.get("character_id", ""))
            if _is_real_character(name):
                chars_present.add(name)

        # Only apply for 2-3 character scenes
        if len(chars_present) < 2 or len(chars_present) > 3:
            continue

        # Process of elimination
        untagged_chars = chars_present - tagged_chars

        if len(untagged_chars) == 1:
            remaining_char = list(untagged_chars)[0]
            # Only override weak suggestions — preserve strong VLM attributions
            if rec["score"] < 0.5 and rec["character"] in tagged_chars:
                rec["character"] = remaining_char
                rec["score"] = 0.55
                rec["reason"] = f"elimination: user tagged {', '.join(sorted(tagged_chars))}"
            elif rec["character"] == remaining_char:
                # Elimination agrees — small boost
                rec["score"] = min(rec["score"] + 0.05, 0.85)
            # else: keep strong VLM attribution as-is

        elif len(untagged_chars) == 0:
            # All characters have been tagged in this scene — use text_element speaker
            for te in rec_scene.get("text_elements", []):
                te_start = te.get("start_time_seconds", 0)
                te_end = te.get("end_time_seconds", 0)
                if abs(te_start - rec["start_sec"]) < 1.0 and abs(te_end - rec["end_sec"]) < 1.0:
                    te_speaker = te.get("speaker", "")
                    if te_speaker in tagged_chars:
                        rec["character"] = te_speaker
                        rec["score"] = min(rec["score"] + 0.1, 0.85)
                        rec["reason"] = f"tag-confirmed speaker"
                    break

    # Re-sort after corrections
    recommendations.sort(key=lambda r: r["score"], reverse=True)


# ============================================================
# AUDIO EXTRACTION — ffmpeg + optional Demucs
# ============================================================
def extract_voice_sample(episode_id, tag_id, use_demucs=False):
    """Extract audio for a tagged time range."""
    tags_data = load_tags(episode_id)
    tag = next((t for t in tags_data["tags"] if t["tag_id"] == tag_id), None)
    if not tag:
        return None

    mkv_path = EPISODE_MKVS.get(episode_id)
    if not mkv_path:
        return None

    character = tag["character"]
    samples_dir = _ep_dir(episode_id) / "voice_samples" / character
    samples_dir.mkdir(parents=True, exist_ok=True)

    start = tag["start_sec"]
    end = tag["end_sec"]
    filename = f"{tag_id}_{start:.1f}-{end:.1f}.wav"
    output_path = samples_dir / filename

    # Extract audio clip with ffmpeg
    audio_track = SCAN_CONFIG["audio_track"]
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(mkv_path),
        "-map", f"0:a:{audio_track}",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "22050",
        "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Audio extraction failed: {result.stderr[:200]}")
        return None

    if use_demucs and output_path.exists():
        output_path = _run_demucs_on_clip(output_path, samples_dir)

    return str(output_path)


def _run_demucs_on_clip(clip_path, output_dir):
    """Run Demucs vocal isolation on a single clip."""
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model("htdemucs_ft")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wav, sr = torchaudio.load(str(clip_path))
    wav = wav.unsqueeze(0).to(device)
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate

    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=False)

    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx]

    # Overwrite the original clip with vocals-only version
    torchaudio.save(str(clip_path), vocals.cpu(), sr)

    del model, sources, wav
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return clip_path


# ============================================================
# VOICE CLONE EVALUATION — XTTS synthesis + quality scoring
# ============================================================
def evaluate_voice_clone(character):
    """Synthesize test lines and score voice clone quality."""
    # Collect all samples for this character across episodes
    all_samples = []
    total_duration = 0

    for ep_id in EPISODE_MKVS:
        samples_dir = VOICE_TAGS_DIR / ep_id / "voice_samples" / character
        if samples_dir.exists():
            import soundfile as sf
            for wav_path in sorted(samples_dir.glob("*.wav")):
                try:
                    info = sf.info(str(wav_path))
                    all_samples.append(str(wav_path))
                    total_duration += info.duration
                except Exception:
                    continue

    if not all_samples:
        return {
            "character": character,
            "status": "no_samples",
            "total_samples": 0,
            "total_duration": 0,
            "message": "No voice samples found. Tag some voice segments first.",
        }

    if total_duration < 6:
        return {
            "character": character,
            "status": "insufficient",
            "total_samples": len(all_samples),
            "total_duration": round(total_duration, 1),
            "message": f"Only {total_duration:.1f}s of audio. Need at least 6s for voice cloning.",
        }

    # Select best reference clips (up to 24s total)
    import soundfile as sf
    scored = []
    for wav_path in all_samples:
        info = sf.info(wav_path)
        dur = info.duration
        # Prefer clips in 2-10s range
        if 2 <= dur <= 10:
            score = 1.0
        elif dur < 2:
            score = dur / 2.0
        else:
            score = 10.0 / dur
        scored.append((wav_path, dur, score))
    scored.sort(key=lambda x: x[2], reverse=True)

    ref_clips = []
    ref_duration = 0
    for wav_path, dur, _ in scored:
        ref_clips.append(wav_path)
        ref_duration += dur
        if ref_duration >= 24:
            break

    # Synthesize test line
    test_text = "I am going to become the king of the pirates! Nobody can stop me!"

    try:
        from TTS.api import TTS
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        eval_dir = VOICE_TAGS_DIR / "_evaluations" / character
        eval_dir.mkdir(parents=True, exist_ok=True)
        synth_path = eval_dir / "test_synthesis.wav"

        tts.tts_to_file(
            text=test_text,
            speaker_wav=ref_clips[0],
            language="en",
            file_path=str(synth_path),
        )

        # Compute speaker similarity using ECAPA-TDNN
        similarity = _compute_speaker_similarity(ref_clips[0], str(synth_path))

        del tts
        if torch.cuda.is_available():  # noqa: torch imported above
            torch.cuda.empty_cache()

        # Quality thresholds
        if similarity >= 0.85:
            quality = "excellent"
        elif similarity >= 0.75:
            quality = "good"
        elif similarity >= 0.65:
            quality = "acceptable"
        else:
            quality = "poor"

        status = "ready" if similarity >= 0.65 and total_duration >= SCAN_CONFIG["target_seconds"] else "needs_work"

        return {
            "character": character,
            "status": status,
            "quality": quality,
            "speaker_similarity": round(similarity, 3),
            "total_samples": len(all_samples),
            "total_duration": round(total_duration, 1),
            "ref_clips_used": len(ref_clips),
            "ref_duration": round(ref_duration, 1),
            "synth_path": str(synth_path),
            "message": f"Speaker similarity: {similarity:.2f} ({quality}). "
                       f"{total_duration:.0f}s of samples across {len(all_samples)} clips.",
        }

    except Exception as e:
        return {
            "character": character,
            "status": "error",
            "total_samples": len(all_samples),
            "total_duration": round(total_duration, 1),
            "message": f"Voice clone evaluation failed: {e}",
        }


def _compute_speaker_similarity(ref_path, synth_path):
    """Compute speaker similarity using Resemblyzer (lightweight alternative to ECAPA-TDNN)."""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()
        ref_wav = preprocess_wav(ref_path)
        synth_wav = preprocess_wav(synth_path)
        ref_emb = encoder.embed_utterance(ref_wav)
        synth_emb = encoder.embed_utterance(synth_wav)
        similarity = float(np.dot(ref_emb, synth_emb))
        return similarity
    except ImportError:
        pass

    # Fallback: use speechbrain ECAPA-TDNN (already in the project)
    try:
        from speechbrain.inference import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        ref_emb = classifier.encode_batch(classifier.load_audio(ref_path)).squeeze()
        synth_emb = classifier.encode_batch(classifier.load_audio(synth_path)).squeeze()
        import torch
        similarity = float(torch.nn.functional.cosine_similarity(ref_emb, synth_emb, dim=0))
        del classifier
        return similarity
    except Exception as e:
        print(f"Speaker similarity computation failed: {e}")
        return 0.0


# ============================================================
# BACKGROUND JOB MANAGEMENT
# ============================================================
def start_job(func, *args, **kwargs):
    """Start a background job and return its ID."""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    with _jobs_lock:
        _jobs[job_id] = {"status": "starting", "progress": 0, "total": 0, "message": "Starting..."}

    def _run():
        try:
            func(*args, job_id=job_id, **kwargs)
        except Exception as e:
            _update_job(job_id, "error", error=str(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return job_id


def _update_job(job_id, status, **kwargs):
    if job_id is None:
        return
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id]["status"] = status
        _jobs[job_id].update(kwargs)


def get_job_status(job_id):
    with _jobs_lock:
        return dict(_jobs.get(job_id, {"status": "not_found"}))


# ============================================================
# CHARACTER LIST (from VLM scene data)
# ============================================================
_GENERIC_NAMES = {
    "unknown", "character", "narrator", "crew", "crew member", "person",
    "man", "woman", "boy", "girl", "villager", "pirate", "marine", "soldier",
    "guard", "captain", "companion", "crowd", "group members", "another character",
    "other character", "young character", "large character", "smaller character",
    "young boy", "young girl", "narrating character", "straw hat character",
    "leader of the group",
}

_GENERIC_PREFIXES = (
    "character ", "unknown", "pirate ", "sailor ", "crew ",
    "pink-haired ", "blue-haired ", "green-haired ",
)


def _is_real_character(name):
    """Filter out generic/placeholder character names from VLM output."""
    if not name:
        return False
    lower = name.lower().strip()
    if lower in _GENERIC_NAMES:
        return False
    for prefix in _GENERIC_PREFIXES:
        if lower.startswith(prefix):
            return False
    return True


def get_enriched_subtitles(episode_id):
    """Pre-match every subtitle to its VLM text_element and compute tiers.

    Returns a flat list where each subtitle carries:
      - scene context (number, type, location, characters)
      - vlm_match (speaker, confidence, emotion, tone_direction)
      - existing tag (if any)
      - tier (1=auto-confirmable, 2=needs review, 3=manual)
      - suggested_character + score + reason
    """
    subtitles = extract_subtitles(episode_id)
    scenes = load_vlm_scenes(episode_id)
    tags_data = load_tags(episode_id)
    existing_tags = tags_data.get("tags", [])
    dismissed = tags_data.get("dismissed", [])

    if not subtitles:
        return []

    # Build dismissed set for filtering
    dismissed_set = {(d["start_sec"], d["end_sec"], d["character"]) for d in dismissed}

    # Build tag lookup: subtitle start/end -> tag
    tag_lookup = {}
    for tag in existing_tags:
        tag_lookup[(round(tag["start_sec"], 2), round(tag["end_sec"], 2))] = tag

    # Build scene index for fast lookup
    scene_list = sorted(scenes, key=lambda s: s["start_time_seconds"])

    def find_scene(time_sec):
        for s in scene_list:
            if s["start_time_seconds"] <= time_sec <= s["end_time_seconds"]:
                return s
        return None

    # Build enriched list
    enriched = []
    for idx, sub in enumerate(subtitles):
        sub_start = sub["start"]
        sub_end = sub["end"]
        sub_mid = (sub_start + sub_end) / 2

        # Find scene
        scene = find_scene(sub_mid)

        # Scene characters (filtered)
        scene_chars = []
        scene_number = None
        scene_type = None
        scene_location = None
        if scene:
            scene_number = scene.get("scene_number")
            scene_type = scene.get("scene_type", "story")
            scene_location = scene.get("location", "")
            for c in scene.get("characters_present", []):
                name = c.get("name", c.get("character_id", ""))
                if _is_real_character(name):
                    scene_chars.append({
                        "name": name,
                        "action": c.get("action", ""),
                        "emotion": c.get("emotion", ""),
                    })

        # Find VLM text_element match by timestamp proximity
        vlm_match = None
        if scene:
            best_te = None
            best_dist = 999
            for te in scene.get("text_elements", []):
                if te.get("type") != "dialog":
                    continue
                te_start = te.get("start_time_seconds", 0)
                dist = abs(te_start - sub_start)
                if dist < 1.5 and dist < best_dist:
                    best_dist = dist
                    best_te = te
            if best_te:
                speaker = best_te.get("speaker", "")
                if speaker and speaker != "Unknown" and _is_real_character(speaker):
                    vlm_match = {
                        "speaker": speaker,
                        "confidence": best_te.get("speaker_confidence", 0),
                        "emotion": best_te.get("emotion", ""),
                        "tone_direction": best_te.get("tone_direction", ""),
                    }

        # Check existing tag
        tag_key = (round(sub_start, 2), round(sub_end, 2))
        existing_tag = tag_lookup.get(tag_key)
        # Also check with tolerance
        if not existing_tag:
            for tag in existing_tags:
                if abs(tag["start_sec"] - sub_start) < 0.5 and abs(tag["end_sec"] - sub_end) < 0.5:
                    existing_tag = tag
                    break

        # Compute suggestion + tier
        suggested_character = None
        suggestion_score = 0
        suggestion_reason = ""

        # Auto-tag OP/ED/eyecatch
        if scene_type in ("opening", "ending"):
            label = "Opening Song" if scene_type == "opening" else "Ending Song"
            suggested_character = label
            suggestion_score = 0.95
            suggestion_reason = f"{scene_type} section"
        elif scene_type == "eyecatch":
            suggested_character = "Narrator"
            suggestion_score = 0.90
            suggestion_reason = "eyecatch section"
        elif vlm_match and vlm_match["confidence"] >= 0.8:
            suggested_character = vlm_match["speaker"]
            suggestion_score = vlm_match["confidence"]
            suggestion_reason = "VLM speaker attribution"
        elif len(scene_chars) == 1:
            suggested_character = scene_chars[0]["name"]
            suggestion_score = 0.90
            suggestion_reason = "only character in scene"
        elif vlm_match and vlm_match["confidence"] >= 0.4:
            suggested_character = vlm_match["speaker"]
            suggestion_score = vlm_match["confidence"]
            suggestion_reason = f"VLM speaker ({len(scene_chars)} chars in scene)"
        elif len(scene_chars) == 2 and vlm_match:
            suggested_character = vlm_match["speaker"]
            suggestion_score = max(vlm_match["confidence"], 0.4)
            suggestion_reason = "2-character scene VLM guess"

        # Filter dismissed suggestions
        if suggested_character and (round(sub_start, 2), round(sub_end, 2), suggested_character) in dismissed_set:
            suggested_character = None
            suggestion_score = 0
            suggestion_reason = ""

        # Assign tier
        if suggestion_score >= 0.8:
            tier = 1
        elif suggestion_score >= 0.4:
            tier = 2
        else:
            tier = 3

        enriched.append({
            "index": idx,
            "start": sub_start,
            "end": sub_end,
            "text": sub["text"],
            "style": sub.get("style", ""),
            "scene_number": scene_number,
            "scene_type": scene_type,
            "scene_location": scene_location,
            "scene_chars": scene_chars,
            "vlm_match": vlm_match,
            "tag": {
                "tag_id": existing_tag["tag_id"],
                "character": existing_tag["character"],
                "confidence": existing_tag.get("confidence", "manual"),
            } if existing_tag else None,
            "tier": tier,
            "suggested_character": suggested_character,
            "suggestion_score": round(suggestion_score, 3),
            "suggestion_reason": suggestion_reason,
        })

    # Apply elimination corrections
    _apply_enriched_corrections(enriched, existing_tags, scenes)

    return enriched


def _apply_enriched_corrections(enriched, existing_tags, scenes):
    """Use existing tags + partial tagging to improve enriched suggestions via elimination."""
    if not existing_tags and not any(e["tag"] for e in enriched):
        return

    # Build map: scene_number -> set of tagged characters
    scene_char_tags = defaultdict(set)
    for tag in existing_tags:
        mid = (tag["start_sec"] + tag["end_sec"]) / 2
        for s in scenes:
            if s["start_time_seconds"] <= mid <= s["end_time_seconds"]:
                scene_char_tags[s["scene_number"]].add(tag["character"])
                break

    if not scene_char_tags:
        return

    # Scene lookup
    scene_by_num = {}
    for s in scenes:
        scene_by_num[s["scene_number"]] = s

    for item in enriched:
        if item["tag"]:
            continue  # Already tagged, skip

        sn = item["scene_number"]
        if sn is None or sn not in scene_char_tags:
            continue

        scene = scene_by_num.get(sn)
        if not scene:
            continue

        tagged_chars = scene_char_tags[sn]
        chars_present = set()
        for c in scene.get("characters_present", []):
            name = c.get("name", c.get("character_id", ""))
            if _is_real_character(name):
                chars_present.add(name)

        if len(chars_present) < 2 or len(chars_present) > 3:
            continue

        untagged_chars = chars_present - tagged_chars
        if len(untagged_chars) == 1:
            remaining = list(untagged_chars)[0]
            # Only override weak/absent suggestions — preserve strong VLM attributions
            if not item["suggested_character"] or item["suggestion_score"] < 0.5:
                item["suggested_character"] = remaining
                item["suggestion_score"] = 0.55
                item["suggestion_reason"] = f"elimination: others tagged ({', '.join(sorted(tagged_chars))})"
                item["tier"] = 2
            elif item["suggested_character"] == remaining:
                # Elimination agrees with existing suggestion — small boost
                item["suggestion_score"] = min(item["suggestion_score"] + 0.05, 0.95)
            # else: keep existing strong VLM suggestion as-is


def bulk_accept_tags(episode_id, accepts):
    """Accept multiple subtitle tags at once.

    accepts: list of {start_sec, end_sec, character, subtitle_text}
    Returns count of tags created.
    """
    data = load_tags(episode_id)
    count = 0
    for accept in accepts:
        # Check not already tagged
        already = False
        for tag in data["tags"]:
            if abs(tag["start_sec"] - accept["start_sec"]) < 0.5 and abs(tag["end_sec"] - accept["end_sec"]) < 0.5:
                already = True
                break
        if already:
            continue

        tag_id = f"tag_{uuid.uuid4().hex[:8]}"
        tag = {
            "tag_id": tag_id,
            "character": accept["character"],
            "start_sec": round(accept["start_sec"], 2),
            "end_sec": round(accept["end_sec"], 2),
            "duration_sec": round(accept["end_sec"] - accept["start_sec"], 2),
            "subtitle_text": accept.get("subtitle_text", ""),
            "confidence": "auto_accepted",
            "face_cluster_id": None,
        }
        data["tags"].append(tag)
        count += 1

    if count > 0:
        save_tags(episode_id, data)
    return count


def get_character_names():
    """Collect all character names from VLM scene data across all episodes."""
    names = set()
    for ep_id in EPISODE_MKVS:
        scenes = load_vlm_scenes(ep_id)
        for scene in scenes:
            for char in scene.get("characters_present", []):
                name = char.get("name", char.get("character_id", ""))
                if _is_real_character(name):
                    names.add(name)
    # Also include characters from existing tags
    for ep_id in EPISODE_MKVS:
        tags_data = load_tags(ep_id)
        for tag in tags_data.get("tags", []):
            if tag.get("character"):
                names.add(tag["character"])
    # Always include special non-character tags
    names.add("Narrator")
    names.add("Opening Song")
    names.add("Ending Song")
    names.add("Extra")
    return sorted(names)


def get_episode_info(episode_id):
    """Get basic episode info (duration, etc.)."""
    mkv_path = EPISODE_MKVS.get(episode_id)
    if not mkv_path or not mkv_path.exists():
        return None

    cap = cv2.VideoCapture(str(mkv_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    return {
        "episode_id": episode_id,
        "duration_sec": round(duration, 1),
        "fps": round(fps, 2),
        "total_frames": total_frames,
    }


# ============================================================
# DIARIZATION — load results and bulk-accept tags
# ============================================================
def get_diarization_status(episode_id):
    """Return cached diarization result dict, or {status: not_found}."""
    result_path = VOICE_TAGS_DIR / episode_id / "diarization" / "result.json"
    if not result_path.exists():
        return {"status": "not_found", "groups": [], "total_groups": 0}
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def bulk_accept_diarization(episode_id, accepts):
    """Batch-tag subtitle lines from confirmed diarization speaker groups.

    accepts: list of {cluster_id, character}
    Returns count of tags created.
    """
    diar_data = get_diarization_status(episode_id)
    if diar_data.get("status") != "done":
        return 0

    subtitles = extract_subtitles(episode_id)
    group_by_id = {g["cluster_id"]: g for g in diar_data.get("groups", [])}
    tags_data = load_tags(episode_id)
    count = 0

    for accept in accepts:
        cluster_id = accept.get("cluster_id")
        character = accept.get("character")
        group = group_by_id.get(cluster_id)
        if not group or not character:
            continue

        for sub_idx in group["subtitle_indices"]:
            if sub_idx >= len(subtitles):
                continue
            sub = subtitles[sub_idx]

            # Skip already-tagged
            already = any(
                abs(t["start_sec"] - sub["start"]) < 0.5
                and abs(t["end_sec"] - sub["end"]) < 0.5
                for t in tags_data["tags"]
            )
            if already:
                continue

            tags_data["tags"].append({
                "tag_id": f"tag_{uuid.uuid4().hex[:8]}",
                "character": character,
                "start_sec": round(sub["start"], 2),
                "end_sec": round(sub["end"], 2),
                "duration_sec": round(sub["end"] - sub["start"], 2),
                "subtitle_text": sub["text"],
                "confidence": "diarization",
                "face_cluster_id": cluster_id,
            })
            count += 1

    if count > 0:
        save_tags(episode_id, tags_data)
    return count
