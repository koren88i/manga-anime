import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import gradio as gr

OUTPUT_DIR = Path(r"c:\Users\User\helloworld_claude\mange-anime\output")
PROJECT_DIR = Path(r"c:\Users\User\helloworld_claude\mange-anime")

# Ensure project is importable
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# DATA HELPERS
# ============================================================
def load_manifest():
    path = OUTPUT_DIR / "manifest.json"
    if not path.exists():
        return {"speakers": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest):
    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_identification():
    path = OUTPUT_DIR / "character_identification.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_character_names():
    refs_dir = OUTPUT_DIR / "references"
    if refs_dir.exists():
        return sorted([d.name for d in refs_dir.iterdir() if d.is_dir()])
    return []


def get_speaker_frames(speaker_id):
    frames_dir = OUTPUT_DIR / "frames" / speaker_id
    if frames_dir.exists():
        return sorted([str(p) for p in frames_dir.glob("*.png")])
    return []


def get_speaker_crops(speaker_id):
    crops_dir = OUTPUT_DIR / "crops" / speaker_id
    if crops_dir.exists():
        return sorted([str(p) for p in crops_dir.glob("*.png")])
    return []


def get_reference_images(character_name):
    ref_dir = OUTPUT_DIR / "references" / character_name
    if ref_dir.exists():
        return sorted([str(p) for p in ref_dir.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    return []


MEDIA_DIR = PROJECT_DIR / "anime"

MKV_PATH = MEDIA_DIR / "[Anime Time] One Piece - 0001 - I'm Luffy! The Man Who's Gonna Be King of the Pirates!.mkv"

EPISODE_MKVS = {
    "ep0001": MEDIA_DIR / "[Anime Time] One Piece - 0001 - I'm Luffy! The Man Who's Gonna Be King of the Pirates!.mkv",
    "ep0002": MEDIA_DIR / "[Anime Time] One Piece - 0002 - Enter the Great Swordsman! Pirate Hunter Roronoa Zoro!.mkv",
    "ep0003": MEDIA_DIR / "[Anime Time] One Piece - 0003 - Morgan versus Luffy! Who's the Mysterious Pretty Girl.mkv",
}


def get_speaker_audio_clips(speaker_id):
    """Find audio clips for a speaker — check both Speaker_XX and renamed folder."""
    manifest = load_manifest()
    sdata = manifest["speakers"].get(speaker_id, {})
    char_name = sdata.get("character_name", speaker_id)

    speakers_dir = OUTPUT_DIR / "speakers"
    # Try renamed folder first, then original Speaker_XX
    for folder_name in [char_name, speaker_id]:
        clip_dir = speakers_dir / folder_name
        if clip_dir.exists():
            wavs = sorted(clip_dir.glob("*.wav"))
            if wavs:
                return wavs
    return []


def build_clip_choices(speaker_id):
    """Build dropdown choices for a speaker's audio clips."""
    if not speaker_id:
        return []
    manifest = load_manifest()
    sdata = manifest["speakers"].get(speaker_id, {})
    clips = sdata.get("clips", [])
    choices = []
    for i, clip in enumerate(clips):
        dur = clip["duration"]
        sub = ""
        if clip.get("subtitle_text"):
            sub = clip["subtitle_text"][0][:40]
        choices.append(f"{i}: [{clip['start']:.1f}s - {clip['end']:.1f}s] ({dur:.1f}s) {sub}")
    return choices


EPISODE_MP4 = OUTPUT_DIR / "episode.mp4"


def ensure_episode_mp4():
    """Remux MKV to browser-friendly MP4 (one-time, stream copy)."""
    if EPISODE_MP4.exists():
        return EPISODE_MP4
    print("Remuxing episode to MP4 (one-time)...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(MKV_PATH),
        "-map", "0:v:0", "-map", "0:a",
        "-c", "copy",
        "-sn",
        str(EPISODE_MP4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Remux failed: {result.stderr[:200]}")
        return None
    return EPISODE_MP4


def build_speaker_choices():
    manifest = load_manifest()
    choices = []
    for sid, sdata in manifest["speakers"].items():
        char = sdata.get("character_name", sid)
        dur = sdata.get("total_duration_seconds", 0)
        clips = sdata.get("num_clips", 0)
        label = sid
        if char and char != sid:
            label += f" -> {char}"
        label += f" ({dur}s, {clips} clips)"
        choices.append(label)
    return choices


def extract_speaker_id(choice_label):
    """Extract Speaker_XX from dropdown label like 'Speaker_11 -> Luffy (384.2s, 80 clips)'"""
    if not choice_label:
        return None
    return choice_label.split(" ")[0]


# ============================================================
# TAB 1: SPEAKER DASHBOARD
# ============================================================
def on_speaker_select(choice_label):
    speaker_id = extract_speaker_id(choice_label)
    if not speaker_id:
        return "", [], [], [], "", "", gr.Dropdown(choices=[], value=None)

    manifest = load_manifest()
    identification = load_identification()

    sdata = manifest["speakers"].get(speaker_id, {})
    char_name = sdata.get("character_name", speaker_id)

    # Summary
    summary = f"### {speaker_id}\n"
    summary += f"**Assigned Character:** {char_name}\n\n"
    summary += f"**Duration:** {sdata.get('total_duration_seconds', 0)}s | "
    summary += f"**Clips:** {sdata.get('num_clips', 0)} | "
    summary += f"**Original label:** {sdata.get('original_label', '?')}\n"

    # Frames
    frames = get_speaker_frames(speaker_id)

    # Crops
    crops = get_speaker_crops(speaker_id)

    # Similarity table
    id_data = identification.get(speaker_id, {})
    votes = id_data.get("votes", {})
    table_rows = []
    for cname, vdata in votes.items():
        table_rows.append([
            cname,
            vdata["vote_count"],
            round(vdata["avg_similarity"], 4),
            round(vdata["max_similarity"], 4),
            round(vdata["score"], 2),
        ])
    table_rows.sort(key=lambda r: r[4], reverse=True)

    # Subtitle samples
    sub_text = ""
    for clip in sdata.get("clips", [])[:5]:
        if clip.get("subtitle_text"):
            texts = " | ".join(clip["subtitle_text"])
            sub_text += f"- **[{clip['start']:.1f}s]** {texts}\n"
    if not sub_text:
        sub_text = "*(No subtitle text)*"

    # Clip choices for playback
    clip_choices = build_clip_choices(speaker_id)
    clip_dd_update = gr.Dropdown(choices=clip_choices, value=clip_choices[0] if clip_choices else None)

    return summary, frames, crops, table_rows, sub_text, char_name, clip_dd_update


def on_clip_select(choice_label, clip_choice):
    """Handle clip selection — find audio and seek episode video."""
    speaker_id = extract_speaker_id(choice_label)
    if not speaker_id or not clip_choice:
        return None, ""

    manifest = load_manifest()
    sdata = manifest["speakers"].get(speaker_id, {})
    clips = sdata.get("clips", [])

    idx = int(clip_choice.split(":")[0]) if clip_choice else 0
    if idx < 0 or idx >= len(clips):
        return None, ""

    clip = clips[idx]
    start = max(0, clip["start"] - 0.5)
    end = clip["end"] + 0.5

    # Find the audio clip file
    audio_path = None
    audio_clips = get_speaker_audio_clips(speaker_id)
    for wav in audio_clips:
        if f"{clip['start']:.2f}" in wav.name:
            audio_path = str(wav)
            break
    if not audio_path and idx < len(audio_clips):
        audio_path = str(audio_clips[idx])

    # Build HTML video player seeked to the right moment
    mp4 = ensure_episode_mp4()
    if not mp4:
        return audio_path, "<p>Episode MP4 not available</p>"
    video_url = "/file=" + str(mp4).replace("\\", "/")
    html = (
        f'<video src="{video_url}#t={start:.1f},{end:.1f}" controls autoplay '
        f'style="width:100%; max-height:400px;"></video>'
    )
    return audio_path, html


def on_save_assignment(choice_label, new_character):
    speaker_id = extract_speaker_id(choice_label)
    if not speaker_id or not new_character:
        return "Select a speaker and character"

    manifest = load_manifest()
    if speaker_id in manifest["speakers"]:
        manifest["speakers"][speaker_id]["character_name"] = new_character.strip()
        save_manifest(manifest)
        return f"Saved: {speaker_id} -> {new_character.strip()}"
    return "Error: speaker not found"


# ============================================================
# TAB 2: REFERENCE IMAGE MANAGER
# ============================================================
def on_char_select_refs(character_name):
    if not character_name:
        return []
    return get_reference_images(character_name)


def on_upload_reference(character_name, files):
    if not character_name or not files:
        return "Select a character and upload files", []

    ref_dir = OUTPUT_DIR / "references" / character_name
    ref_dir.mkdir(parents=True, exist_ok=True)

    existing = list(ref_dir.glob("*"))
    next_idx = len(existing) + 1

    for file_path in files:
        ext = Path(file_path).suffix.lower() or ".png"
        dest = ref_dir / f"ref_{next_idx:03d}{ext}"
        shutil.copy2(file_path, dest)
        next_idx += 1

    return f"Added {len(files)} images to {character_name}", get_reference_images(character_name)


def on_delete_reference(character_name, selected_index):
    if not character_name:
        return "Select a character", []
    images = get_reference_images(character_name)
    idx = int(selected_index) if selected_index is not None else -1
    if 0 <= idx < len(images):
        os.remove(images[idx])
        return f"Deleted image {idx}", get_reference_images(character_name)
    return f"Invalid index (0-{len(images)-1})", images


def on_create_character(new_name, files):
    if not new_name or not new_name.strip():
        return "Provide a character name", gr.Dropdown()
    new_name = new_name.strip()

    ref_dir = OUTPUT_DIR / "references" / new_name
    ref_dir.mkdir(parents=True, exist_ok=True)

    if files:
        for i, file_path in enumerate(files):
            ext = Path(file_path).suffix.lower() or ".png"
            dest = ref_dir / f"ref_{i + 1:03d}{ext}"
            shutil.copy2(file_path, dest)

    count = len(files) if files else 0
    return f"Created '{new_name}' with {count} images", gr.Dropdown(choices=get_character_names())


# ============================================================
# TAB 3: RE-RUN MATCHING
# ============================================================
def rerun_matching(redetect_faces):
    import sys
    sys.path.insert(0, str(PROJECT_DIR))
    from identify import (
        CONFIG as ID_CONFIG,
        encode_references,
        detect_faces_in_frames,
        match_faces_to_characters,
    )
    import torch

    log = []

    # Gather references from disk
    refs_dir = OUTPUT_DIR / "references"
    downloaded = {}
    for char_dir in refs_dir.iterdir():
        if char_dir.is_dir():
            imgs = [p for p in char_dir.glob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if imgs:
                downloaded[char_dir.name] = imgs

    log.append(f"References: {len(downloaded)} characters ({', '.join(downloaded.keys())})")

    config = dict(ID_CONFIG)
    config["character_names"] = list(downloaded.keys())

    # Encode references
    log.append("Encoding references with CLIP...")
    clip_model, preprocess, character_embeddings, device = encode_references(config, downloaded)
    log.append(f"  Encoded {len(character_embeddings)} characters on {device}")

    manifest = load_manifest()

    # Get face crops
    if redetect_faces:
        del clip_model
        torch.cuda.empty_cache()

        all_frames = {}
        for speaker_id in manifest["speakers"]:
            frames_dir = OUTPUT_DIR / "frames" / speaker_id
            if frames_dir.exists():
                all_frames[speaker_id] = sorted(frames_dir.glob("*.png"))
            else:
                all_frames[speaker_id] = []

        log.append("Re-detecting faces with YOLO (slow)...")
        all_crops = detect_faces_in_frames(config, all_frames)

        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            config["clip_model"], pretrained=config["clip_pretrained"]
        )
        clip_model = clip_model.to(device).eval()
    else:
        from PIL import Image
        log.append("Loading saved crops from disk...")
        all_crops = {}
        for speaker_id in manifest["speakers"]:
            crops = []
            for crop_path in sorted((OUTPUT_DIR / "crops" / speaker_id).glob("*.png")) if (OUTPUT_DIR / "crops" / speaker_id).exists() else []:
                img = Image.open(crop_path).convert("RGB")
                w, h = img.size
                crops.append({
                    "image": img,
                    "area": w * h,
                    "centrality": 0.8,
                    "confidence": 1.0,
                    "frame": crop_path.stem,
                })
            all_crops[speaker_id] = crops
            if crops:
                log.append(f"  {speaker_id}: {len(crops)} crops")

    # Match
    log.append("Matching faces to characters...")
    results = match_faces_to_characters(
        config, clip_model, preprocess, character_embeddings, device, all_crops
    )

    del clip_model
    torch.cuda.empty_cache()

    # Save results
    serializable = {}
    for sid, result in results.items():
        serializable[sid] = {
            "predicted_character": result["predicted_character"],
            "confidence": round(result["confidence"], 4),
            "votes": result["votes"],
        }
    with open(OUTPUT_DIR / "character_identification.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    for sid, result in results.items():
        if sid in manifest["speakers"]:
            manifest["speakers"][sid]["character_name"] = result["predicted_character"] or sid
    save_manifest(manifest)

    # Build results table
    table_rows = []
    for sid, result in sorted(results.items()):
        pred = result["predicted_character"] or "Unknown"
        conf = result["confidence"]
        table_rows.append([sid, pred, round(conf, 4)])

    log.append("Done! Manifest and identification results updated.")

    return "\n".join(log), table_rows


# ============================================================
# TAB 4: VOICE REGISTRY
# ============================================================
def load_voice_registry():
    path = OUTPUT_DIR / "voice_registry.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_voice_table():
    """Build voice registry table rows."""
    registry = load_voice_registry()
    if not registry:
        return []
    rows = []
    for vkey, vdata in registry.get("voices", {}).items():
        rows.append([
            vkey,
            vdata.get("display_name", vkey),
            vdata.get("source_speaker_id", "?"),
            vdata.get("num_clips", 0),
            vdata.get("total_audio_seconds", 0),
            len(vdata.get("reference_clips", {}).get("default", [])),
        ])
    rows.sort(key=lambda r: r[0])
    return rows


def on_voice_select(voice_key):
    """Show details for a selected voice."""
    if not voice_key:
        return "", None
    registry = load_voice_registry()
    if not registry:
        return "No voice registry found. Run: python voice_clone.py", None
    voices = registry.get("voices", {})
    if voice_key not in voices:
        return f"Voice '{voice_key}' not found", None
    v = voices[voice_key]

    info = f"### {v['display_name']} (`{voice_key}`)\n"
    info += f"**Source speaker:** {v['source_speaker_id']}\n\n"
    info += f"**Clips:** {v['num_clips']} | **Duration:** {v['total_audio_seconds']}s\n\n"

    ref_clips = v.get("reference_clips", {})
    info += "**Reference clips per emotion:**\n"
    for emotion, clips in ref_clips.items():
        info += f"- {emotion}: {len(clips)} clips\n"

    vp = v.get("voice_profile", {})
    if vp:
        info += f"\n**Voice style:** {vp.get('suggested_voice_style', 'N/A')}\n"
        emotions = vp.get("observed_emotions", [])
        if emotions:
            info += f"**Emotions:** {', '.join(emotions[:10])}\n"

    # Return first default reference clip for playback
    default_refs = ref_clips.get("default", [])
    audio_path = None
    if default_refs:
        folder = OUTPUT_DIR.parent / v["speaker_folder"]
        first_clip = folder / default_refs[0]
        if first_clip.exists():
            audio_path = str(first_clip)

    return info, audio_path


def on_rebuild_registry():
    """Rebuild the voice registry."""
    from voice_clone import build_voice_registry, CONFIG as VC_CONFIG
    try:
        registry = build_voice_registry(VC_CONFIG)
        n = len(registry["voices"])
        unmapped = registry.get("unmapped_manga_characters", [])
        msg = f"Registry rebuilt: {n} voices."
        if unmapped:
            msg += f"\nUnmapped manga characters: {', '.join(unmapped)}"
        return msg, build_voice_table()
    except Exception as e:
        return f"Error: {e}", []


def build_voice_key_choices():
    registry = load_voice_registry()
    if not registry:
        return []
    return sorted(registry.get("voices", {}).keys())


# ============================================================
# TAB 5: TTS PREVIEW
# ============================================================
def load_chapter_script():
    path = OUTPUT_DIR / "tts_scripts" / "0001" / "chapter_script.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_script_table():
    """Build a summary table of script lines."""
    script = load_chapter_script()
    if not script:
        return []
    rows = []
    for line in script.get("lines", []):
        rows.append([
            line["line_id"],
            line.get("character_id") or "narrator",
            line["voice_key"],
            line.get("emotion", ""),
            line["text_for_tts"][:60],
            line.get("speech_bubble_type", ""),
        ])
    return rows


def on_build_script():
    """Build chapter script from manga metadata."""
    from script_builder import build_chapter_script, CONFIG as SB_CONFIG
    try:
        script, path = build_chapter_script(SB_CONFIG)
        stats = script["statistics"]
        msg = (f"Script built: {script['total_lines']} lines\n"
               f"Dialog: {stats['dialog_lines']} | Narration: {stats['narration_lines']} | "
               f"Captions: {stats['caption_lines']}\n"
               f"Characters: {', '.join(stats['characters_used'])}")
        if stats["unmapped_speakers"]:
            msg += f"\nUnmapped: {', '.join(stats['unmapped_speakers'])}"
        return msg, build_script_table()
    except Exception as e:
        return f"Error: {e}", []


def on_generate_single_line(line_id):
    """Generate audio for a single script line."""
    if not line_id:
        return "Select a line first", None

    script = load_chapter_script()
    if not script:
        return "No chapter script found. Build it first.", None

    # Find the line
    target = None
    for line in script.get("lines", []):
        if line["line_id"] == line_id:
            target = line
            break
    if not target:
        return f"Line {line_id} not found", None

    try:
        from voice_clone import (
            load_tts_model, load_voice_registry, get_voice_entry,
            synthesize_line, apply_post_processing, CONFIG as VC_CONFIG,
        )
        import soundfile as sf

        registry = load_voice_registry(VC_CONFIG)
        voice_entry = get_voice_entry(registry, target["voice_key"])
        if not voice_entry:
            return f"No voice for {target['voice_key']}", None

        tts_model = load_tts_model(VC_CONFIG)
        audio = synthesize_line(
            tts_model, target["text_for_tts"], voice_entry,
            target.get("emotion", "neutral"), target.get("prosody_hints", {}),
            VC_CONFIG,
        )
        audio = apply_post_processing(
            audio, VC_CONFIG["output_sample_rate"],
            target.get("prosody_hints", {}),
            target.get("speech_bubble_type", "speech"),
        )

        # Save
        output_path = Path(target["output_file"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, VC_CONFIG["output_sample_rate"], subtype="PCM_16")

        import torch
        del tts_model
        torch.cuda.empty_cache()

        return f"Generated: {target['text_for_tts'][:60]}...", str(output_path)
    except Exception as e:
        return f"Error: {e}", None


def on_generate_full_chapter():
    """Generate all lines in the chapter script."""
    try:
        from tts_synthesize import synthesize_chapter, concatenate_chapter_audio, CONFIG as TTS_CONFIG
        synthesize_chapter(TTS_CONFIG)
        if TTS_CONFIG["concatenate_chapter"]:
            concatenate_chapter_audio(TTS_CONFIG)
        return "Full chapter synthesis complete!"
    except Exception as e:
        return f"Error: {e}"


def build_line_id_choices():
    script = load_chapter_script()
    if not script:
        return []
    choices = []
    for line in script.get("lines", []):
        char = line.get("character_id") or "narrator"
        text = line["text_for_tts"][:40]
        choices.append(f"{line['line_id']} | {char}: {text}")
    return choices


def extract_line_id(choice):
    if not choice:
        return None
    return choice.split(" | ")[0]


# ============================================================
# TAB 6: GROUND TRUTH REVIEW
# ============================================================

# -- Data loaders --

def list_available_ground_truth():
    """Scan output/ground_truth/ for available episode+source combinations."""
    gt_base = OUTPUT_DIR / "ground_truth"
    if not gt_base.exists():
        return [], []
    episodes = sorted([d.name for d in gt_base.iterdir() if d.is_dir()])
    sources = set()
    for ep_dir in gt_base.iterdir():
        if ep_dir.is_dir():
            for f in ep_dir.glob("assemblyai_*.json"):
                # e.g. assemblyai_raw_audio.json -> raw_audio
                source = f.stem.replace("assemblyai_", "")
                sources.add(source)
    return episodes, sorted(sources)


def load_ground_truth(episode_id, source):
    """Load ground truth JSON for an episode+source."""
    path = OUTPUT_DIR / "ground_truth" / episode_id / f"assemblyai_{source}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_gt_assignments(episode_id, source):
    """Load saved speaker assignments for an episode."""
    path = OUTPUT_DIR / "ground_truth" / episode_id / "speaker_assignments.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("source") == source:
                return data.get("assignments", {})
    return {}


def save_gt_assignments(episode_id, source, assignments):
    """Persist speaker assignments."""
    path = OUTPUT_DIR / "ground_truth" / episode_id / "speaker_assignments.json"
    data = {
        "episode_id": episode_id,
        "source": source,
        "assignments": assignments,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_manga_dialogue_index():
    """Load the manga dialogue_index from chapter_manifest.json."""
    path = OUTPUT_DIR / "manga_metadata" / "0001" / "chapter_manifest.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("dialogue_index", [])
    return []


# -- Text matching engine --

def _normalize_text(text):
    """Normalize text for fuzzy matching: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def recommend_speakers_from_text(gt_data, dialogue_index):
    """Match ground truth speakers against manga dialogue using fuzzy text matching.

    Returns dict: {speaker_label: {character, confidence, match_count, total_utterances, matches}}
    """
    if not dialogue_index or not gt_data:
        return {}

    # Pre-normalize manga lines
    manga_normalized = []
    for entry in dialogue_index:
        manga_normalized.append({
            "norm": _normalize_text(entry.get("content", "")),
            "speaker_id": entry.get("speaker_id"),
            "speaker": entry.get("speaker", ""),
            "content": entry.get("content", ""),
        })

    # Group utterances by speaker
    speakers = defaultdict(list)
    for utt in gt_data.get("utterances", []):
        speakers[utt["speaker"]].append(utt)

    results = {}
    for speaker_label, utterances in speakers.items():
        # Track which manga characters matched this speaker's lines
        char_matches = defaultdict(list)

        for utt in utterances:
            utt_norm = _normalize_text(utt.get("text", ""))
            if len(utt_norm) < 5:
                continue  # Skip very short utterances

            best_ratio = 0
            best_manga = None
            for manga in manga_normalized:
                if len(manga["norm"]) < 5:
                    continue
                ratio = SequenceMatcher(None, utt_norm, manga["norm"]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_manga = manga

            if best_ratio >= 0.4 and best_manga and best_manga["speaker_id"]:
                char_matches[best_manga["speaker_id"]].append({
                    "ratio": best_ratio,
                    "anime_text": utt.get("text", "")[:80],
                    "manga_text": best_manga["content"][:80],
                    "manga_speaker": best_manga["speaker"],
                })

        # Pick the character with the most matches
        if char_matches:
            ranked = sorted(char_matches.items(), key=lambda x: len(x[1]), reverse=True)
            best_char, best_matches = ranked[0]
            avg_conf = sum(m["ratio"] for m in best_matches) / len(best_matches)
            results[speaker_label] = {
                "character": best_char,
                "character_display": best_matches[0]["manga_speaker"],
                "confidence": round(avg_conf, 3),
                "match_count": len(best_matches),
                "total_utterances": len(utterances),
                "matches": best_matches[:5],  # Top 5 sample pairs
                "all_candidates": [
                    {"character": c, "count": len(m), "avg_ratio": round(sum(x["ratio"] for x in m) / len(m), 3)}
                    for c, m in ranked
                ],
            }
        else:
            results[speaker_label] = {
                "character": None,
                "character_display": None,
                "confidence": 0,
                "match_count": 0,
                "total_utterances": len(utterances),
                "matches": [],
                "all_candidates": [],
            }

    return results


# -- Video frame extraction --

def extract_gt_frame(episode_id, timestamp_sec):
    """Extract a video frame at the given timestamp. Returns path to cached PNG."""
    mkv_path = EPISODE_MKVS.get(episode_id)
    if not mkv_path or not mkv_path.exists():
        return None

    frames_dir = OUTPUT_DIR / "ground_truth" / episode_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Cache key: round to nearest 0.5s to avoid too many frames
    ts_key = round(timestamp_sec * 2) / 2
    frame_path = frames_dir / f"frame_{ts_key:.1f}.png"

    if frame_path.exists():
        return str(frame_path)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(mkv_path),
        "-vframes", "1",
        "-sn",
        "-q:v", "2",
        str(frame_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and frame_path.exists():
        return str(frame_path)
    return None


def extract_gt_video_clip(episode_id, start_sec, end_sec):
    """Extract a short H.264 video clip for browser playback. Returns path to MP4."""
    mkv_path = EPISODE_MKVS.get(episode_id)
    if not mkv_path or not mkv_path.exists():
        return None

    clips_dir = OUTPUT_DIR / "ground_truth" / episode_id / "video_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Cache key: round to 0.5s
    s_key = round(start_sec * 2) / 2
    e_key = round(end_sec * 2) / 2
    clip_path = clips_dir / f"clip_{s_key:.1f}_{e_key:.1f}.mp4"

    if clip_path.exists():
        return clip_path

    duration = end_sec - start_sec + 1.0  # add 1s padding
    ss = max(0, start_sec - 0.5)

    # Transcode to H.264 (browser-compatible), try GPU first then CPU
    for vcodec in ["h264_nvenc", "libx264"]:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ss:.3f}",
            "-i", str(mkv_path),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", vcodec,
            "-c:a", "aac",
            "-preset", "fast" if vcodec == "h264_nvenc" else "ultrafast",
            "-crf", "23",
            "-sn",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and clip_path.exists():
            return clip_path

    return None


# -- Ground truth state (shared across callbacks via gr.State) --

_gt_cache = {}  # Mutable cache for ground truth data within session


def _gt_speaker_stats(gt_data):
    """Compute per-speaker stats from ground truth data."""
    speakers = defaultdict(lambda: {"time": 0.0, "count": 0})
    for utt in gt_data.get("utterances", []):
        spk = utt["speaker"]
        dur = utt["end_sec"] - utt["start_sec"]
        speakers[spk]["time"] += dur
        speakers[spk]["count"] += 1
    return dict(speakers)


# -- Event handlers --

def on_gt_load(episode_id, source):
    """Load ground truth data and run recommendations. Returns overview table + speaker choices."""
    if not episode_id or not source:
        return [], [], "", gr.Dropdown(choices=[], value=None)

    gt_data = load_ground_truth(episode_id, source)
    if not gt_data:
        return [], [], f"No ground truth found for {episode_id}/{source}", gr.Dropdown(choices=[], value=None)

    # Cache for other callbacks
    _gt_cache["data"] = gt_data
    _gt_cache["episode"] = episode_id
    _gt_cache["source"] = source

    # Load existing assignments
    assignments = load_gt_assignments(episode_id, source)
    _gt_cache["assignments"] = assignments

    # Run text matching recommendations
    dialogue_index = load_manga_dialogue_index()
    recommendations = recommend_speakers_from_text(gt_data, dialogue_index)
    _gt_cache["recommendations"] = recommendations

    # Build speaker overview table
    stats = _gt_speaker_stats(gt_data)
    table_rows = []
    for spk in sorted(stats.keys()):
        s = stats[spk]
        rec = recommendations.get(spk, {})
        rec_char = rec.get("character_display") or rec.get("character") or "-"
        rec_conf = rec.get("confidence", 0)
        assigned = assignments.get(spk, {})
        status = assigned.get("status", "unreviewed")
        assigned_char = assigned.get("character", "")

        mins = int(s["time"] // 60)
        secs = int(s["time"] % 60)
        table_rows.append([
            spk,
            s["count"],
            f"{mins}m {secs:02d}s",
            rec_char,
            f"{rec_conf:.2f}" if rec_conf > 0 else "-",
            assigned_char or "-",
            status,
        ])

    # Build speaker dropdown choices
    speaker_choices = []
    for spk in sorted(stats.keys()):
        s = stats[spk]
        rec = recommendations.get(spk, {})
        rec_char = rec.get("character_display") or ""
        label = f"Speaker {spk} ({s['count']} utt"
        if rec_char:
            label += f", rec: {rec_char}"
        label += ")"
        speaker_choices.append(label)

    info = f"Loaded {episode_id}/{source}: {len(stats)} speakers, {len(gt_data.get('utterances', []))} utterances"

    return table_rows, speaker_choices, info, gr.Dropdown(choices=speaker_choices, value=None)


def on_gt_speaker_select(speaker_choice):
    """Show detail for selected speaker: recommendation breakdown + utterance list."""
    if not speaker_choice or "data" not in _gt_cache:
        return "", [], gr.Dropdown(choices=[], value=None), ""

    # Extract speaker label from choice like "Speaker A (41 utt, rec: Luffy)"
    speaker_label = speaker_choice.split(" ")[1] if " " in speaker_choice else speaker_choice

    gt_data = _gt_cache["data"]
    recommendations = _gt_cache.get("recommendations", {})
    assignments = _gt_cache.get("assignments", {})

    # Speaker utterances
    utterances = [u for u in gt_data.get("utterances", []) if u["speaker"] == speaker_label]

    # Build summary markdown
    rec = recommendations.get(speaker_label, {})
    assigned = assignments.get(speaker_label, {})
    total_time = sum(u["end_sec"] - u["start_sec"] for u in utterances)

    md = f"### Speaker {speaker_label}\n"
    md += f"**Utterances:** {len(utterances)} | **Speaking time:** {total_time / 60:.1f} min\n\n"

    if rec.get("character"):
        md += f"**Recommended:** {rec.get('character_display', rec['character'])} "
        md += f"(confidence: {rec['confidence']:.2f}, {rec['match_count']}/{rec['total_utterances']} lines matched)\n\n"

        if rec.get("all_candidates"):
            md += "**All candidates:**\n"
            for cand in rec["all_candidates"][:5]:
                md += f"- {cand['character']}: {cand['count']} matches (avg {cand['avg_ratio']:.2f})\n"
            md += "\n"

        if rec.get("matches"):
            md += "**Sample text matches:**\n"
            for m in rec["matches"][:3]:
                md += f"- Anime: *\"{m['anime_text']}\"*\n"
                md += f"  Manga ({m['manga_speaker']}): *\"{m['manga_text']}\"* ({m['ratio']:.2f})\n"
    else:
        md += "**Recommended:** No match found (may be narrator, OP/ED song, or minor character)\n"

    if assigned.get("character"):
        md += f"\n**Current assignment:** {assigned['character']} ({assigned.get('status', '?')})\n"

    # Build match detail table
    match_table = []
    if rec.get("all_candidates"):
        for cand in rec["all_candidates"]:
            match_table.append([cand["character"], cand["count"], cand["avg_ratio"]])

    # Build utterance dropdown
    utt_choices = []
    for i, utt in enumerate(utterances):
        m = int(utt["start_sec"] // 60)
        s = int(utt["start_sec"] % 60)
        text = utt.get("text", "")[:50]
        utt_choices.append(f"{i}: [{m}:{s:02d}] {text}")

    # Pre-fill character assignment
    prefill = assigned.get("character") or rec.get("character_display") or rec.get("character") or ""

    return md, match_table, gr.Dropdown(choices=utt_choices, value=utt_choices[0] if utt_choices else None), prefill


def on_gt_utterance_select(speaker_choice, utt_choice):
    """Show video frame + episode playback for selected utterance."""
    if not speaker_choice or not utt_choice or "data" not in _gt_cache:
        return None, "", ""

    speaker_label = speaker_choice.split(" ")[1] if " " in speaker_choice else speaker_choice
    episode_id = _gt_cache["episode"]
    gt_data = _gt_cache["data"]

    utterances = [u for u in gt_data.get("utterances", []) if u["speaker"] == speaker_label]
    idx = int(utt_choice.split(":")[0]) if utt_choice else 0
    if idx < 0 or idx >= len(utterances):
        return None, "", ""

    utt = utterances[idx]
    midpoint = (utt["start_sec"] + utt["end_sec"]) / 2

    # Extract frame
    frame_path = extract_gt_frame(episode_id, midpoint)

    # Extract short video clip (H.264, browser-compatible)
    start = max(0, utt["start_sec"] - 1.0)
    end = utt["end_sec"] + 1.0
    clip_path = extract_gt_video_clip(episode_id, start, end)
    video_html = ""
    if clip_path:
        video_url = "/file=" + str(clip_path).replace("\\", "/")
        video_html = (
            f'<video src="{video_url}" controls autoplay '
            f'style="width:100%; max-height:400px;"></video>'
        )
    else:
        video_html = "<p>Video clip extraction failed</p>"

    # Utterance text display
    text_md = f"**Speaker {speaker_label}** [{utt['start_sec']:.1f}s - {utt['end_sec']:.1f}s]\n\n"
    text_md += f"> {utt.get('text', '')}\n"

    return frame_path, video_html, text_md


def on_gt_save(speaker_choice, character_name, episode_id, source):
    """Save character assignment for a speaker."""
    if not speaker_choice or not character_name or not episode_id or not source:
        return "Select a speaker and enter a character name"

    speaker_label = speaker_choice.split(" ")[1] if " " in speaker_choice else speaker_choice
    character_name = character_name.strip()

    assignments = _gt_cache.get("assignments", {})
    rec = _gt_cache.get("recommendations", {}).get(speaker_label, {})

    assignments[speaker_label] = {
        "character": character_name,
        "status": "confirmed",
        "recommended": rec.get("character"),
        "recommendation_confidence": rec.get("confidence", 0),
    }
    _gt_cache["assignments"] = assignments
    save_gt_assignments(episode_id, source, assignments)

    return f"Saved: Speaker {speaker_label} -> {character_name}"


# ============================================================
# BUILD APP
# ============================================================
def build_app():
    try:
        manifest = load_manifest()
        has_manifest = True
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = {"speakers": {}}
        has_manifest = False

    with gr.Blocks(title="Anime Character Validation") as app:
        gr.Markdown("# Anime Character Identification — Validation UI")
        if has_manifest:
            gr.Markdown(f"**Source:** {manifest.get('source_file', '?')} | "
                        f"**Speakers:** {len(manifest['speakers'])}")
        else:
            gr.Markdown("**Note:** manifest.json not found (diarization pipeline not yet complete). "
                        "Ground Truth Review tab is available.")

        # ---- TAB 1: Speaker Dashboard ----
        with gr.Tab("Speaker Dashboard"):
            with gr.Row():
                with gr.Column(scale=1):
                    speaker_dd = gr.Dropdown(
                        label="Select Speaker",
                        choices=build_speaker_choices(),
                    )
                    summary_md = gr.Markdown()
                    char_dd = gr.Dropdown(
                        label="Assign Character",
                        choices=get_character_names() + ["Unknown"],
                        allow_custom_value=True,
                    )
                    save_btn = gr.Button("Save Assignment", variant="primary")
                    save_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("#### Extracted Frames")
                    frames_gal = gr.Gallery(label="Frames", columns=5, height=200,
                                            object_fit="contain")
                    gr.Markdown("#### Detected Face Crops")
                    crops_gal = gr.Gallery(label="Face Crops", columns=6, height=180,
                                           object_fit="contain")

                with gr.Column(scale=1):
                    gr.Markdown("#### Similarity Scores")
                    scores_df = gr.Dataframe(
                        headers=["Character", "Votes", "Avg Sim", "Max Sim", "Score"],
                        datatype=["str", "number", "number", "number", "number"],
                    )
                    gr.Markdown("#### Sample Dialogue")
                    subtitle_md = gr.Markdown()

            # ---- Playback Section ----
            gr.Markdown("---")
            gr.Markdown("#### Episode Playback")
            with gr.Row():
                clip_dd = gr.Dropdown(label="Select Clip", choices=[], scale=3)
            with gr.Row():
                audio_player = gr.Audio(label="Speaker Audio (isolated vocals)",
                                        type="filepath")
                video_player = gr.HTML(label="Episode Video Clip")

            # ---- Event Handlers ----
            speaker_dd.change(
                fn=on_speaker_select,
                inputs=[speaker_dd],
                outputs=[summary_md, frames_gal, crops_gal, scores_df, subtitle_md,
                         char_dd, clip_dd],
            )
            clip_dd.change(
                fn=on_clip_select,
                inputs=[speaker_dd, clip_dd],
                outputs=[audio_player, video_player],
            )
            save_btn.click(
                fn=on_save_assignment,
                inputs=[speaker_dd, char_dd],
                outputs=[save_status],
            )

        # ---- TAB 2: Reference Images ----
        with gr.Tab("Reference Images"):
            with gr.Row():
                with gr.Column():
                    ref_char_dd = gr.Dropdown(
                        label="Select Character",
                        choices=get_character_names(),
                    )
                    ref_gal = gr.Gallery(label="Current References", columns=3,
                                         height=250, object_fit="contain")
                    with gr.Row():
                        del_idx = gr.Number(label="Image index to delete", precision=0)
                        del_btn = gr.Button("Delete", variant="stop", size="sm")
                    del_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    gr.Markdown("#### Upload to Existing Character")
                    upload_files = gr.File(label="Images", file_count="multiple",
                                           file_types=["image"])
                    upload_btn = gr.Button("Add Images", variant="primary")
                    upload_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    gr.Markdown("#### Create New Character")
                    new_name = gr.Textbox(label="Character Name")
                    new_files = gr.File(label="Reference Images", file_count="multiple",
                                        file_types=["image"])
                    create_btn = gr.Button("Create", variant="primary")
                    create_status = gr.Textbox(label="Status", interactive=False)

            ref_char_dd.change(fn=on_char_select_refs, inputs=[ref_char_dd],
                               outputs=[ref_gal])
            del_btn.click(fn=on_delete_reference, inputs=[ref_char_dd, del_idx],
                          outputs=[del_status, ref_gal])
            upload_btn.click(fn=on_upload_reference, inputs=[ref_char_dd, upload_files],
                             outputs=[upload_status, ref_gal])
            create_btn.click(fn=on_create_character, inputs=[new_name, new_files],
                             outputs=[create_status, ref_char_dd])

        # ---- TAB 3: Re-run Matching ----
        with gr.Tab("Re-run Matching"):
            gr.Markdown(
                "### Re-run Character Matching\n"
                "After editing reference images or adding characters, click below to "
                "re-encode and re-match all speakers against updated references."
            )
            redetect_cb = gr.Checkbox(
                label="Also re-detect faces with YOLO (slow, ~2-3 min)",
                value=False,
            )
            rerun_btn = gr.Button("Re-run Matching", variant="primary")
            progress_log = gr.Textbox(label="Progress", lines=12, interactive=False)
            results_df = gr.Dataframe(
                headers=["Speaker", "Character", "Confidence"],
                label="Updated Results",
            )

            rerun_btn.click(
                fn=rerun_matching,
                inputs=[redetect_cb],
                outputs=[progress_log, results_df],
            )

        # ---- TAB 4: Voice Registry ----
        with gr.Tab("Voice Registry"):
            gr.Markdown(
                "### Voice Registry\n"
                "Maps speakers to TTS voice clones. Rebuild after changing character assignments."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    voice_table = gr.Dataframe(
                        value=build_voice_table(),
                        headers=["Voice Key", "Display Name", "Source Speaker",
                                 "Clips", "Duration (s)", "Default Refs"],
                        datatype=["str", "str", "str", "number", "number", "number"],
                        label="Registered Voices",
                    )
                    rebuild_btn = gr.Button("Rebuild Voice Registry", variant="primary")
                    rebuild_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("#### Voice Details")
                    voice_dd = gr.Dropdown(
                        label="Select Voice",
                        choices=build_voice_key_choices(),
                    )
                    voice_info_md = gr.Markdown()
                    voice_audio = gr.Audio(label="Reference Clip Sample", type="filepath")

            rebuild_btn.click(
                fn=on_rebuild_registry,
                outputs=[rebuild_status, voice_table],
            )
            voice_dd.change(
                fn=on_voice_select,
                inputs=[voice_dd],
                outputs=[voice_info_md, voice_audio],
            )

        # ---- TAB 5: TTS Preview ----
        with gr.Tab("TTS Preview"):
            gr.Markdown(
                "### Text-to-Speech Preview\n"
                "Build a TTS script from manga metadata, then generate audio per line or full chapter."
            )
            with gr.Row():
                build_script_btn = gr.Button("Build Chapter Script", variant="primary")
                script_status = gr.Textbox(label="Status", interactive=False, scale=3)

            script_table = gr.Dataframe(
                value=build_script_table(),
                headers=["Line ID", "Character", "Voice Key", "Emotion", "Text", "Bubble Type"],
                datatype=["str", "str", "str", "str", "str", "str"],
                label="Chapter Script Lines",
            )

            gr.Markdown("---")
            gr.Markdown("#### Generate Single Line")
            with gr.Row():
                line_dd = gr.Dropdown(
                    label="Select Line",
                    choices=build_line_id_choices(),
                    scale=3,
                )
                gen_single_btn = gr.Button("Generate", variant="primary")
            with gr.Row():
                gen_single_status = gr.Textbox(label="Status", interactive=False)
                gen_single_audio = gr.Audio(label="Generated Audio", type="filepath")

            gr.Markdown("---")
            gr.Markdown("#### Generate Full Chapter")
            gen_full_btn = gr.Button("Generate All Lines", variant="primary")
            gen_full_status = gr.Textbox(label="Status", lines=3, interactive=False)

            build_script_btn.click(
                fn=on_build_script,
                outputs=[script_status, script_table],
            )
            gen_single_btn.click(
                fn=lambda choice: on_generate_single_line(extract_line_id(choice)),
                inputs=[line_dd],
                outputs=[gen_single_status, gen_single_audio],
            )
            gen_full_btn.click(
                fn=on_generate_full_chapter,
                outputs=[gen_full_status],
            )

        # ---- TAB 6: Ground Truth Review ----
        with gr.Tab("Ground Truth Review"):
            gr.Markdown(
                "### Ground Truth Speaker Validation\n"
                "Review AssemblyAI diarization results. Text matching against the manga script "
                "provides character recommendations. Confirm by checking video frames and audio."
            )

            available_eps, available_sources = list_available_ground_truth()

            # Row 1: Controls
            with gr.Row():
                gt_episode_dd = gr.Dropdown(
                    label="Episode",
                    choices=available_eps,
                    value=available_eps[0] if available_eps else None,
                    scale=1,
                )
                gt_source_dd = gr.Dropdown(
                    label="Audio Source",
                    choices=available_sources,
                    value=available_sources[0] if available_sources else None,
                    scale=1,
                )
                gt_load_btn = gr.Button("Load & Recommend", variant="primary", scale=1)
                gt_load_status = gr.Textbox(label="Status", interactive=False, scale=2)

            # Row 2: Speaker overview table
            gt_overview_df = gr.Dataframe(
                headers=["Speaker", "Utterances", "Time", "Recommended", "Conf", "Assigned", "Status"],
                datatype=["str", "number", "str", "str", "str", "str", "str"],
                label="Speaker Overview",
            )

            gr.Markdown("---")

            # Row 3: Speaker detail
            with gr.Row():
                # Left: Speaker info + assignment
                with gr.Column(scale=1):
                    gt_speaker_dd = gr.Dropdown(label="Select Speaker", choices=[])
                    gt_speaker_md = gr.Markdown()
                    gt_match_df = gr.Dataframe(
                        headers=["Character", "Matches", "Avg Similarity"],
                        datatype=["str", "number", "number"],
                        label="Text Match Candidates",
                    )
                    gt_char_input = gr.Textbox(label="Assign Character Name", placeholder="e.g. luffy, narrator, op_song")
                    gt_save_btn = gr.Button("Save Assignment", variant="primary")
                    gt_save_status = gr.Textbox(label="Save Status", interactive=False)

                # Right: Utterance review
                with gr.Column(scale=2):
                    gt_utt_dd = gr.Dropdown(label="Select Utterance", choices=[])
                    gt_utt_text = gr.Markdown()
                    with gr.Row():
                        gt_frame_img = gr.Image(label="Video Frame", height=300)
                        gt_video_html = gr.HTML(label="Episode Playback")

            # Hidden state for passing episode/source to save handler
            gt_ep_state = gr.State("")
            gt_src_state = gr.State("")

            # ---- Event wiring ----

            def _gt_load_wrapper(episode, source):
                table, choices, info, dd = on_gt_load(episode, source)
                return table, info, dd, episode, source

            gt_load_btn.click(
                fn=_gt_load_wrapper,
                inputs=[gt_episode_dd, gt_source_dd],
                outputs=[gt_overview_df, gt_load_status, gt_speaker_dd, gt_ep_state, gt_src_state],
            )

            gt_speaker_dd.change(
                fn=on_gt_speaker_select,
                inputs=[gt_speaker_dd],
                outputs=[gt_speaker_md, gt_match_df, gt_utt_dd, gt_char_input],
            )

            gt_utt_dd.change(
                fn=on_gt_utterance_select,
                inputs=[gt_speaker_dd, gt_utt_dd],
                outputs=[gt_frame_img, gt_video_html, gt_utt_text],
            )

            gt_save_btn.click(
                fn=on_gt_save,
                inputs=[gt_speaker_dd, gt_char_input, gt_ep_state, gt_src_state],
                outputs=[gt_save_status],
            )

    return app


if __name__ == "__main__":
    app = build_app()
    # Include ground truth dirs in allowed_paths for video/frame serving
    gt_dir = OUTPUT_DIR / "ground_truth"
    allowed = [str(OUTPUT_DIR), str(PROJECT_DIR)]
    if gt_dir.exists():
        allowed.append(str(gt_dir))
    app.launch(server_port=7860, share=False, theme=gr.themes.Soft(), allowed_paths=allowed)
