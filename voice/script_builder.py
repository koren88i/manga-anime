import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import re
import time
from pathlib import Path

_PROJECT_ROOT = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Input
    "manga_metadata_dir": _PROJECT_ROOT / "output" / "manga_metadata" / "0001",
    "voice_registry_path": _PROJECT_ROOT / "output" / "voice_registry.json",

    # Output
    "output_dir": _PROJECT_ROOT / "output" / "tts_scripts" / "0001",
    "tts_output_dir": _PROJECT_ROOT / "output" / "tts_output" / "0001",

    # Chapter info
    "chapter_id": "0001",

    # What to include
    "include_narration": True,
    "include_sfx_descriptions": False,
    "include_captions": True,
    "include_internal_monologue": True,

    # Pacing
    "default_pause_between_panels_ms": 400,
    "default_pause_between_pages_ms": 800,
}


# ============================================================
# PROSODY DERIVATION
# ============================================================
BUBBLE_TYPE_PROSODY = {
    "shout": {"speed": 1.1, "volume": 1.3},
    "whisper": {"speed": 0.9, "volume": 0.65},
    "thought": {"speed": 0.95, "volume": 0.8},
    "narration_box": {"speed": 0.9, "volume": 1.0},
    "speech": {"speed": 1.0, "volume": 1.0},
}

# Emotions that modify base prosody
EMOTION_MODIFIERS = {
    "angry": {"speed_delta": 0.05, "volume_delta": 0.1},
    "excited": {"speed_delta": 0.05, "volume_delta": 0.1},
    "scared": {"speed_delta": 0.05, "volume_delta": -0.1},
    "panicked": {"speed_delta": 0.1, "volume_delta": 0.05},
    "sad": {"speed_delta": -0.05, "volume_delta": -0.1},
    "calm": {"speed_delta": -0.05, "volume_delta": 0.0},
    "determined": {"speed_delta": 0.0, "volume_delta": 0.05},
    "dramatic": {"speed_delta": -0.05, "volume_delta": 0.05},
}


def compute_prosody_hints(emotion, tone_direction, speech_bubble_type, text_type,
                          is_first_in_panel=False, is_last_in_panel=False,
                          is_last_in_page=False, config=None):
    """Derive speed/volume/pause values from manga metadata fields."""
    config = config or CONFIG

    # Base prosody from bubble type
    base = BUBBLE_TYPE_PROSODY.get(speech_bubble_type, BUBBLE_TYPE_PROSODY["speech"]).copy()
    speed = base["speed"]
    volume = base["volume"]

    # Apply emotion modifiers
    emotion_lower = (emotion or "").lower()
    if emotion_lower in EMOTION_MODIFIERS:
        mod = EMOTION_MODIFIERS[emotion_lower]
        speed += mod.get("speed_delta", 0)
        volume += mod.get("volume_delta", 0)

    # Pauses
    pause_before = 0
    pause_after = 100  # Small default gap between lines

    if is_first_in_panel:
        pause_before = config["default_pause_between_panels_ms"]
    if is_last_in_page:
        pause_after = config["default_pause_between_pages_ms"]
    elif is_last_in_panel:
        pause_after = config["default_pause_between_panels_ms"]

    # Narration gets longer pauses
    if text_type in ("narration", "caption", "internal_monologue"):
        pause_before = max(pause_before, 300)
        pause_after = max(pause_after, 300)

    return {
        "speed": round(speed, 2),
        "volume": round(volume, 2),
        "pause_before_ms": pause_before,
        "pause_after_ms": pause_after,
    }


# ============================================================
# TEXT PREPROCESSING
# ============================================================
def preprocess_text_for_tts(text_raw, speech_bubble_type, emotion):
    """Clean and adapt text for TTS input."""
    text = text_raw.strip()
    if not text:
        return text

    # Remove manga artifacts
    text = text.replace("\\N", " ")  # ASS subtitle newlines
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace

    # Handle ALL CAPS text (common in manga) - convert to sentence case
    # but keep intentional emphasis
    if text == text.upper() and len(text) > 10:
        # Full caps line -> convert to title-like for more natural TTS
        text = text.capitalize()

    # Shout: ensure exclamation for prosody cue
    if speech_bubble_type == "shout":
        if not text.endswith(("!", "?", "!!")):
            text = text.rstrip(".") + "!"

    # Whisper: add softening
    if speech_bubble_type == "whisper":
        # Add ellipses for pausing effect
        text = text.replace(". ", "... ")

    # Clean up double/triple punctuation artifacts
    text = re.sub(r"!{3,}", "!!", text)
    text = re.sub(r"\?{3,}", "??", text)
    text = re.sub(r"\.{4,}", "...", text)

    # Remove leading/trailing artifacts
    text = text.strip(" -–—")

    return text


# ============================================================
# VOICE KEY RESOLUTION
# ============================================================
def resolve_voice_key(speaker_id, text_type, voice_registry):
    """Map a manga speaker_id to a voice registry key.

    Returns (voice_key, resolved) where resolved indicates if the
    character has a dedicated voice.
    """
    voices = voice_registry.get("voices", {})
    fallback = voice_registry.get("fallback_voice")

    # Narration / no speaker -> use fallback
    if not speaker_id or text_type in ("narration", "caption"):
        return fallback, False

    # Direct match
    if speaker_id in voices:
        return speaker_id, True

    # Try without normalization quirks
    for vkey in voices:
        if vkey.replace("_", "") == speaker_id.replace("_", ""):
            return vkey, True

    # Unmapped character -> fallback
    return fallback, False


# ============================================================
# MAIN: BUILD CHAPTER SCRIPT
# ============================================================
def build_chapter_script(config):
    """Convert manga metadata pages into a TTS-ready chapter script."""
    metadata_dir = config["manga_metadata_dir"]
    output_dir = config["output_dir"]
    tts_output_dir = config["tts_output_dir"]
    chapter_id = config["chapter_id"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load voice registry
    with open(config["voice_registry_path"], "r", encoding="utf-8") as f:
        voice_registry = json.load(f)

    # Discover and sort page JSON files
    page_files = sorted(metadata_dir.glob("page_*.json"))
    if not page_files:
        raise FileNotFoundError(f"No page_*.json files found in {metadata_dir}")

    print(f"  Found {len(page_files)} page metadata files")

    # Process each page
    lines = []
    global_order = 0
    stats = {
        "total_dialog": 0,
        "total_narration": 0,
        "total_internal_monologue": 0,
        "total_sfx": 0,
        "total_caption": 0,
        "unmapped_speakers": set(),
        "characters_used": set(),
    }

    for page_file in page_files:
        with open(page_file, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        page_num = page_data.get("page_number", 0)
        page_type = page_data.get("page_type", "story")

        # Skip non-story pages
        if page_type in ("cover", "credits", "title_spread"):
            continue

        panels = page_data.get("panels", [])
        # Sort panels by reading_order
        panels.sort(key=lambda p: p.get("reading_order", 0))

        for panel_idx, panel in enumerate(panels):
            panel_id = panel.get("panel_id", f"{page_num}_panel_{panel_idx + 1}")
            text_elements = panel.get("text_elements", [])

            # Sort text elements by reading_order within panel
            text_elements.sort(key=lambda t: t.get("reading_order", 0))

            for te_idx, te in enumerate(text_elements):
                te_type = te.get("type", "dialog")

                # Normalize: "shout"/"whisper" are bubble types, not text types;
                # treat them as dialog when used as a type value
                if te_type in ("shout", "whisper"):
                    te_type = "dialog"

                # Filter based on config
                if te_type == "sfx" and not config["include_sfx_descriptions"]:
                    continue
                if te_type == "narration" and not config["include_narration"]:
                    continue
                if te_type == "caption" and not config["include_captions"]:
                    continue
                if te_type == "internal_monologue" and not config["include_internal_monologue"]:
                    continue

                content = te.get("content", "").strip()
                if not content:
                    continue

                # Resolve voice
                speaker_id = te.get("speaker_id")
                voice_key, resolved = resolve_voice_key(
                    speaker_id, te_type, voice_registry
                )

                if speaker_id and not resolved:
                    stats["unmapped_speakers"].add(speaker_id)
                if voice_key:
                    stats["characters_used"].add(voice_key)

                # Skip if no voice available at all
                if not voice_key:
                    print(f"  Warning: No voice for {speaker_id} on page {page_num}, skipping")
                    continue

                # Preprocess text
                speech_bubble_type = te.get("speech_bubble_type", "speech")
                emotion = te.get("emotion", "neutral")
                text_for_tts = preprocess_text_for_tts(content, speech_bubble_type, emotion)

                if not text_for_tts:
                    continue

                # Compute prosody
                is_first = (te_idx == 0)
                is_last_panel = (te_idx == len(text_elements) - 1)
                is_last_page = is_last_panel and (panel_idx == len(panels) - 1)

                prosody = compute_prosody_hints(
                    emotion=emotion,
                    tone_direction=te.get("tone_direction", ""),
                    speech_bubble_type=speech_bubble_type,
                    text_type=te_type,
                    is_first_in_panel=is_first,
                    is_last_in_panel=is_last_panel,
                    is_last_in_page=is_last_page,
                    config=config,
                )

                global_order += 1
                line_id = f"{chapter_id}_{page_num:03d}_p{panel.get('reading_order', panel_idx + 1)}_t{te.get('reading_order', te_idx + 1)}"

                # Output file path
                output_file = str(tts_output_dir / f"{line_id}.wav")

                line = {
                    "line_id": line_id,
                    "page": page_num,
                    "panel_id": panel_id,
                    "reading_order_global": global_order,
                    "type": te_type,
                    "character_id": speaker_id,
                    "voice_key": voice_key,
                    "text_raw": content,
                    "text_for_tts": text_for_tts,
                    "emotion": emotion,
                    "tone_direction": te.get("tone_direction", ""),
                    "speech_bubble_type": speech_bubble_type,
                    "prosody_hints": prosody,
                    "output_file": output_file,
                }
                lines.append(line)

                # Stats
                if te_type == "dialog":
                    stats["total_dialog"] += 1
                elif te_type == "narration":
                    stats["total_narration"] += 1
                elif te_type == "internal_monologue":
                    stats["total_internal_monologue"] += 1
                elif te_type == "sfx":
                    stats["total_sfx"] += 1
                elif te_type == "caption":
                    stats["total_caption"] += 1

    # Build script document
    script = {
        "chapter_id": chapter_id,
        "generated_from": str(metadata_dir),
        "voice_registry": str(config["voice_registry_path"]),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_lines": len(lines),
        "statistics": {
            "dialog_lines": stats["total_dialog"],
            "narration_lines": stats["total_narration"],
            "internal_monologue_lines": stats["total_internal_monologue"],
            "sfx_lines": stats["total_sfx"],
            "caption_lines": stats["total_caption"],
            "characters_used": sorted(stats["characters_used"]),
            "unmapped_speakers": sorted(stats["unmapped_speakers"]),
        },
        "lines": lines,
    }

    # Save
    script_path = output_dir / "chapter_script.json"
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    return script, script_path


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Script Builder — Manga to TTS Script")
    print(f"Chapter: {CONFIG['chapter_id']}")
    print("=" * 60)

    if not CONFIG["voice_registry_path"].exists():
        raise FileNotFoundError(
            f"Voice registry not found at {CONFIG['voice_registry_path']}\n"
            "Run: python voice_clone.py  to build it first."
        )

    print("\n[1/1] Building chapter script...")
    script, script_path = build_chapter_script(CONFIG)

    stats = script["statistics"]
    print(f"\n{'=' * 60}")
    print(f"Chapter script built successfully")
    print(f"  Total lines: {script['total_lines']}")
    print(f"  Dialog: {stats['dialog_lines']}")
    print(f"  Narration: {stats['narration_lines']}")
    print(f"  Internal monologue: {stats['internal_monologue_lines']}")
    print(f"  Captions: {stats['caption_lines']}")
    print(f"  Characters used: {', '.join(stats['characters_used'])}")
    if stats["unmapped_speakers"]:
        print(f"  Unmapped speakers: {', '.join(stats['unmapped_speakers'])}")
        print(f"    (These will use the fallback voice)")
    print(f"\nOutput: {script_path}")
    print(f"\nNext step:")
    print(f"  Run: python tts_synthesize.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
