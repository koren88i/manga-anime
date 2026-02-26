import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa

_PROJECT_ROOT = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Output directory (same as other scripts)
    "output_dir": _PROJECT_ROOT / "output",

    # Manga metadata (for character registry cross-reference)
    "manga_metadata_dir": _PROJECT_ROOT / "output" / "manga_metadata" / "0001",

    # TTS model
    "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "en",

    # Audio settings
    "output_sample_rate": 22050,

    # Reference clip selection
    "max_reference_duration": 24,      # seconds of reference audio per synthesis call
    "min_reference_duration": 6,       # minimum viable reference audio
    "reference_clips_per_emotion": 3,  # how many clips per emotion bucket
    "preferred_clip_duration": (2.0, 10.0),  # ideal clip length range (seconds)
    "min_clip_energy": 0.005,          # skip near-silent clips

    # GPU
    "gpu_device": "cuda",
}


# ============================================================
# CHARACTER ID NORMALIZATION (mirrors manga_metadata.py)
# ============================================================
CHARACTER_ALIASES = {
    "red-haired shanks": "shanks",
    "red hair shanks": "shanks",
    "captain shanks": "shanks",
    "monkey d. luffy": "luffy",
    "monkey d luffy": "luffy",
    "straw hat luffy": "luffy",
    "gol d. roger": "gold_roger",
    "gol d roger": "gold_roger",
    "gold roger": "gold_roger",
    "pirate king roger": "gold_roger",
    "king of the pirates roger": "gold_roger",
    "higuma the bear": "higuma",
    "mountain bandit higuma": "higuma",
    "lucky roo": "lucky_roux",
    "lucky roux": "lucky_roux",
    "ben beckman": "ben_beckman",
    "benn beckman": "ben_beckman",
    "lord of the coast": "lord_of_the_coast",
    "sea king": "lord_of_the_coast",
    "sea monster": "lord_of_the_coast",
}


def normalize_character_id(name):
    """Normalize a character name to a stable snake_case ID."""
    normalized = name.lower().strip()
    if normalized in CHARACTER_ALIASES:
        return CHARACTER_ALIASES[normalized]
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


# ============================================================
# STAGE 1: Build Voice Registry
# ============================================================
def scan_speaker_folders(config):
    """Scan output/speakers/ for all speaker directories and their WAV files."""
    speakers_dir = config["output_dir"] / "speakers"
    if not speakers_dir.exists():
        raise FileNotFoundError(f"Speakers directory not found: {speakers_dir}")

    speakers = {}
    for folder in sorted(speakers_dir.iterdir()):
        if not folder.is_dir():
            continue
        wavs = sorted(folder.glob("*.wav"))
        if wavs:
            speakers[folder.name] = {
                "folder_path": folder,
                "clips": wavs,
            }
    return speakers


def score_clip(clip_path, config):
    """Score a clip for quality as a TTS reference. Higher = better."""
    try:
        info = sf.info(str(clip_path))
        duration = info.duration
    except Exception:
        return 0.1  # Still usable, just unknown quality

    # Duration score: prefer clips in the ideal range
    min_pref, max_pref = config["preferred_clip_duration"]
    if min_pref <= duration <= max_pref:
        dur_score = 1.0
    elif duration < 1.0:
        dur_score = 0.2  # Very short clips are poor references
    elif duration < min_pref:
        dur_score = duration / min_pref
    else:
        dur_score = max(0.3, max_pref / duration)

    return dur_score


def classify_clip_emotion(clip_info):
    """Heuristic emotion classification from subtitle text."""
    text = " ".join(clip_info.get("subtitle_text", [])).lower()
    if not text:
        return "default"

    # Exclamation-heavy = excited/angry
    exclaim_count = text.count("!")
    question_count = text.count("?")

    if exclaim_count >= 2 or any(w in text for w in ["die!", "damn", "bastard", "kill"]):
        return "angry"
    if exclaim_count >= 1 and any(w in text for w in ["great", "yeah", "yep", "alright", "jackpot", "fun"]):
        return "excited"
    if question_count >= 1 and exclaim_count == 0:
        return "calm"
    if any(w in text for w in ["relief", "calm", "surprise", "hm", "well"]):
        return "calm"

    return "default"


def build_voice_registry(config):
    """Build voice_registry.json by scanning speakers, manifest, and manga metadata."""
    output_dir = config["output_dir"]

    # Load manifest
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found. Run speaker_diarization.py first.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load manga character registry
    registry_path = config["manga_metadata_dir"] / "character_registry.json"
    manga_characters = {}
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            manga_reg = json.load(f)
            manga_characters = manga_reg.get("characters", {})

    # Scan speaker folders on disk
    speaker_folders = scan_speaker_folders(config)

    # Build mapping: manifest speaker_id -> folder on disk
    # The manifest uses Speaker_XX keys, but folders may have been renamed
    voices = {}
    speaker_to_character = {}

    for manifest_id, sdata in manifest["speakers"].items():
        char_name = sdata.get("character_name", manifest_id)

        # Determine which folder this speaker's clips are in
        folder_name = None
        if char_name in speaker_folders:
            folder_name = char_name
        elif manifest_id in speaker_folders:
            folder_name = manifest_id

        if folder_name is None:
            print(f"  Warning: No folder found for {manifest_id} (character: {char_name})")
            continue

        folder_info = speaker_folders[folder_name]

        # Normalize to character_id for the voice key
        if char_name != manifest_id:
            # Speaker was identified -> normalize name
            voice_key = normalize_character_id(char_name)
        else:
            # Unidentified -> use manifest_id as-is (lowercase)
            voice_key = manifest_id.lower()

        speaker_to_character[manifest_id] = voice_key

        # Score and select reference clips
        clip_scores = []
        for clip_path in folder_info["clips"]:
            score = score_clip(clip_path, config)
            clip_scores.append((clip_path, score))

        # Sort by score descending
        clip_scores.sort(key=lambda x: x[1], reverse=True)

        # Build emotion buckets from manifest clip metadata
        emotion_buckets = defaultdict(list)
        for clip_meta in sdata.get("clips", []):
            emotion = classify_clip_emotion(clip_meta)
            # Find matching file on disk
            for clip_path, score in clip_scores:
                if clip_meta["filename"].split("_")[-1] in clip_path.name:
                    emotion_buckets[emotion].append(clip_path.name)
                    break

        # Build default bucket from top-scored clips
        max_clips = config["reference_clips_per_emotion"]
        default_clips = [p.name for p, s in clip_scores[:max_clips] if s > 0]

        # Ensure each emotion bucket has at most N clips
        reference_clips = {"default": default_clips}
        for emotion, clips in emotion_buckets.items():
            if emotion != "default":
                reference_clips[emotion] = clips[:max_clips]

        # Calculate total duration
        total_duration = sdata.get("total_duration_seconds", 0)

        # Look up voice profile from manga character registry
        voice_profile = {}
        if voice_key in manga_characters:
            voice_profile = manga_characters[voice_key].get("voice_profile", {})

        voices[voice_key] = {
            "character_id": voice_key,
            "display_name": char_name if char_name != manifest_id else manifest_id,
            "source_speaker_id": manifest_id,
            "speaker_folder": str(folder_info["folder_path"].relative_to(output_dir.parent)),
            "reference_clips": reference_clips,
            "total_audio_seconds": round(total_duration, 1),
            "num_clips": len(folder_info["clips"]),
            "language": config["language"],
            "voice_profile": voice_profile,
        }

        print(f"  {manifest_id} -> {voice_key}: {len(default_clips)} default refs, "
              f"{total_duration:.1f}s total")

    # Find unmapped manga characters
    mapped_keys = set(voices.keys())
    all_manga_keys = set(manga_characters.keys())
    unmapped = sorted(all_manga_keys - mapped_keys)

    # Determine fallback voice (prefer narrator-like speaker or the one with most audio)
    fallback = None
    if voices:
        fallback = max(voices.keys(), key=lambda k: voices[k]["total_audio_seconds"])

    registry = {
        "version": "1.0",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "default_tts_model": config["tts_model"],
        "default_sample_rate": config["output_sample_rate"],
        "language": config["language"],
        "voices": voices,
        "unmapped_manga_characters": unmapped,
        "fallback_voice": fallback,
    }

    # Save
    registry_path = output_dir / "voice_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\nVoice registry saved: {registry_path}")
    print(f"  Voices: {len(voices)}")
    print(f"  Unmapped manga characters: {unmapped}")
    print(f"  Fallback voice: {fallback}")

    return registry


# ============================================================
# STAGE 2: TTS Model Loading
# ============================================================
def load_tts_model(config):
    """Load the XTTS v2 model."""
    import torch
    from TTS.api import TTS

    device = config["gpu_device"] if torch.cuda.is_available() else "cpu"
    print(f"Loading TTS model: {config['tts_model']}")
    print(f"  Device: {device}")

    tts = TTS(model_name=config["tts_model"]).to(device)

    print("  Model loaded successfully")
    return tts


# ============================================================
# STAGE 3: Reference Clip Selection
# ============================================================
def select_reference_clips(voice_entry, emotion, config):
    """Select the best reference clips for a given voice and emotion.

    Returns a list of absolute paths to reference WAV files.
    """
    output_dir = config["output_dir"]
    speaker_folder = output_dir.parent / voice_entry["speaker_folder"]
    ref_clips = voice_entry.get("reference_clips", {})
    max_duration = config["max_reference_duration"]

    # Try emotion-specific clips first, fall back to default
    candidates = ref_clips.get(emotion, [])
    if not candidates:
        candidates = ref_clips.get("default", [])
    if not candidates:
        # Last resort: take any clip from the folder
        all_wavs = sorted(speaker_folder.glob("*.wav"))
        candidates = [w.name for w in all_wavs[:3]]

    # Resolve to absolute paths and accumulate up to max_duration
    selected = []
    total_dur = 0
    for clip_name in candidates:
        clip_path = speaker_folder / clip_name
        if not clip_path.exists():
            continue
        try:
            info = sf.info(str(clip_path))
            dur = info.duration
        except Exception:
            continue
        selected.append(str(clip_path))
        total_dur += dur
        if total_dur >= max_duration:
            break

    # If we don't have enough, grab more from the folder
    if total_dur < config["min_reference_duration"]:
        for wav_path in sorted(speaker_folder.glob("*.wav")):
            if str(wav_path) not in selected:
                try:
                    info = sf.info(str(wav_path))
                    dur = info.duration
                except Exception:
                    continue
                selected.append(str(wav_path))
                total_dur += dur
                if total_dur >= max_duration:
                    break

    return selected


# ============================================================
# STAGE 4: Synthesize a Single Line
# ============================================================
def synthesize_line(tts_model, text, voice_entry, emotion, prosody_hints, config):
    """Generate audio for a single line of dialogue.

    Returns a numpy array of audio samples at the configured sample rate.
    """
    # Select reference clips
    ref_paths = select_reference_clips(voice_entry, emotion, config)
    if not ref_paths:
        print(f"  Warning: No reference clips for {voice_entry['character_id']}")
        return np.zeros(config["output_sample_rate"])  # 1 second of silence

    # Use the first reference clip as the speaker wav
    # XTTS v2 accepts a single speaker_wav path
    speaker_wav = ref_paths[0]

    # Synthesize
    wav = tts_model.tts(
        text=text,
        speaker_wav=speaker_wav,
        language=config["language"],
    )

    # Convert to numpy array
    audio = np.array(wav, dtype=np.float32)

    return audio


# ============================================================
# STAGE 5: Audio Post-Processing
# ============================================================
def apply_post_processing(audio, sr, prosody_hints, speech_bubble_type):
    """Apply volume, speed, and effect adjustments to generated audio."""
    # Speed adjustment
    speed = prosody_hints.get("speed", 1.0)
    if speed != 1.0 and len(audio) > 0:
        audio = librosa.effects.time_stretch(audio, rate=speed)

    # Volume adjustment
    volume = prosody_hints.get("volume", 1.0)
    if volume != 1.0:
        audio = audio * volume

    # Normalize to prevent clipping
    peak = np.abs(audio).max()
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    # Add silence padding
    pause_before = int(sr * prosody_hints.get("pause_before_ms", 0) / 1000)
    pause_after = int(sr * prosody_hints.get("pause_after_ms", 0) / 1000)
    audio = np.concatenate([
        np.zeros(pause_before, dtype=np.float32),
        audio,
        np.zeros(pause_after, dtype=np.float32),
    ])

    return audio


# ============================================================
# UTILITIES
# ============================================================
def load_voice_registry(config):
    """Load voice_registry.json from disk."""
    path = config["output_dir"] / "voice_registry.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Voice registry not found at {path}\n"
            "Run: python voice_clone.py  to build it first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_voice_entry(registry, voice_key):
    """Look up a voice entry by key, with fallback."""
    voices = registry.get("voices", {})
    if voice_key in voices:
        return voices[voice_key]
    # Try fallback
    fallback = registry.get("fallback_voice")
    if fallback and fallback in voices:
        return voices[fallback]
    # Return first available voice
    if voices:
        return next(iter(voices.values()))
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Voice Clone — Registry Builder")
    print("=" * 60)

    print("\n[1/1] Building voice registry...")
    registry = build_voice_registry(CONFIG)

    num_voices = len(registry["voices"])
    unmapped = registry["unmapped_manga_characters"]

    print(f"\n{'=' * 60}")
    print(f"Voice registry built successfully")
    print(f"  Voices registered: {num_voices}")
    print(f"  Unmapped manga characters: {len(unmapped)}")
    if unmapped:
        print(f"    {', '.join(unmapped)}")
    print(f"\nOutput: {CONFIG['output_dir'] / 'voice_registry.json'}")
    print(f"\nNext steps:")
    print(f"  1. Review voice_registry.json")
    print(f"  2. Run script_builder.py to create TTS scripts")
    print(f"  3. Run tts_synthesize.py to generate audio")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
