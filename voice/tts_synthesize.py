import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Input
    "script_path": _PROJECT_ROOT / "output" / "tts_scripts" / "0001" / "chapter_script.json",
    "voice_registry_path": _PROJECT_ROOT / "output" / "voice_registry.json",
    "output_dir": _PROJECT_ROOT / "output",

    # Output
    "tts_output_dir": _PROJECT_ROOT / "output" / "tts_output" / "0001",

    # Audio
    "output_sample_rate": 22050,

    # Processing
    "resume": True,         # Skip already-generated lines
    "concatenate_chapter": True,

    # TTS config (passed to voice_clone)
    "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "en",
    "gpu_device": "cuda",
    "max_reference_duration": 24,
    "min_reference_duration": 6,
    "reference_clips_per_emotion": 3,
    "preferred_clip_duration": (2.0, 10.0),
    "min_clip_energy": 0.005,
}


# ============================================================
# MAIN SYNTHESIS
# ============================================================
def synthesize_chapter(config):
    """Run TTS for each line in the chapter script."""
    import torch
    from voice.voice_clone import (
        load_tts_model,
        load_voice_registry,
        get_voice_entry,
        synthesize_line,
        apply_post_processing,
    )

    tts_output_dir = config["tts_output_dir"]
    tts_output_dir.mkdir(parents=True, exist_ok=True)

    # Load script
    with open(config["script_path"], "r", encoding="utf-8") as f:
        script = json.load(f)

    lines = script.get("lines", [])
    total = len(lines)
    print(f"  Chapter script: {total} lines to synthesize")

    # Load voice registry
    registry = load_voice_registry(config)
    print(f"  Voice registry: {len(registry['voices'])} voices")

    # Determine which lines need generation
    if config["resume"]:
        pending = []
        skipped = 0
        for line in lines:
            output_path = Path(line["output_file"])
            if output_path.exists() and output_path.stat().st_size > 0:
                skipped += 1
            else:
                pending.append(line)
        if skipped:
            print(f"  Resuming: {skipped} lines already generated, {len(pending)} remaining")
    else:
        pending = lines

    if not pending:
        print("  All lines already generated!")
        return script

    # Load TTS model
    tts_model = load_tts_model(config)

    # Synthesize each line
    start_time = time.time()
    errors = []

    for i, line in enumerate(pending):
        line_id = line["line_id"]
        voice_key = line["voice_key"]
        text = line["text_for_tts"]
        emotion = line.get("emotion", "neutral")
        prosody = line.get("prosody_hints", {})
        bubble_type = line.get("speech_bubble_type", "speech")
        output_path = Path(line["output_file"])

        # Progress
        elapsed = time.time() - start_time
        if i > 0:
            avg_per_line = elapsed / i
            remaining = avg_per_line * (len(pending) - i)
            eta = f" (ETA: {remaining / 60:.1f}min)"
        else:
            eta = ""

        char_display = line.get("character_id") or voice_key
        print(f"  [{i + 1}/{len(pending)}]{eta} {line_id} ({char_display}): \"{text[:50]}...\"")

        # Get voice entry
        voice_entry = get_voice_entry(registry, voice_key)
        if not voice_entry:
            print(f"    Warning: No voice found for {voice_key}, skipping")
            errors.append({"line_id": line_id, "error": f"No voice for {voice_key}"})
            continue

        try:
            # Synthesize
            audio = synthesize_line(tts_model, text, voice_entry, emotion, prosody, config)

            # Post-process
            audio = apply_post_processing(audio, config["output_sample_rate"], prosody, bubble_type)

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, config["output_sample_rate"], subtype="PCM_16")

        except Exception as e:
            print(f"    Error: {e}")
            errors.append({"line_id": line_id, "error": str(e)})

        # Periodic GPU cleanup
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n  Synthesis complete: {len(pending) - len(errors)}/{len(pending)} lines "
          f"in {elapsed / 60:.1f} minutes")

    if errors:
        print(f"  Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"    {err['line_id']}: {err['error']}")

    # Free GPU memory
    del tts_model
    torch.cuda.empty_cache()

    return script


# ============================================================
# CHAPTER CONCATENATION
# ============================================================
def concatenate_chapter_audio(config):
    """Concatenate all per-line WAVs into a single chapter audio file."""
    # Load script for ordering
    with open(config["script_path"], "r", encoding="utf-8") as f:
        script = json.load(f)

    lines = script.get("lines", [])
    sr = config["output_sample_rate"]

    print(f"  Concatenating {len(lines)} lines...")

    segments = []
    missing = 0

    for line in lines:
        output_path = Path(line["output_file"])
        if output_path.exists():
            try:
                audio, file_sr = sf.read(str(output_path))
                if file_sr != sr:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
                segments.append(audio.astype(np.float32))
            except Exception as e:
                print(f"    Warning: Could not read {output_path.name}: {e}")
                missing += 1
        else:
            missing += 1
            # Insert silence as placeholder
            pause_ms = line.get("prosody_hints", {}).get("pause_after_ms", 200)
            segments.append(np.zeros(int(sr * pause_ms / 1000), dtype=np.float32))

    if missing:
        print(f"  Warning: {missing} lines missing, inserted silence placeholders")

    # Concatenate
    if segments:
        chapter_audio = np.concatenate(segments)

        chapter_path = config["tts_output_dir"] / "chapter_audio.wav"
        sf.write(str(chapter_path), chapter_audio, sr, subtype="PCM_16")

        duration = len(chapter_audio) / sr
        size_mb = chapter_path.stat().st_size / 1024 / 1024
        print(f"  Chapter audio: {chapter_path}")
        print(f"  Duration: {duration / 60:.1f} min ({duration:.1f}s)")
        print(f"  Size: {size_mb:.1f} MB")

        # Save timing manifest
        timing = []
        offset = 0.0
        for line in lines:
            output_path = Path(line["output_file"])
            if output_path.exists():
                try:
                    info = sf.info(str(output_path))
                    dur = info.duration
                except Exception:
                    dur = 0.0
            else:
                dur = line.get("prosody_hints", {}).get("pause_after_ms", 200) / 1000

            timing.append({
                "line_id": line["line_id"],
                "page": line["page"],
                "character_id": line.get("character_id"),
                "offset_seconds": round(offset, 3),
                "duration_seconds": round(dur, 3),
            })
            offset += dur

        timing_path = config["tts_output_dir"] / "chapter_timing.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2, ensure_ascii=False)
        print(f"  Timing manifest: {timing_path}")

        return chapter_path

    return None


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("TTS Synthesizer — Chapter Audio Generation")
    print("=" * 60)

    if not CONFIG["script_path"].exists():
        raise FileNotFoundError(
            f"Chapter script not found at {CONFIG['script_path']}\n"
            "Run: python script_builder.py  to create it first."
        )

    start_time = time.time()

    print("\n[1/2] Synthesizing dialogue lines...")
    script = synthesize_chapter(CONFIG)

    if CONFIG["concatenate_chapter"]:
        print("\n[2/2] Concatenating chapter audio...")
        concatenate_chapter_audio(CONFIG)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"TTS synthesis complete in {elapsed / 60:.1f} minutes")
    print(f"Output: {CONFIG['tts_output_dir']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
