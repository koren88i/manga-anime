import os
import json
import time
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa

# Compatibility patch: pyannote 3.3.2 uses deprecated 'use_auth_token' kwarg
# which newer huggingface_hub no longer accepts. Patch before pyannote imports it.
import huggingface_hub as _hfh
_orig_hf_download = _hfh.hf_hub_download
def _compat_hf_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _orig_hf_download(*args, **kwargs)
_hfh.hf_hub_download = _compat_hf_download

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Input file
    "mkv_path": Path(r"c:\Users\User\helloworld_claude\mange-anime\anime\[Anime Time] One Piece - 0001 - I'm Luffy! The Man Who's Gonna Be King of the Pirates!.mkv"),

    # Which audio track to extract (0-based among audio streams)
    # 0 = Japanese (44100 Hz), 1 = English dub (48000 Hz)
    "audio_stream_index": 1,

    # Which subtitle track to extract (0-based among subtitle streams)
    # 0 = English ASS, 1 = Arabic SRT, 2 = Portuguese SRT, 3 = Spanish SRT
    "subtitle_stream_index": 0,

    # Output directory
    "output_dir": Path(r"c:\Users\User\helloworld_claude\mange-anime\output"),

    # HuggingFace token for pyannote.audio
    "hf_token": os.environ.get("HF_TOKEN") or __import__("huggingface_hub").get_token() or "YOUR_TOKEN_HERE",

    # Demucs model: "htdemucs" (fast) or "htdemucs_ft" (better quality, 4x slower)
    "demucs_model": "htdemucs_ft",

    # Segment filtering for voice cloning quality
    "min_segment_duration": 1.0,    # Discard clips shorter than this (seconds)
    "max_segment_duration": 30.0,   # Split clips longer than this
    "merge_gap": 0.3,               # Merge same-speaker segments if gap < this

    # VAD pre-filtering
    "vad_min_speech_duration": 0.25,  # Minimum speech chunk duration (seconds)
    "vad_min_silence_duration": 0.3,  # Minimum silence to split on (seconds)

    # Energy-based filtering
    "energy_threshold_db": -40,     # Discard segments below this RMS energy (dB)

    # Audio output format
    "output_sample_rate": 22050,    # 22050 for most voice cloning (RVC, so-vits-svc)
    "output_format": "wav",
}


# ============================================================
# STAGE 1: Extract Audio from MKV
# ============================================================
def extract_audio(config):
    """Extract the selected audio track from the MKV file as WAV."""
    mkv_path = config["mkv_path"]
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_audio_path = output_dir / "raw_audio.wav"

    # Stream mapping: video is index 0, audio tracks start at 1
    audio_absolute_index = config["audio_stream_index"] + 1

    cmd = [
        "ffmpeg", "-y",
        "-i", str(mkv_path),
        "-map", f"0:{audio_absolute_index}",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        str(raw_audio_path),
    ]

    print(f"Extracting audio from: {mkv_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")

    size_mb = raw_audio_path.stat().st_size / 1024 / 1024
    print(f"Audio extracted to: {raw_audio_path} ({size_mb:.1f} MB)")
    return raw_audio_path


# ============================================================
# STAGE 2: Extract Subtitles from MKV
# ============================================================
def extract_subtitles(config):
    """Extract subtitle track as reference metadata."""
    mkv_path = config["mkv_path"]
    output_dir = config["output_dir"]

    # Subtitle streams start after video (1) + audio tracks (2) = index 3
    sub_absolute_index = config["subtitle_stream_index"] + 3
    subs_path = output_dir / "subtitles.ass"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(mkv_path),
        "-map", f"0:{sub_absolute_index}",
        "-c:s", "ass",
        str(subs_path),
    ]

    print("Extracting subtitles...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Subtitle extraction failed: {result.stderr[:200]}")
        return None

    print(f"Subtitles extracted to: {subs_path}")
    return subs_path


def parse_ass_subtitles(subs_path):
    """Parse ASS subtitles for timing and text cross-reference."""
    if subs_path is None or not subs_path.exists():
        return []

    import ass

    with open(subs_path, "r", encoding="utf-8-sig") as f:
        doc = ass.parse(f)

    events = []
    for event in doc.events:
        if event.TYPE == "Dialogue":
            events.append({
                "start": event.start.total_seconds(),
                "end": event.end.total_seconds(),
                "text": event.text,
                "style": event.style,
            })

    print(f"Parsed {len(events)} subtitle events")
    return events


# ============================================================
# STAGE 3: Vocal Isolation with Demucs
# ============================================================
def isolate_vocals(config, raw_audio_path):
    """Use Demucs to separate vocals from BGM and SFX."""
    output_dir = config["output_dir"]
    demucs_model = config["demucs_model"]

    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    print(f"Loading Demucs model: {demucs_model}")
    model = get_model(demucs_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on: {device}")

    wav, sr = torchaudio.load(str(raw_audio_path))
    wav = wav.unsqueeze(0).to(device)

    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate

    duration_min = wav.shape[-1] / sr / 60
    print(f"Running vocal separation ({duration_min:.1f} min of audio)...")

    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=True)

    # Extract vocals (sources order for htdemucs: drums, bass, other, vocals)
    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx]

    vocals_path = output_dir / "vocals.wav"
    torchaudio.save(str(vocals_path), vocals.cpu(), sr)

    # Save no-vocals mix for verification
    no_vocals_path = output_dir / "no_vocals.wav"
    no_vocals = sources[0].sum(dim=0) - vocals
    torchaudio.save(str(no_vocals_path), no_vocals.cpu(), sr)

    size_mb = vocals_path.stat().st_size / 1024 / 1024
    print(f"Vocals isolated: {vocals_path} ({size_mb:.1f} MB)")

    del model, sources, wav
    torch.cuda.empty_cache()

    return vocals_path


# ============================================================
# STAGE 3.5: VAD Pre-Filtering
# ============================================================
def vad_prefilter(config, vocals_path):
    """Apply Voice Activity Detection to clean vocals before diarization.

    Uses Silero VAD to remove non-speech regions (Demucs artifacts, breathing,
    background noise leakage) that would confuse the diarization model.
    """
    output_dir = config["output_dir"]
    vad_vocals_path = output_dir / "vocals_vad.wav"

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps, _, read_audio, _, _ = utils

    print("Running Silero VAD on isolated vocals...")
    wav = read_audio(str(vocals_path), sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=int(config["vad_min_speech_duration"] * 1000),
        min_silence_duration_ms=int(config["vad_min_silence_duration"] * 1000),
        return_seconds=False,
    )

    if not speech_timestamps:
        print("Warning: VAD found no speech — skipping VAD filtering")
        return vocals_path

    # Load original vocals at full quality
    vocals, sr = torchaudio.load(str(vocals_path))
    scale = sr / 16000  # Map 16kHz VAD timestamps to original sample rate

    # Zero out non-speech regions
    mask = torch.zeros_like(vocals)
    for ts in speech_timestamps:
        start = int(ts["start"] * scale)
        end = int(ts["end"] * scale)
        mask[:, start:end] = 1.0

    vocals_clean = vocals * mask
    torchaudio.save(str(vad_vocals_path), vocals_clean, sr)

    speech_ratio = mask.sum() / mask.numel()
    print(f"VAD kept {speech_ratio:.1%} of audio ({len(speech_timestamps)} speech chunks)")
    print(f"VAD-filtered vocals: {vad_vocals_path}")

    return vad_vocals_path


# ============================================================
# STAGE 4: Speaker Diarization with pyannote.audio
# ============================================================
def diarize_speakers(config, vocals_path):
    """Run speaker diarization on isolated vocals."""
    from pyannote.audio import Pipeline

    hf_token = config["hf_token"]
    if hf_token == "YOUR_TOKEN_HERE":
        raise ValueError(
            "HuggingFace token required!\n"
            "  1. Go to https://huggingface.co/settings/tokens\n"
            "  2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  3. Accept terms at https://huggingface.co/pyannote/segmentation-3.0\n"
            "  4. Set HF_TOKEN env variable or edit CONFIG in this script"
        )

    # Patch: pyannote 3.3.2 passes use_auth_token to hf_hub_download,
    # but newer huggingface_hub removed that parameter. Monkey-patch it
    # in pyannote's own module namespace (where it holds its reference).
    import pyannote.audio.core.pipeline as _pp
    _orig_download = _pp.hf_hub_download
    def _patched_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_download(*args, **kwargs)
    _pp.hf_hub_download = _patched_download

    print("Loading pyannote speaker diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"Diarization running on: {device}")

    print("Running speaker diarization (auto speaker count)...")
    diarization = pipeline(str(vocals_path))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "duration": turn.end - turn.start,
        })

    num_speakers = len(set(s["speaker"] for s in segments))
    print(f"Found {num_speakers} speakers in {len(segments)} segments")

    del pipeline
    torch.cuda.empty_cache()

    return segments


# ============================================================
# STAGE 4.5: Speaker Embedding Re-Clustering
# ============================================================
def recluster_speakers(config, vocals_path, segments):
    """Merge over-segmented speakers by comparing ECAPA-TDNN embeddings.

    pyannote sometimes assigns different IDs to the same speaker. This stage
    computes a mean embedding per speaker and merges speakers whose embeddings
    are closer than a cosine similarity threshold.
    """
    from speechbrain.inference.speaker import EncoderClassifier

    if len(set(s["speaker"] for s in segments)) <= 1:
        print("Only 1 speaker found — skipping re-clustering")
        return segments

    print("Loading ECAPA-TDNN speaker encoder...")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    vocals, sr = torchaudio.load(str(vocals_path))
    if vocals.shape[0] > 1:
        vocals = vocals.mean(dim=0, keepdim=True)

    # Compute mean embedding per speaker (sample up to 10 segments each)
    speaker_ids = sorted(set(s["speaker"] for s in segments))
    speaker_embeddings = {}

    for spk in speaker_ids:
        spk_segs = [s for s in segments if s["speaker"] == spk]
        spk_segs = sorted(spk_segs, key=lambda s: s["duration"], reverse=True)[:10]

        embeddings = []
        for seg in spk_segs:
            start = int(seg["start"] * sr)
            end = int(seg["end"] * sr)
            chunk = vocals[:, start:end]
            if chunk.shape[1] < sr * 0.5:  # Skip chunks < 0.5s
                continue
            with torch.no_grad():
                emb = encoder.encode_batch(chunk)
            embeddings.append(emb.squeeze())

        if embeddings:
            speaker_embeddings[spk] = torch.stack(embeddings).mean(dim=0)

    # Find pairs to merge (cosine similarity > 0.75)
    merge_threshold = 0.75
    merge_map = {}
    spk_list = list(speaker_embeddings.keys())

    for i in range(len(spk_list)):
        if spk_list[i] in merge_map:
            continue
        for j in range(i + 1, len(spk_list)):
            if spk_list[j] in merge_map:
                continue
            sim = torch.nn.functional.cosine_similarity(
                speaker_embeddings[spk_list[i]].unsqueeze(0),
                speaker_embeddings[spk_list[j]].unsqueeze(0),
            ).item()
            if sim > merge_threshold:
                merge_map[spk_list[j]] = spk_list[i]
                print(f"  Merging {spk_list[j]} -> {spk_list[i]} (similarity: {sim:.3f})")

    if not merge_map:
        print("No speakers to merge — embeddings are sufficiently distinct")
        return segments

    # Apply merges
    for seg in segments:
        if seg["speaker"] in merge_map:
            seg["speaker"] = merge_map[seg["speaker"]]

    new_count = len(set(s["speaker"] for s in segments))
    print(f"Re-clustering: {len(spk_list)} -> {new_count} speakers")
    return segments


# ============================================================
# STAGE 4.7: Energy-Based Filtering
# ============================================================
def energy_filter(config, vocals_path, segments):
    """Remove segments where RMS energy is below threshold.

    Catches Demucs artifacts and near-silent segments that aren't real speech.
    """
    threshold_db = config["energy_threshold_db"]
    vocals, sr = librosa.load(str(vocals_path), sr=None, mono=True)

    filtered = []
    dropped = 0
    for seg in segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        chunk = vocals[start:end]

        if len(chunk) == 0:
            dropped += 1
            continue

        rms = np.sqrt(np.mean(chunk ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        if rms_db >= threshold_db:
            filtered.append(seg)
        else:
            dropped += 1

    print(f"Energy filter: dropped {dropped} segments below {threshold_db} dB")
    return filtered


# ============================================================
# STAGE 5: Post-Processing
# ============================================================
def postprocess_segments(segments, config):
    """Merge nearby segments, filter short ones, split long ones."""
    min_dur = config["min_segment_duration"]
    max_dur = config["max_segment_duration"]
    merge_gap = config["merge_gap"]

    segments = sorted(segments, key=lambda s: s["start"])

    # Merge consecutive same-speaker segments with small gaps
    merged = []
    for seg in segments:
        if (merged
                and merged[-1]["speaker"] == seg["speaker"]
                and seg["start"] - merged[-1]["end"] < merge_gap):
            merged[-1]["end"] = seg["end"]
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            merged.append(dict(seg))

    # Filter short segments
    filtered = [s for s in merged if s["duration"] >= min_dur]

    # Split long segments
    final = []
    for seg in filtered:
        if seg["duration"] <= max_dur:
            final.append(seg)
        else:
            t = seg["start"]
            while t < seg["end"]:
                end = min(t + max_dur, seg["end"])
                if end - t >= min_dur:
                    final.append({
                        "start": t,
                        "end": end,
                        "speaker": seg["speaker"],
                        "duration": end - t,
                    })
                t = end

    original_count = len(segments)
    print(f"Post-processing: {original_count} -> {len(final)} segments")
    return final


# ============================================================
# STAGE 6: Extract Clips & Organize
# ============================================================
def find_matching_subtitles(segment, subtitle_events):
    """Find subtitle lines overlapping with an audio segment."""
    matches = []
    for event in subtitle_events:
        if event["start"] < segment["end"] and event["end"] > segment["start"]:
            matches.append(event["text"])
    return matches


def extract_and_organize(config, vocals_path, segments, subtitle_events):
    """Cut vocals into per-speaker clips and organize into folders."""
    output_dir = config["output_dir"]
    speakers_dir = output_dir / "speakers"
    target_sr = config["output_sample_rate"]

    print("Loading vocals for segmentation...")
    vocals, sr = librosa.load(str(vocals_path), sr=None, mono=True)

    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz...")
        vocals = librosa.resample(vocals, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Group segments by speaker
    speaker_segments = defaultdict(list)
    for seg in segments:
        speaker_segments[seg["speaker"]].append(seg)

    manifest = {
        "source_file": config["mkv_path"].name,
        "audio_track": "Japanese" if config["audio_stream_index"] == 0 else "English",
        "sample_rate": target_sr,
        "format": config["output_format"],
        "demucs_model": config["demucs_model"],
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "speakers": {},
    }

    for speaker_id in sorted(speaker_segments.keys()):
        segs = speaker_segments[speaker_id]

        num = int(speaker_id.split("_")[-1]) + 1
        folder_name = f"Speaker_{num:02d}"
        speaker_dir = speakers_dir / folder_name
        speaker_dir.mkdir(parents=True, exist_ok=True)

        total_duration = 0
        clip_info = []

        for i, seg in enumerate(segs):
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(vocals), end_sample)

            clip = vocals[start_sample:end_sample]

            if clip.max() < 0.01:
                continue

            clip_name = f"{folder_name}_{i+1:04d}_{seg['start']:.2f}-{seg['end']:.2f}.wav"
            clip_path = speaker_dir / clip_name
            sf.write(str(clip_path), clip, sr, subtype="PCM_16")

            matching_subs = find_matching_subtitles(seg, subtitle_events)

            clip_info.append({
                "filename": clip_name,
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "duration": round(seg["duration"], 3),
                "subtitle_text": matching_subs,
            })
            total_duration += seg["duration"]

        manifest["speakers"][folder_name] = {
            "original_label": speaker_id,
            "num_clips": len(clip_info),
            "total_duration_seconds": round(total_duration, 1),
            "clips": clip_info,
        }

        print(f"  {folder_name}: {len(clip_info)} clips, {total_duration:.1f}s total")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest written to: {manifest_path}")
    return speakers_dir


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Anime Speaker Diarization Pipeline")
    print("=" * 60)

    if not CONFIG["mkv_path"].exists():
        raise FileNotFoundError(f"MKV file not found: {CONFIG['mkv_path']}")

    start_time = time.time()

    print("\n[1/9] Extracting audio from MKV...")
    raw_audio_path = extract_audio(CONFIG)

    print("\n[2/9] Extracting subtitles...")
    subs_path = extract_subtitles(CONFIG)
    subtitle_events = parse_ass_subtitles(subs_path)

    print("\n[3/9] Isolating vocals with Demucs...")
    vocals_path = isolate_vocals(CONFIG, raw_audio_path)

    print("\n[4/9] VAD pre-filtering...")
    vocals_path = vad_prefilter(CONFIG, vocals_path)

    print("\n[5/9] Running speaker diarization...")
    raw_segments = diarize_speakers(CONFIG, vocals_path)

    print("\n[6/9] Re-clustering speakers by embedding similarity...")
    raw_segments = recluster_speakers(CONFIG, vocals_path, raw_segments)

    print("\n[7/9] Energy-based filtering...")
    raw_segments = energy_filter(CONFIG, vocals_path, raw_segments)

    print("\n[8/9] Post-processing segments...")
    segments = postprocess_segments(raw_segments, CONFIG)

    print("\n[9/9] Extracting clips and organizing...")
    speakers_dir = extract_and_organize(CONFIG, vocals_path, segments, subtitle_events)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 60:.1f} minutes")
    print(f"Output directory: {CONFIG['output_dir']}")
    print(f"Speaker folders: {speakers_dir}")
    print(f"\nNext steps:")
    print(f"  1. Listen to clips in each Speaker_XX folder")
    print(f"  2. Rename folders to character names (e.g., Speaker_01 -> Luffy)")
    print(f"  3. Review manifest.json for subtitle cross-reference")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
