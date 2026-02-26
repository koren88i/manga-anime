"""Compare speaker diarization APIs for voice cloning.

Runs 4 cloud diarization services on anime episode Japanese audio,
normalizes results, prints comparison stats, and exports tags for
human review in voice_tagger UI.

Usage:
    python diarize_compare.py --episode 0001 --services all --resume
    python diarize_compare.py --episode 0001 --services assemblyai deepgram
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
import uuid
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import httpx


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

EPISODE_MKVS = {
    "0001": BASE_DIR / "anime" / "[Anime Time] One Piece - 0001 - I'm Luffy! The Man Who's Gonna Be King of the Pirates!.mkv",
    "0002": BASE_DIR / "anime" / "[Anime Time] One Piece - 0002 - Enter the Great Swordsman! Pirate Hunter Roronoa Zoro!.mkv",
    "0003": BASE_DIR / "anime" / "[Anime Time] One Piece - 0003 - Morgan versus Luffy! Who's the Mysterious Pretty Girl.mkv",
}

ALL_SERVICES = ["assemblyai", "deepgram", "pyannoteai", "revai"]


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class DiarSegment:
    speaker_label: str
    start_sec: float
    end_sec: float
    text: str
    confidence: float

    @property
    def duration(self):
        return self.end_sec - self.start_sec


# ============================================================
# STAGE 1: Audio Extraction
# ============================================================
def extract_japanese_audio(mkv_path, output_path, sample_rate=16000):
    """Extract Japanese audio track (track 0) from MKV."""
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  Audio already extracted: {output_path.name} ({size_mb:.1f} MB)")
        return output_path

    print(f"  Extracting Japanese audio at {sample_rate}Hz...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mkv_path),
        "-map", "0:a:0",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Extracted: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


# ============================================================
# STAGE 2: API Adapters
# ============================================================

# --- AssemblyAI ---
def run_assemblyai(audio_path, output_dir, resume=False):
    """Run AssemblyAI speaker diarization."""
    cache_path = output_dir / "assemblyai_raw.json"
    if resume and cache_path.exists():
        print("  [assemblyai] Using cached result")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not api_key:
        print("  [assemblyai] SKIP - ASSEMBLYAI_API_KEY not set")
        print("    Get key at: https://www.assemblyai.com/dashboard/signup ($50 free)")
        return None

    base = "https://api.assemblyai.com/v2"
    headers = {"authorization": api_key}

    # Upload
    file_size = audio_path.stat().st_size / 1024 / 1024
    print(f"  [assemblyai] Uploading {file_size:.1f} MB...")
    with open(audio_path, "rb") as f:
        resp = httpx.post(f"{base}/upload", headers=headers, content=f, timeout=600.0)
    resp.raise_for_status()
    upload_url = resp.json()["upload_url"]

    # Create transcript
    print("  [assemblyai] Starting transcription (Japanese, speaker diarization)...")
    resp = httpx.post(
        f"{base}/transcript",
        headers={**headers, "content-type": "application/json"},
        json={
            "audio_url": upload_url,
            "speaker_labels": True,
            "speech_models": ["universal-2"],
            "language_code": "ja",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    transcript_id = resp.json()["id"]
    print(f"  [assemblyai] Transcript ID: {transcript_id}")

    # Poll
    print("  [assemblyai] Polling for completion...")
    start_time = time.time()
    while True:
        resp = httpx.get(f"{base}/transcript/{transcript_id}", headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] == "completed":
            break
        elif data["status"] == "error":
            raise RuntimeError(f"AssemblyAI error: {data.get('error', 'unknown')}")
        elapsed = time.time() - start_time
        print(f"    Status: {data['status']} ({elapsed:.0f}s elapsed)")
        time.sleep(5)

    elapsed = time.time() - start_time
    print(f"  [assemblyai] Done in {elapsed:.0f}s")

    # Save raw
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def normalize_assemblyai(raw):
    """Normalize AssemblyAI response to DiarSegment list."""
    if not raw:
        return []
    segments = []
    for utt in raw.get("utterances", []):
        segments.append(DiarSegment(
            speaker_label=f"Speaker_{utt['speaker']}",
            start_sec=utt["start"] / 1000.0,
            end_sec=utt["end"] / 1000.0,
            text=utt.get("text", ""),
            confidence=utt.get("confidence", 1.0),
        ))
    return segments


# --- Deepgram ---
def run_deepgram(audio_path, output_dir, resume=False):
    """Run Deepgram speaker diarization."""
    cache_path = output_dir / "deepgram_raw.json"
    if resume and cache_path.exists():
        print("  [deepgram] Using cached result")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    api_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not api_key:
        print("  [deepgram] SKIP - DEEPGRAM_API_KEY not set")
        print("    Get key at: https://console.deepgram.com/signup ($200 free)")
        return None

    url = "https://api.deepgram.com/v1/listen"
    params = {
        "diarize": "true",
        "language": "ja",
        "utterances": "true",
        "punctuate": "true",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }

    file_size = audio_path.stat().st_size / 1024 / 1024
    print(f"  [deepgram] Sending {file_size:.1f} MB (synchronous)...")
    start_time = time.time()

    with open(audio_path, "rb") as f:
        resp = httpx.post(url, params=params, headers=headers, content=f, timeout=600.0)
    resp.raise_for_status()
    data = resp.json()

    elapsed = time.time() - start_time
    print(f"  [deepgram] Done in {elapsed:.0f}s")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def normalize_deepgram(raw):
    """Normalize Deepgram response to DiarSegment list."""
    if not raw:
        return []
    segments = []

    # Prefer utterances if available (speaker-grouped)
    utterances = raw.get("results", {}).get("utterances", [])
    if utterances:
        for utt in utterances:
            segments.append(DiarSegment(
                speaker_label=f"Speaker_{utt['speaker']}",
                start_sec=utt["start"],
                end_sec=utt["end"],
                text=utt.get("transcript", ""),
                confidence=utt.get("confidence", 1.0),
            ))
        return segments

    # Fallback: build from words
    words = raw.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("words", [])
    if not words:
        return []

    # Group consecutive words by speaker
    current_speaker = None
    current_start = 0.0
    current_end = 0.0
    current_text = []
    current_conf = []

    for w in words:
        spk = w.get("speaker", 0)
        if spk != current_speaker and current_speaker is not None:
            segments.append(DiarSegment(
                speaker_label=f"Speaker_{current_speaker}",
                start_sec=current_start,
                end_sec=current_end,
                text=" ".join(current_text),
                confidence=sum(current_conf) / len(current_conf) if current_conf else 1.0,
            ))
            current_text = []
            current_conf = []
            current_start = w["start"]

        if current_speaker is None or spk != current_speaker:
            current_speaker = spk
            current_start = w["start"]

        current_end = w["end"]
        current_text.append(w.get("word", w.get("punctuated_word", "")))
        if "speaker_confidence" in w:
            current_conf.append(w["speaker_confidence"])

    if current_text:
        segments.append(DiarSegment(
            speaker_label=f"Speaker_{current_speaker}",
            start_sec=current_start,
            end_sec=current_end,
            text=" ".join(current_text),
            confidence=sum(current_conf) / len(current_conf) if current_conf else 1.0,
        ))

    return segments


# --- PyannoteAI ---
def run_pyannoteai(audio_path, output_dir, resume=False):
    """Run PyannoteAI speaker diarization."""
    cache_path = output_dir / "pyannoteai_raw.json"
    if resume and cache_path.exists():
        print("  [pyannoteai] Using cached result")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    api_key = os.environ.get("PYANNOTEAI_API_KEY", "")
    if not api_key:
        print("  [pyannoteai] SKIP - PYANNOTEAI_API_KEY not set")
        print("    Get key at: https://dashboard.pyannote.ai (free tier)")
        return None

    base = "https://api.pyannote.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Step 1: Get presigned upload URL
    media_key = f"ep{output_dir.name}-ja-{int(time.time())}"
    print(f"  [pyannoteai] Requesting upload URL (media://{media_key})...")
    resp = httpx.post(
        f"{base}/media/input",
        headers=headers,
        json={"url": f"media://{media_key}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    presigned_url = resp.json()["url"]

    # Step 2: Upload audio
    file_size = audio_path.stat().st_size / 1024 / 1024
    print(f"  [pyannoteai] Uploading {file_size:.1f} MB...")
    with open(audio_path, "rb") as f:
        resp = httpx.put(presigned_url, content=f, timeout=600.0)
    resp.raise_for_status()

    # Step 3: Start diarization
    print("  [pyannoteai] Starting diarization...")
    resp = httpx.post(
        f"{base}/diarize",
        headers=headers,
        json={
            "url": f"media://{media_key}",
            "confidence": True,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    job_id = resp.json()["jobId"]
    print(f"  [pyannoteai] Job ID: {job_id}")

    # Step 4: Poll
    print("  [pyannoteai] Polling for completion...")
    start_time = time.time()
    while True:
        resp = httpx.get(
            f"{base}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")

        if status == "succeeded":
            break
        elif status in ("failed", "canceled"):
            raise RuntimeError(f"PyannoteAI job {status}: {data}")

        elapsed = time.time() - start_time
        print(f"    Status: {status} ({elapsed:.0f}s elapsed)")
        time.sleep(10)

    elapsed = time.time() - start_time
    print(f"  [pyannoteai] Done in {elapsed:.0f}s")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def normalize_pyannoteai(raw):
    """Normalize PyannoteAI response to DiarSegment list."""
    if not raw:
        return []
    segments = []
    output = raw.get("output", {})
    diarization = output.get("diarization", [])

    for seg in diarization:
        # Confidence may be a dict {SPEAKER_00: 93, SPEAKER_01: 16} or absent
        conf = 1.0
        conf_data = seg.get("confidence", {})
        if isinstance(conf_data, dict) and seg["speaker"] in conf_data:
            conf = conf_data[seg["speaker"]] / 100.0

        segments.append(DiarSegment(
            speaker_label=seg["speaker"],
            start_sec=seg["start"],
            end_sec=seg["end"],
            text="",  # PyannoteAI diarization-only, no text
            confidence=conf,
        ))
    return segments


# --- Rev.ai ---
def run_revai(audio_path, output_dir, resume=False):
    """Run Rev.ai speaker diarization."""
    cache_path = output_dir / "revai_raw.json"
    if resume and cache_path.exists():
        print("  [revai] Using cached result")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    api_key = os.environ.get("REVAI_ACCESS_TOKEN", "")
    if not api_key:
        print("  [revai] SKIP - REVAI_ACCESS_TOKEN not set")
        print("    Get key at: https://www.rev.ai/auth/signup (trial credits)")
        return None

    base = "https://api.rev.ai/speechtotext/v1"
    auth_header = {"Authorization": f"Bearer {api_key}"}

    # Submit job with multipart
    file_size = audio_path.stat().st_size / 1024 / 1024
    print(f"  [revai] Submitting {file_size:.1f} MB job...")
    start_time = time.time()

    with open(audio_path, "rb") as f:
        resp = httpx.post(
            f"{base}/jobs",
            headers=auth_header,
            files={"media": (audio_path.name, f, "audio/wav")},
            data={"options": json.dumps({"language": "ja"})},
            timeout=600.0,
        )
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]
    print(f"  [revai] Job ID: {job_id}")

    # Poll for completion
    print("  [revai] Polling for completion...")
    while True:
        resp = httpx.get(f"{base}/jobs/{job_id}", headers=auth_header, timeout=30.0)
        resp.raise_for_status()
        job_status = resp.json()
        status = job_status.get("status", "unknown")

        if status == "transcribed":
            break
        elif status == "failed":
            raise RuntimeError(f"Rev.ai job failed: {job_status}")

        elapsed = time.time() - start_time
        print(f"    Status: {status} ({elapsed:.0f}s elapsed)")
        time.sleep(5)

    # Get transcript
    print("  [revai] Fetching transcript...")
    resp = httpx.get(
        f"{base}/jobs/{job_id}/transcript",
        headers={
            **auth_header,
            "Accept": "application/vnd.rev.transcript.v1.0+json",
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()

    elapsed = time.time() - start_time
    print(f"  [revai] Done in {elapsed:.0f}s")

    # Add job metadata to saved result
    data["_job"] = job_status

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def normalize_revai(raw):
    """Normalize Rev.ai response to DiarSegment list."""
    if not raw:
        return []
    segments = []

    for mono in raw.get("monologues", []):
        speaker = mono.get("speaker", 0)
        elements = mono.get("elements", [])

        # Build text and find start/end from elements
        texts = []
        seg_start = None
        seg_end = None
        confidences = []

        for elem in elements:
            if elem.get("type") == "text":
                texts.append(elem.get("value", ""))
                if seg_start is None and "ts" in elem:
                    seg_start = elem["ts"]
                if "end_ts" in elem:
                    seg_end = elem["end_ts"]
                elif "ts_end" in elem:
                    seg_end = elem["ts_end"]
                if elem.get("confidence") is not None:
                    confidences.append(elem["confidence"])
            elif elem.get("type") == "punct":
                texts.append(elem.get("value", ""))

        if seg_start is not None and seg_end is not None:
            segments.append(DiarSegment(
                speaker_label=f"Speaker_{speaker}",
                start_sec=seg_start,
                end_sec=seg_end,
                text="".join(texts),
                confidence=sum(confidences) / len(confidences) if confidences else 1.0,
            ))

    return segments


# ============================================================
# STAGE 3: Stats & Comparison
# ============================================================
def compute_stats(service_name, segments):
    """Compute and print stats for a service's diarization results."""
    if not segments:
        return {"service": service_name, "status": "skipped"}

    speakers = defaultdict(lambda: {"count": 0, "total_sec": 0.0, "segments": []})
    for seg in segments:
        speakers[seg.speaker_label]["count"] += 1
        speakers[seg.speaker_label]["total_sec"] += seg.duration
        if len(speakers[seg.speaker_label]["segments"]) < 3:
            speakers[seg.speaker_label]["segments"].append(seg)

    total_speech = sum(s["total_sec"] for s in speakers.values())

    stats = {
        "service": service_name,
        "num_speakers": len(speakers),
        "num_segments": len(segments),
        "total_speech_sec": round(total_speech, 1),
        "speakers": {},
    }

    for spk, info in sorted(speakers.items(), key=lambda x: x[1]["total_sec"], reverse=True):
        stats["speakers"][spk] = {
            "segments": info["count"],
            "total_sec": round(info["total_sec"], 1),
            "pct": round(info["total_sec"] / total_speech * 100, 1) if total_speech > 0 else 0,
            "sample_texts": [s.text for s in info["segments"] if s.text],
        }

    return stats


def print_stats(all_stats):
    """Print comparison table to console."""
    print("\n" + "=" * 70)
    print("DIARIZATION COMPARISON RESULTS")
    print("=" * 70)

    for stats in all_stats:
        if stats.get("status") == "skipped":
            print(f"\n  {stats['service']}: SKIPPED (no API key)")
            continue

        print(f"\n  {stats['service']}:")
        print(f"    Speakers: {stats['num_speakers']}")
        print(f"    Segments: {stats['num_segments']}")
        print(f"    Total speech: {stats['total_speech_sec']:.1f}s ({stats['total_speech_sec']/60:.1f}min)")
        print(f"    {'Speaker':<16} {'Segs':>5} {'Time':>8} {'%':>6}  {'Sample text'}")
        print(f"    {'-'*16} {'-'*5} {'-'*8} {'-'*6}  {'-'*30}")

        for spk, info in stats["speakers"].items():
            mins = int(info["total_sec"] // 60)
            secs = int(info["total_sec"] % 60)
            sample = info["sample_texts"][0][:40] if info["sample_texts"] else "(no text)"
            # ASCII-safe for Windows cp1252 console
            sample = sample.encode("ascii", errors="replace").decode("ascii")
            print(f"    {spk:<16} {info['segments']:>5} {mins}m{secs:02d}s {info['pct']:>5.1f}%  {sample}")

    print("\n" + "=" * 70)


def save_stats(all_stats, output_dir):
    """Save comparison stats to file."""
    stats_path = output_dir / "comparison_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # Also save human-readable text
    txt_path = output_dir / "comparison_stats.txt"
    lines = []
    lines.append("DIARIZATION COMPARISON RESULTS")
    lines.append("=" * 60)

    for stats in all_stats:
        if stats.get("status") == "skipped":
            lines.append(f"\n{stats['service']}: SKIPPED")
            continue

        lines.append(f"\n{stats['service']}:")
        lines.append(f"  Speakers: {stats['num_speakers']}")
        lines.append(f"  Segments: {stats['num_segments']}")
        lines.append(f"  Total speech: {stats['total_speech_sec']:.1f}s ({stats['total_speech_sec']/60:.1f}min)")

        for spk, info in stats["speakers"].items():
            mins = int(info["total_sec"] // 60)
            secs = int(info["total_sec"] % 60)
            lines.append(f"  {spk}: {info['segments']} segments, {mins}m{secs:02d}s ({info['pct']:.1f}%)")
            for sample in info["sample_texts"]:
                lines.append(f"    \"{sample[:60]}\"")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Stats saved to: {stats_path.name}, {txt_path.name}")


# ============================================================
# STAGE 4: Export to voice_tagger format
# ============================================================
def export_tags(service_name, segments, output_dir):
    """Export diarization segments as voice_tagger-compatible tags.json."""
    if not segments:
        return

    tags = []
    speaker_totals = defaultdict(lambda: {"total_seconds": 0.0, "tag_count": 0, "status": "diarized"})

    for seg in segments:
        if seg.duration < 0.5:  # skip very short segments
            continue

        tag_id = f"tag_{uuid.uuid4().hex[:8]}"
        tags.append({
            "tag_id": tag_id,
            "character": seg.speaker_label,
            "start_sec": round(seg.start_sec, 3),
            "end_sec": round(seg.end_sec, 3),
            "duration_sec": round(seg.duration, 3),
            "subtitle_text": seg.text,
            "confidence": f"diarized_{service_name}",
            "face_cluster_id": None,
        })

        speaker_totals[seg.speaker_label]["total_seconds"] += seg.duration
        speaker_totals[seg.speaker_label]["tag_count"] += 1

    # Round totals
    for spk in speaker_totals:
        speaker_totals[spk]["total_seconds"] = round(speaker_totals[spk]["total_seconds"], 1)

    tags_data = {
        "tags": tags,
        "character_totals": dict(speaker_totals),
    }

    tags_path = output_dir / f"{service_name}_tags.json"
    with open(tags_path, "w", encoding="utf-8") as f:
        json.dump(tags_data, f, indent=2, ensure_ascii=False)

    print(f"  [{service_name}] Exported {len(tags)} tags -> {tags_path.name}")


def save_normalized(service_name, segments, output_dir):
    """Save normalized segments."""
    norm_path = output_dir / f"{service_name}_normalized.json"
    data = [asdict(seg) for seg in segments]
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# SERVICE REGISTRY
# ============================================================
SERVICE_RUNNERS = {
    "assemblyai": (run_assemblyai, normalize_assemblyai),
    "deepgram": (run_deepgram, normalize_deepgram),
    "pyannoteai": (run_pyannoteai, normalize_pyannoteai),
    "revai": (run_revai, normalize_revai),
}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compare speaker diarization APIs for voice cloning")
    parser.add_argument("--episode", default="0001", help="Episode ID (default: 0001)")
    parser.add_argument("--services", nargs="+", default=["all"],
                        choices=ALL_SERVICES + ["all"],
                        help="Which services to run (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip services with cached results")
    args = parser.parse_args()

    episode_id = args.episode
    services = ALL_SERVICES if "all" in args.services else args.services

    # Validate episode
    if episode_id not in EPISODE_MKVS:
        print(f"Error: Unknown episode {episode_id}")
        print(f"Available: {list(EPISODE_MKVS.keys())}")
        return 1

    mkv_path = EPISODE_MKVS[episode_id]
    if not mkv_path.exists():
        print(f"Error: MKV not found: {mkv_path}")
        return 1

    output_dir = BASE_DIR / "output" / "diarization_compare" / episode_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Speaker Diarization API Comparison")
    print(f"Episode: {episode_id}")
    print(f"Services: {', '.join(services)}")
    print(f"Resume: {args.resume}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Stage 1: Extract audio
    print("\n[1/4] Audio Extraction")
    audio_path = extract_japanese_audio(
        mkv_path,
        output_dir / "audio_16k.wav",
        sample_rate=16000,
    )

    # Stage 2: Run services
    print("\n[2/4] Running Diarization Services")
    all_segments = {}
    for service in services:
        print(f"\n--- {service} ---")
        runner, normalizer = SERVICE_RUNNERS[service]

        raw = runner(audio_path, output_dir, resume=args.resume)
        segments = normalizer(raw)
        all_segments[service] = segments

        if segments:
            save_normalized(service, segments, output_dir)
            print(f"  [{service}] {len(segments)} segments from {len(set(s.speaker_label for s in segments))} speakers")

    # Stage 3: Stats
    print("\n[3/4] Computing Stats")
    all_stats = []
    for service in services:
        segments = all_segments.get(service, [])
        stats = compute_stats(service, segments)
        all_stats.append(stats)

    print_stats(all_stats)
    save_stats(all_stats, output_dir)

    # Stage 4: Export tags
    print("\n[4/4] Exporting Tags for Voice Tagger")
    for service in services:
        segments = all_segments.get(service, [])
        export_tags(service, segments, output_dir)

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print(f"  1. Review tags in voice_tagger UI")
    print(f"     Copy a service's tags to load:")
    for service in services:
        tags_path = output_dir / f"{service}_tags.json"
        if tags_path.exists():
            print(f"       {tags_path}")
    print(f"  2. Compare which service has the best speaker separation")
    print(f"  3. Assign character names to speaker labels in the UI")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
