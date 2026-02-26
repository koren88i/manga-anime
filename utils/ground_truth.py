import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import httpx


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

CONFIG = {
    # Episodes to process
    "episodes": [
        {
            "id": "ep0001",
            "mkv_path": BASE_DIR / "anime" / "[Anime Time] One Piece - 0001 - I'm Luffy! The Man Who's Gonna Be King of the Pirates!.mkv",
        },
        {
            "id": "ep0002",
            "mkv_path": BASE_DIR / "anime" / "[Anime Time] One Piece - 0002 - Enter the Great Swordsman! Pirate Hunter Roronoa Zoro!.mkv",
        },
        {
            "id": "ep0003",
            "mkv_path": BASE_DIR / "anime" / "[Anime Time] One Piece - 0003 - Morgan versus Luffy! Who's the Mysterious Pretty Girl.mkv",
        },
    ],

    # Output
    "output_dir": BASE_DIR / "output",

    # Audio extraction (match speaker_diarization.py)
    "audio_stream_index": 1,   # 0=Japanese, 1=English dub
    "audio_sample_rate": 44100,

    # Which audio to send: "raw", "vocals", "both"
    "audio_source": "both",

    # API provider: "assemblyai" (later: "pyannoteai", "deepgram")
    "provider": "assemblyai",

    # AssemblyAI settings
    "assemblyai_api_key": os.environ.get("ASSEMBLYAI_API_KEY", ""),
    "assemblyai_speakers_expected": None,  # None = auto-detect
}


# ============================================================
# STAGE 1: Prepare Audio for Episode
# ============================================================
def extract_audio_for_episode(episode, config):
    """Ensure audio files are available for an episode.

    For ep0001, reuses existing output/raw_audio.wav and output/vocals.wav.
    For other episodes, extracts raw audio via ffmpeg.

    Returns dict mapping source labels to paths, e.g.:
        {"raw_audio": Path(...), "vocals": Path(...)}
    """
    ep_id = episode["id"]
    mkv_path = episode["mkv_path"]
    output_dir = config["output_dir"]
    gt_dir = output_dir / "ground_truth" / ep_id
    gt_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = {}

    # Check for pre-existing audio from prior pipeline run (ep0001)
    existing_raw = output_dir / "raw_audio.wav"
    existing_vocals = output_dir / "vocals.wav"

    if ep_id == "ep0001" and existing_raw.exists():
        audio_paths["raw_audio"] = existing_raw
        size_mb = existing_raw.stat().st_size / 1024 / 1024
        print(f"  Reusing existing: {existing_raw.name} ({size_mb:.1f} MB)")
    else:
        raw_path = gt_dir / "raw_audio.wav"
        if raw_path.exists():
            audio_paths["raw_audio"] = raw_path
            size_mb = raw_path.stat().st_size / 1024 / 1024
            print(f"  Found cached: {raw_path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  Extracting audio from: {mkv_path.name}")
            audio_absolute_index = config["audio_stream_index"] + 1
            cmd = [
                "ffmpeg", "-y",
                "-i", str(mkv_path),
                "-map", f"0:{audio_absolute_index}",
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(config["audio_sample_rate"]),
                "-ac", "2",
                str(raw_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")
            size_mb = raw_path.stat().st_size / 1024 / 1024
            print(f"  Extracted: {raw_path.name} ({size_mb:.1f} MB)")
            audio_paths["raw_audio"] = raw_path

    if ep_id == "ep0001" and existing_vocals.exists():
        audio_paths["vocals"] = existing_vocals
        size_mb = existing_vocals.stat().st_size / 1024 / 1024
        print(f"  Reusing existing: {existing_vocals.name} ({size_mb:.1f} MB)")

    return audio_paths


# ============================================================
# STAGE 2: API Transcription
# ============================================================
ASSEMBLYAI_BASE = "https://api.assemblyai.com/v2"


def _assemblyai_headers(api_key):
    return {"authorization": api_key, "content-type": "application/json"}


def _upload_file(audio_path, api_key):
    """Upload a local audio file to AssemblyAI, return the upload URL."""
    headers = {"authorization": api_key}
    file_size = Path(audio_path).stat().st_size / 1024 / 1024
    print(f"  Uploading {file_size:.1f} MB to AssemblyAI...")

    with open(audio_path, "rb") as f:
        resp = httpx.post(
            f"{ASSEMBLYAI_BASE}/upload",
            headers=headers,
            content=f,
            timeout=600.0,
        )
    resp.raise_for_status()
    upload_url = resp.json()["upload_url"]
    print(f"  Upload complete.")
    return upload_url


def _poll_transcript(transcript_id, api_key):
    """Poll until transcription is complete, return the full response."""
    headers = _assemblyai_headers(api_key)
    url = f"{ASSEMBLYAI_BASE}/transcript/{transcript_id}"

    while True:
        resp = httpx.get(url, headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]

        if status == "completed":
            return data
        elif status == "error":
            raise RuntimeError(f"AssemblyAI transcription failed: {data.get('error', 'unknown')}")
        else:
            time.sleep(5)


def transcribe_assemblyai(audio_path, config):
    """Send audio to AssemblyAI for transcription + speaker diarization.

    Uses the REST API directly to avoid SDK enum compatibility issues.
    Returns standardized response dict.
    """
    api_key = config["assemblyai_api_key"]
    print(f"  Uploading and transcribing: {Path(audio_path).name}")
    print(f"  (this may take several minutes for large files...)")

    start = time.time()

    # Step 1: Upload file
    upload_url = _upload_file(audio_path, api_key)

    # Step 2: Create transcription request
    request_body = {
        "audio_url": upload_url,
        "speaker_labels": True,
        "speech_models": ["universal-2"],
    }
    if config["assemblyai_speakers_expected"]:
        request_body["speakers_expected"] = config["assemblyai_speakers_expected"]

    print(f"  Starting transcription (speech_models: universal-2)...")
    resp = httpx.post(
        f"{ASSEMBLYAI_BASE}/transcript",
        headers=_assemblyai_headers(api_key),
        json=request_body,
        timeout=30.0,
    )
    resp.raise_for_status()
    transcript_id = resp.json()["id"]
    print(f"  Transcript ID: {transcript_id}")

    # Step 3: Poll for completion
    print(f"  Waiting for transcription to complete...")
    data = _poll_transcript(transcript_id, api_key)
    elapsed = time.time() - start
    print(f"  Transcription complete in {elapsed / 60:.1f} minutes")

    # Build standardized response
    response = {
        "provider": "assemblyai",
        "transcript_id": transcript_id,
        "status": data["status"],
        "audio_duration_seconds": data.get("audio_duration"),
        "utterances": [],
        "words": [],
    }

    for utt in data.get("utterances", []):
        response["utterances"].append({
            "speaker": utt["speaker"],
            "text": utt["text"],
            "start_ms": utt["start"],
            "end_ms": utt["end"],
            "start_sec": utt["start"] / 1000.0,
            "end_sec": utt["end"] / 1000.0,
        })

    for word in data.get("words", []):
        response["words"].append({
            "text": word["text"],
            "speaker": word.get("speaker"),
            "start_ms": word["start"],
            "end_ms": word["end"],
            "confidence": word.get("confidence"),
        })

    num_speakers = len(set(u["speaker"] for u in response["utterances"])) if response["utterances"] else 0
    response["num_speakers"] = num_speakers
    print(f"  Found {num_speakers} speakers, {len(response['utterances'])} utterances")

    return response


# Provider dispatcher
PROVIDERS = {
    "assemblyai": transcribe_assemblyai,
}


def transcribe(audio_path, config):
    """Route to the configured provider."""
    provider = config["provider"]
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider](audio_path, config)


# ============================================================
# STAGE 3: Convert to RTTM Format
# ============================================================
def utterances_to_rttm(utterances, file_id):
    """Convert utterance list to RTTM format lines.

    RTTM format per line:
        SPEAKER <file_id> 1 <start_sec> <duration_sec> <NA> <NA> <speaker_id> <NA> <NA>
    """
    lines = []
    for utt in utterances:
        start = utt["start_sec"]
        duration = utt["end_sec"] - utt["start_sec"]
        if duration <= 0:
            continue
        speaker = utt["speaker"]
        line = f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        lines.append(line)
    return lines


# ============================================================
# STAGE 4: Generate Human-Readable Summary
# ============================================================
def generate_summary(raw_response, file_id):
    """Generate a human-readable summary of ground truth labels."""
    utterances = raw_response.get("utterances", [])
    duration_sec = raw_response.get("audio_duration_seconds", 0)

    lines = []
    lines.append(f"Ground Truth Summary: {file_id}")
    lines.append(f"Provider: {raw_response.get('provider', 'unknown')}")
    lines.append(f"Transcript ID: {raw_response.get('transcript_id', 'N/A')}")
    if duration_sec:
        lines.append(f"Audio Duration: {duration_sec / 60:.1f} minutes")
    lines.append("=" * 50)
    lines.append("")

    if not utterances:
        lines.append("No utterances found.")
        return "\n".join(lines)

    # Speaker stats
    speakers = defaultdict(lambda: {"time": 0.0, "count": 0, "first": float("inf"), "last": 0.0, "samples": []})
    for utt in utterances:
        spk = utt["speaker"]
        dur = utt["end_sec"] - utt["start_sec"]
        speakers[spk]["time"] += dur
        speakers[spk]["count"] += 1
        speakers[spk]["first"] = min(speakers[spk]["first"], utt["start_sec"])
        speakers[spk]["last"] = max(speakers[spk]["last"], utt["end_sec"])
        if len(speakers[spk]["samples"]) < 3:
            speakers[spk]["samples"].append(utt)

    total_speech = sum(s["time"] for s in speakers.values())

    lines.append(f"Speakers: {len(speakers)} detected")
    lines.append(f"Total utterances: {len(utterances)}")
    lines.append(f"Total speech: {total_speech / 60:.1f} minutes")
    lines.append("")

    # Per-speaker stats sorted by total time descending
    lines.append("Speaker Stats:")
    for spk, stats in sorted(speakers.items(), key=lambda x: x[1]["time"], reverse=True):
        mins = int(stats["time"] // 60)
        secs = int(stats["time"] % 60)
        pct = (stats["time"] / total_speech * 100) if total_speech > 0 else 0
        lines.append(f"  Speaker {spk}: {mins}m {secs:02d}s ({pct:.1f}%), {stats['count']} utterances")

    lines.append("")

    # Timeline
    lines.append("Timeline:")
    for spk, stats in sorted(speakers.items(), key=lambda x: x[1]["first"]):
        first_m, first_s = int(stats["first"] // 60), int(stats["first"] % 60)
        last_m, last_s = int(stats["last"] // 60), int(stats["last"] % 60)
        lines.append(f"  Speaker {spk}: first at {first_m}:{first_s:02d}, last at {last_m}:{last_s:02d}")

    lines.append("")

    # Sample utterances
    lines.append("Sample Utterances:")
    for spk, stats in sorted(speakers.items(), key=lambda x: x[1]["time"], reverse=True):
        for sample in stats["samples"]:
            m = int(sample["start_sec"] // 60)
            s = int(sample["start_sec"] % 60)
            text = sample.get("text", "")
            if len(text) > 80:
                text = text[:77] + "..."
            lines.append(f'  Speaker {spk} [{m}:{s:02d}]: "{text}"')
    lines.append("")

    return "\n".join(lines)


# ============================================================
# STAGE 5: Save All Outputs
# ============================================================
def save_ground_truth(episode_id, provider, source_label, raw_response, config):
    """Save raw JSON, RTTM, and summary for one episode run."""
    gt_dir = config["output_dir"] / "ground_truth" / episode_id
    gt_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{provider}_{source_label}"

    # Raw JSON
    json_path = gt_dir / f"{prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_response, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path.name}")

    # RTTM
    rttm_lines = utterances_to_rttm(raw_response.get("utterances", []), episode_id)
    rttm_path = gt_dir / f"{prefix}.rttm"
    with open(rttm_path, "w") as f:
        f.write("\n".join(rttm_lines) + "\n")
    print(f"  Saved: {rttm_path.name} ({len(rttm_lines)} segments)")

    # Summary
    summary = generate_summary(raw_response, episode_id)
    summary_path = gt_dir / f"{prefix}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  Saved: {summary_path.name}")

    return {"json": json_path, "rttm": rttm_path, "summary": summary_path}


# ============================================================
# STAGE 6: DER Comparison
# ============================================================
def load_rttm_as_annotation(rttm_path):
    """Parse RTTM file into pyannote Annotation object."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            annotation[Segment(start, start + duration)] = speaker
    return annotation


def compute_der(reference_rttm, hypothesis_rttm):
    """Compute Diarization Error Rate between reference and hypothesis.

    Requires pyannote.metrics: pip install pyannote.metrics
    Returns dict with DER components or None if pyannote.metrics unavailable.
    """
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        print("  Warning: pyannote.metrics not available.")
        print("  Install with: pip install pyannote.metrics")
        return None

    ref = load_rttm_as_annotation(reference_rttm)
    hyp = load_rttm_as_annotation(hypothesis_rttm)

    metric = DiarizationErrorRate()
    der = metric(ref, hyp)

    return {
        "der": float(der),
        "der_pct": f"{der * 100:.1f}%",
        "reference_rttm": str(reference_rttm),
        "hypothesis_rttm": str(hypothesis_rttm),
    }


def run_comparison(config, pipeline_rttm_path):
    """Compare pipeline output against all available ground truth RTTMs."""
    print("\n" + "=" * 60)
    print("DER Comparison: Pipeline vs Ground Truth")
    print("=" * 60)

    if not Path(pipeline_rttm_path).exists():
        print(f"  Pipeline RTTM not found: {pipeline_rttm_path}")
        print("  Run the open-source pipeline first to generate RTTM output.")
        return

    gt_dir = config["output_dir"] / "ground_truth"
    rttm_files = sorted(gt_dir.glob("**/*.rttm"))

    if not rttm_files:
        print("  No ground truth RTTM files found. Run ground truth generation first.")
        return

    results = []
    for rttm_path in rttm_files:
        print(f"\n  Comparing against: {rttm_path.relative_to(gt_dir)}")
        result = compute_der(rttm_path, pipeline_rttm_path)
        if result:
            print(f"  DER: {result['der_pct']}")
            results.append(result)

    if results:
        comparison_path = gt_dir / "comparison_results.json"
        with open(comparison_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {comparison_path}")


# ============================================================
# MAIN
# ============================================================
def determine_audio_sources(audio_paths, config):
    """Decide which audio sources to transcribe based on config."""
    source = config["audio_source"]
    sources = {}

    if source in ("raw", "both") and "raw_audio" in audio_paths:
        sources["raw_audio"] = audio_paths["raw_audio"]
    if source in ("vocals", "both") and "vocals" in audio_paths:
        sources["vocals"] = audio_paths["vocals"]

    # Fallback: if "both" requested but vocals unavailable, just do raw
    if not sources and "raw_audio" in audio_paths:
        sources["raw_audio"] = audio_paths["raw_audio"]

    return sources


def main():
    parser = argparse.ArgumentParser(description="Generate ground truth diarization labels via API")
    parser.add_argument("--episodes", nargs="+", default=None,
                        help="Episode IDs to process (e.g., ep0001 ep0002). Default: all")
    parser.add_argument("--source", choices=["raw", "vocals", "both"], default=None,
                        help="Which audio to send to API. Default: from CONFIG")
    parser.add_argument("--provider", choices=["assemblyai"], default=None,
                        help="API provider. Default: from CONFIG")
    parser.add_argument("--compare", action="store_true",
                        help="Compare pipeline output against ground truth (DER)")
    parser.add_argument("--pipeline-rttm", type=Path, default=None,
                        help="Path to pipeline RTTM for --compare mode")
    args = parser.parse_args()

    # Override CONFIG with CLI args
    if args.source:
        CONFIG["audio_source"] = args.source
    if args.provider:
        CONFIG["provider"] = args.provider

    # Comparison mode
    if args.compare:
        if not args.pipeline_rttm:
            print("Error: --pipeline-rttm is required with --compare")
            return
        run_comparison(CONFIG, args.pipeline_rttm)
        return

    # Validate API key
    if not CONFIG["assemblyai_api_key"]:
        print("=" * 60)
        print("ERROR: AssemblyAI API key required!")
        print()
        print("  Set the ASSEMBLYAI_API_KEY environment variable:")
        print("    export ASSEMBLYAI_API_KEY=your_key_here")
        print()
        print("  Get a free key ($50 credit) at:")
        print("    https://www.assemblyai.com/dashboard/signup")
        print("=" * 60)
        return

    # Filter episodes
    episodes = CONFIG["episodes"]
    if args.episodes:
        episode_ids = set(args.episodes)
        episodes = [ep for ep in episodes if ep["id"] in episode_ids]
        if not episodes:
            print(f"Error: No matching episodes found for: {args.episodes}")
            print(f"Available: {[ep['id'] for ep in CONFIG['episodes']]}")
            return

    print("=" * 60)
    print("Ground Truth Generator - API Diarization")
    print(f"Provider: {CONFIG['provider']}")
    print(f"Audio source: {CONFIG['audio_source']}")
    print(f"Episodes: {[ep['id'] for ep in episodes]}")
    print("=" * 60)

    total_start = time.time()

    for i, episode in enumerate(episodes):
        ep_id = episode["id"]
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(episodes)}] Episode: {ep_id}")
        print(f"{'=' * 60}")

        # Stage 1: Prepare audio
        print(f"\n[1/3] Preparing audio...")
        audio_paths = extract_audio_for_episode(episode, CONFIG)

        # Determine which sources to run
        sources = determine_audio_sources(audio_paths, CONFIG)
        if not sources:
            print(f"  No audio available for {ep_id}, skipping.")
            continue

        for source_label, audio_path in sources.items():
            gt_dir = CONFIG["output_dir"] / "ground_truth" / ep_id

            # Check cache
            cached = gt_dir / f"{CONFIG['provider']}_{source_label}.json"
            if cached.exists():
                print(f"\n[2/3] Loading cached result: {cached.name}")
                with open(cached, encoding="utf-8") as f:
                    raw_response = json.load(f)
            else:
                # Stage 2: Transcribe
                print(f"\n[2/3] Transcribing {source_label} with {CONFIG['provider']}...")
                raw_response = transcribe(audio_path, CONFIG)

            # Stage 3: Save outputs
            print(f"\n[3/3] Saving outputs for {source_label}...")
            save_ground_truth(ep_id, CONFIG["provider"], source_label, raw_response, CONFIG)

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Ground truth generation complete in {elapsed / 60:.1f} minutes")
    print(f"Output: {CONFIG['output_dir'] / 'ground_truth'}")
    print()
    print("Next steps:")
    print("  1. Review summaries in output/ground_truth/epNNNN/")
    print("  2. Run open-source pipeline to generate hypothesis RTTM")
    print("  3. Run: python ground_truth.py --compare --pipeline-rttm <path>")
    print("=" * 60)


if __name__ == "__main__":
    main()
