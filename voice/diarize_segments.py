"""
VLM-guided per-segment speaker diarization.

Phases:
  1. prepare_episode_audio  — extract raw audio + Demucs vocal isolation (cached)
  2. compute_segments       — scene boundaries + VLM speaker count hints
  3. _slice_segment         — ffmpeg slice of vocals track per segment (16 kHz mono)
  4. diarize_segment        — pyannote 3.1 with min/max speaker hints
  5. map_turns_to_subtitles — assign diarization labels to subtitles by overlap
  6. resolve_clusters       — cross-reference with VLM speakers → character + tiers
  7. diarize_episode        — full pipeline, caches result.json

Output: output/voice_tags/{episode_id}/diarization/result.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from voice import voice_tagger_backend as backend


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEMUCS_MODEL = "htdemucs"       # "htdemucs" (fast) | "htdemucs_ft" (better, ~4x slower)
MIN_SEGMENT_DURATION = 10.0     # Segments shorter than this are merged with neighbor
MAX_SEGMENT_DURATION = 120.0    # Segments longer than this are sub-split
MIN_SPLIT_GAP = 3.0             # Subtitle gap (seconds) used as a sub-split boundary
VLM_MIN_BUFFER = 1              # min_speakers = max(1, vlm_count - VLM_MIN_BUFFER)
VLM_MAX_BUFFER = 2              # max_speakers = vlm_count + VLM_MAX_BUFFER
PURITY_CLEAN = 0.80             # VLM agreement >= 80% => "clean" (auto-assignable)
PURITY_LIKELY = 0.60            # VLM agreement 60-79% => "likely" (needs review)
DIAR_SUBDIR = "diarization"     # Sub-dir under output/voice_tags/{episode_id}/


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _diar_dir(episode_id):
    d = backend.VOICE_TAGS_DIR / episode_id / DIAR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _count_vlm_speakers_in_range(scene, start_sec, end_sec):
    """Count unique named VLM speakers in text_elements within [start_sec, end_sec]."""
    speakers = set()
    for te in scene.get("text_elements", []):
        if te.get("type") != "dialog":
            continue
        te_start = te.get("start_time_seconds", 0)
        if start_sec - 0.5 <= te_start <= end_sec + 0.5:
            sp = te.get("speaker", "")
            if sp and sp != "Unknown" and backend._is_real_character(sp):
                speakers.add(sp)
    return len(speakers)


def _make_segment(scene_number, scene_type, start, end, subs, vlm_speakers):
    """Build a raw segment dict (segment_id assigned later by _assign_segment_ids)."""
    if vlm_speakers > 0:
        min_sp = max(1, vlm_speakers - VLM_MIN_BUFFER)
        max_sp = vlm_speakers + VLM_MAX_BUFFER
    else:
        min_sp, max_sp = 1, 4
    return {
        "scene_number": scene_number,
        "scene_type": scene_type,
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(end - start, 2),
        "subtitle_indices": [i for i, _ in subs],
        "vlm_speakers": vlm_speakers,
        "min_speakers": min_sp,
        "max_speakers": max_sp,
    }


def _sub_split(scene_number, scene_type, subs_in_range, start_sec, end_sec, scene):
    """Recursively split a long scene at the best subtitle gap near the midpoint.
    Returns list of raw segment dicts (without segment_id).
    """
    dur = end_sec - start_sec

    # Base case: short enough or only one subtitle
    if dur <= MAX_SEGMENT_DURATION or len(subs_in_range) <= 1:
        vlm_sp = _count_vlm_speakers_in_range(scene, start_sec, end_sec)
        return [_make_segment(scene_number, scene_type, start_sec, end_sec, subs_in_range, vlm_sp)]

    # Find all gaps >= MIN_SPLIT_GAP, ranked by distance from midpoint
    midpoint = (start_sec + end_sec) / 2
    gaps = []
    for i in range(len(subs_in_range) - 1):
        _, s1 = subs_in_range[i]
        _, s2 = subs_in_range[i + 1]
        gap = s2["start"] - s1["end"]
        if gap >= MIN_SPLIT_GAP:
            gap_mid = (s1["end"] + s2["start"]) / 2
            gaps.append((abs(gap_mid - midpoint), i))

    if not gaps:
        # No usable gap — return as single segment
        vlm_sp = _count_vlm_speakers_in_range(scene, start_sec, end_sec)
        return [_make_segment(scene_number, scene_type, start_sec, end_sec, subs_in_range, vlm_sp)]

    # Pick gap closest to midpoint
    _, split_after = min(gaps)
    _, s_before = subs_in_range[split_after]
    _, s_after = subs_in_range[split_after + 1]
    split_time = (s_before["end"] + s_after["start"]) / 2

    subs_a = subs_in_range[:split_after + 1]
    subs_b = subs_in_range[split_after + 1:]

    result = []
    result.extend(_sub_split(scene_number, scene_type, subs_a, start_sec, split_time, scene))
    result.extend(_sub_split(scene_number, scene_type, subs_b, split_time, end_sec, scene))
    return result


def _merge_short(segments):
    """Merge segments shorter than MIN_SEGMENT_DURATION with their previous neighbor."""
    if not segments:
        return segments
    result = [dict(segments[0])]
    for seg in segments[1:]:
        prev = result[-1]
        if seg["duration_sec"] < MIN_SEGMENT_DURATION and len(seg["subtitle_indices"]) <= 3:
            # Merge into previous segment
            prev["end_sec"] = seg["end_sec"]
            prev["duration_sec"] = round(prev["end_sec"] - prev["start_sec"], 2)
            prev["subtitle_indices"].extend(seg["subtitle_indices"])
            combined_vlm = max(prev["vlm_speakers"], seg["vlm_speakers"])
            prev["vlm_speakers"] = combined_vlm
            prev["min_speakers"] = max(1, combined_vlm - VLM_MIN_BUFFER)
            prev["max_speakers"] = combined_vlm + VLM_MAX_BUFFER
        else:
            result.append(dict(seg))
    return result


def _assign_segment_ids(segments):
    """Assign unique segment_id to each segment in-place.
    Single segment per scene: "s{scene_num:02d}"
    Multiple (sub-split): "s{scene_num:02d}a", "s{scene_num:02d}b", ...
    """
    scene_counts = Counter(s["scene_number"] for s in segments)
    scene_idx = defaultdict(int)
    for seg in segments:
        sn = seg["scene_number"]
        idx = scene_idx[sn]
        if scene_counts[sn] == 1:
            seg["segment_id"] = f"s{sn:02d}"
        else:
            seg["segment_id"] = f"s{sn:02d}{chr(ord('a') + idx)}"
        scene_idx[sn] += 1


# ─────────────────────────────────────────────────────────────
# PHASE 1: AUDIO PREPARATION
# ─────────────────────────────────────────────────────────────
def prepare_episode_audio(episode_id, diar_dir=None):
    """Extract episode audio and run Demucs vocal isolation.
    Returns Path to vocals.wav (cached after first run; ~3-5 min on GPU).
    """
    if diar_dir is None:
        diar_dir = _diar_dir(episode_id)
    vocals_path = diar_dir / "vocals.wav"
    if vocals_path.exists():
        print(f"[diarize] Using cached vocals: {vocals_path.name}")
        return vocals_path

    mkv_path = backend.EPISODE_MKVS.get(episode_id)
    if not mkv_path or not Path(mkv_path).exists():
        raise FileNotFoundError(f"MKV not found for {episode_id}: {mkv_path}")

    # Step 1: extract raw audio
    raw_path = diar_dir / "raw_audio.wav"
    if not raw_path.exists():
        print(f"[diarize] Extracting audio track {backend.SCAN_CONFIG['audio_track']} from MKV...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(mkv_path),
            "-map", f"0:a:{backend.SCAN_CONFIG['audio_track']}",
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(raw_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Audio extraction failed:\n{r.stderr[:300]}")
        print(f"[diarize] Raw audio: {raw_path.stat().st_size // 1024 // 1024} MB")

    # Step 2: Demucs vocal isolation
    print(f"[diarize] Running Demucs ({DEMUCS_MODEL}) — takes ~3-5 min on GPU...")
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model(DEMUCS_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[diarize] Demucs running on {device}")

    wav, sr = torchaudio.load(str(raw_path))
    wav = wav.unsqueeze(0).to(device)
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate

    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=True)

    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx]
    torchaudio.save(str(vocals_path), vocals.cpu(), sr)

    del model, sources, wav, vocals
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[diarize] Vocals saved: {vocals_path.stat().st_size // 1024 // 1024} MB")
    return vocals_path


# ─────────────────────────────────────────────────────────────
# PHASE 2: SEGMENTATION
# ─────────────────────────────────────────────────────────────
def compute_segments(episode_id):
    """Build per-scene diarization segments with VLM speaker count hints.
    Long scenes (>120s) are sub-split at subtitle gaps; short ones (<10s) merged.
    Returns list of segment dicts with segment_id, timing, subtitle_indices, speaker bounds.
    """
    scenes = backend.load_vlm_scenes(episode_id)
    subtitles = backend.extract_subtitles(episode_id)
    if not scenes or not subtitles:
        return []

    raw_segments = []
    for scene in sorted(scenes, key=lambda s: s["start_time_seconds"]):
        sc_start = scene["start_time_seconds"]
        sc_end = scene["end_time_seconds"]
        sc_num = scene["scene_number"]
        sc_type = scene.get("scene_type", "story")

        # Subtitles that fall within this scene (with 0.5s tolerance)
        subs_in_scene = [
            (i, sub) for i, sub in enumerate(subtitles)
            if sub["start"] >= sc_start - 0.5 and sub["end"] <= sc_end + 0.5
        ]
        if not subs_in_scene:
            continue  # No dialogue in this scene

        duration = sc_end - sc_start
        if duration <= MAX_SEGMENT_DURATION:
            vlm_sp = _count_vlm_speakers_in_range(scene, sc_start, sc_end)
            segs = [_make_segment(sc_num, sc_type, sc_start, sc_end, subs_in_scene, vlm_sp)]
        else:
            segs = _sub_split(sc_num, sc_type, subs_in_scene, sc_start, sc_end, scene)

        raw_segments.extend(segs)

    # Merge very short segments with previous neighbor
    raw_segments = _merge_short(raw_segments)

    # Assign unique segment_ids
    _assign_segment_ids(raw_segments)

    return raw_segments


# ─────────────────────────────────────────────────────────────
# PHASE 3: SEGMENT AUDIO SLICING
# ─────────────────────────────────────────────────────────────
def _slice_segment(vocals_path, segment, seg_dir):
    """Slice vocals track for one segment -> 16 kHz mono WAV (cached)."""
    seg_path = seg_dir / f"{segment['segment_id']}.wav"
    if seg_path.exists():
        return seg_path
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(segment["start_sec"]),
        "-to", str(segment["end_sec"]),
        "-i", str(vocals_path),
        "-acodec", "pcm_s16le",
        "-ar", "16000",   # pyannote expects 16 kHz
        "-ac", "1",        # mono
        str(seg_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Segment slice failed [{segment['segment_id']}]: {r.stderr[:200]}")
    return seg_path


# ─────────────────────────────────────────────────────────────
# PHASE 4: DIARIZATION
# ─────────────────────────────────────────────────────────────
def _load_pyannote():
    """Load pyannote speaker-diarization-3.1 pipeline (GPU if available)."""
    import huggingface_hub as _hfh

    # Compat patch: pyannote 3.x passes 'use_auth_token' which newer hf_hub removed
    _orig = _hfh.hf_hub_download
    def _compat(*args, **kw):
        if "use_auth_token" in kw:
            kw["token"] = kw.pop("use_auth_token")
        return _orig(*args, **kw)
    _hfh.hf_hub_download = _compat

    try:
        import pyannote.audio.core.pipeline as _pp
        _orig_pp = _pp.hf_hub_download
        def _compat_pp(*args, **kw):
            if "use_auth_token" in kw:
                kw["token"] = kw.pop("use_auth_token")
            return _orig_pp(*args, **kw)
        _pp.hf_hub_download = _compat_pp
    except Exception:
        pass

    hf_token = os.environ.get("HF_TOKEN") or _hfh.get_token() or None
    if not hf_token:
        raise RuntimeError(
            "HuggingFace token required for pyannote.\n"
            "  Set HF_TOKEN env variable, or run: huggingface-cli login\n"
            "  Also accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"[diarize] Pyannote loaded on {device}")
    return pipeline


def diarize_segment(seg_audio_path, segment, pipeline):
    """Run pyannote on one segment with VLM-derived speaker count hints.
    Returns list of turn dicts: {start, end, speaker}.
    """
    diarization = pipeline(
        str(seg_audio_path),
        min_speakers=segment["min_speakers"],
        max_speakers=segment["max_speakers"],
    )
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })
    return turns


# ─────────────────────────────────────────────────────────────
# PHASE 5: TURN-TO-SUBTITLE MAPPING
# ─────────────────────────────────────────────────────────────
def map_turns_to_subtitles(turns, subtitle_indices, subtitles, segment_start, segment_id):
    """Assign each subtitle to the diarization turn with max time overlap.
    Returns {subtitle_index: cluster_id}.
    cluster_id format: "{segment_id}_{speaker_label}" e.g. "s21_SPEAKER_00"
    """
    result = {}
    for sub_idx in subtitle_indices:
        sub = subtitles[sub_idx]
        sub_start_rel = sub["start"] - segment_start
        sub_end_rel = sub["end"] - segment_start

        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            o_start = max(sub_start_rel, turn["start"])
            o_end = min(sub_end_rel, turn["end"])
            overlap = max(0.0, o_end - o_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        if best_speaker:
            result[sub_idx] = f"{segment_id}_{best_speaker}"
    return result


# ─────────────────────────────────────────────────────────────
# PHASE 6: CLUSTER-TO-CHARACTER RESOLUTION
# ─────────────────────────────────────────────────────────────
def resolve_clusters(segment, subtitle_labels, subtitles, scenes):
    """Cross-reference diarization clusters with VLM speaker guesses.
    Assigns character name + purity tier to each cluster.
    Returns list of cluster resolution dicts.
    """
    segment_id = segment["segment_id"]
    prefix = segment_id + "_"

    # Group subtitle indices by cluster_id
    cluster_subs = defaultdict(list)
    for sub_idx, cluster_id in subtitle_labels.items():
        if cluster_id.startswith(prefix):
            cluster_subs[cluster_id].append(sub_idx)

    if not cluster_subs:
        return []

    # Find the VLM scene for this segment
    sc_num = segment["scene_number"]
    scene = next((s for s in scenes if s["scene_number"] == sc_num), None)

    resolutions = []
    for cluster_id, sub_indices in sorted(cluster_subs.items()):
        speaker_label = cluster_id[len(prefix):]

        # Vote: for each subtitle in cluster, find matching VLM text_element
        vlm_votes = Counter()
        for sub_idx in sub_indices:
            sub = subtitles[sub_idx]
            if not scene:
                continue
            best_te = None
            best_dist = 999.0
            for te in scene.get("text_elements", []):
                if te.get("type") != "dialog":
                    continue
                te_start = te.get("start_time_seconds", 0)
                dist = abs(te_start - sub["start"])
                if dist < 1.5 and dist < best_dist:
                    best_dist = dist
                    best_te = te
            if best_te:
                sp = best_te.get("speaker", "")
                if sp and sp != "Unknown" and backend._is_real_character(sp):
                    vlm_votes[sp] += 1

        total_votes = sum(vlm_votes.values())
        if total_votes > 0:
            top_char, top_count = vlm_votes.most_common(1)[0]
            purity = top_count / total_votes
        else:
            top_char = None
            purity = 0.0

        if top_char and purity >= PURITY_CLEAN:
            tier = "clean"
        elif top_char and purity >= PURITY_LIKELY:
            tier = "likely"
        else:
            tier = "impure"

        total_dur = sum(subtitles[i]["end"] - subtitles[i]["start"] for i in sub_indices)
        first_sub = subtitles[min(sub_indices)]

        resolutions.append({
            "cluster_id": cluster_id,
            "segment_id": segment_id,
            "scene_number": sc_num,
            "speaker_label": speaker_label,
            "subtitle_indices": sorted(sub_indices),
            "subtitle_count": len(sub_indices),
            "total_duration_sec": round(total_dur, 2),
            "character": top_char,
            "confidence": round(purity, 3),
            "purity_score": round(purity, 3),
            "tier": tier,
            "vlm_breakdown": dict(vlm_votes.most_common()),
            "sample_start_sec": round(first_sub["start"], 3),
            "sample_end_sec": round(first_sub["end"], 3),
        })

    return sorted(resolutions, key=lambda r: r["sample_start_sec"])


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────
def diarize_episode(episode_id, job_id=None):
    """Run full diarization pipeline for an episode. Caches result.json.
    Accepts optional job_id for progress reporting via backend job system.
    """
    def _upd(status, msg, progress=0, **kw):
        backend._update_job(job_id, status, message=msg, progress=progress, **kw)
        print(f"[diarize] [{progress:3d}%] {msg}")

    diar_dir = _diar_dir(episode_id)
    result_path = diar_dir / "result.json"

    try:
        # Phase 1: audio
        _upd("running", "Extracting and isolating vocals (Demucs)...", 5)
        vocals_path = prepare_episode_audio(episode_id, diar_dir)

        # Phase 2: segments
        _upd("running", "Computing diarization segments from VLM scene data...", 15)
        segments = compute_segments(episode_id)
        if not segments:
            _upd("error", "No segments found. Check that scene JSON and subtitles exist.", 0)
            return None

        subtitles = backend.extract_subtitles(episode_id)
        scenes = backend.load_vlm_scenes(episode_id)

        print(f"[diarize] {len(segments)} segments across {len(set(s['scene_number'] for s in segments))} scenes")

        # Phase 3 & 4: load pyannote, diarize each segment
        _upd("running", "Loading pyannote pipeline...", 20)
        pipeline = _load_pyannote()

        seg_dir = diar_dir / "segments"
        seg_dir.mkdir(exist_ok=True)

        all_subtitle_labels = {}   # sub_idx -> cluster_id
        all_groups = []
        total = len(segments)
        failed = 0

        for i, segment in enumerate(segments):
            progress = 20 + int(70 * i / total)
            _upd("running",
                 f"Diarizing {segment['segment_id']} ({i+1}/{total}, "
                 f"{segment['duration_sec']:.0f}s, "
                 f"~{segment['vlm_speakers']} speakers)...",
                 progress)

            try:
                seg_audio = _slice_segment(vocals_path, segment, seg_dir)
                turns = diarize_segment(seg_audio, segment, pipeline)
                sub_labels = map_turns_to_subtitles(
                    turns, segment["subtitle_indices"], subtitles,
                    segment["start_sec"], segment["segment_id"],
                )
                all_subtitle_labels.update(sub_labels)
                groups = resolve_clusters(segment, sub_labels, subtitles, scenes)
                all_groups.extend(groups)
                print(f"[diarize]   {segment['segment_id']}: {len(turns)} turns -> {len(groups)} groups")
            except Exception as e:
                failed += 1
                print(f"[diarize]   {segment['segment_id']} FAILED: {e}")
                continue

        # Cleanup GPU
        del pipeline
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if failed:
            print(f"[diarize] Warning: {failed}/{total} segments failed.")

        # Phase 7: save result
        _upd("running", "Saving results...", 96)

        clean_count = sum(1 for g in all_groups if g["tier"] == "clean")
        likely_count = sum(1 for g in all_groups if g["tier"] == "likely")
        impure_count = sum(1 for g in all_groups if g["tier"] == "impure")

        result = {
            "episode_id": episode_id,
            "status": "done",
            "total_segments": len(segments),
            "total_groups": len(all_groups),
            "groups": all_groups,
            "subtitle_labels": {str(k): v for k, v in all_subtitle_labels.items()},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        summary = (
            f"Done: {len(all_groups)} groups "
            f"(clean:{clean_count} likely:{likely_count} impure:{impure_count}) "
            f"across {len(segments)} segments"
        )
        _upd("done", summary, 100, result_path=str(result_path))
        print(f"[diarize] {summary}")
        return result

    except Exception as e:
        _upd("error", str(e), 0)
        raise
