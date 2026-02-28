# VLM-Guided Per-Segment Diarization Plan

## Problem

The voice tagger needs to assign character names to ~300 subtitle lines per episode. Currently:
- **VLM text_elements**: 88% coverage, but it's a visual guess (GPT-4o infers speaker from keyframes). It's one model's opinion, not an audio signal.
- **AssemblyAI diarization**: 55% coverage (ep0001: 167/302 lines), groups by voice similarity into anonymous clusters (Speaker_A, Speaker_B, ...). Complementary audio signal but doesn't know names.
- **Neither signal alone is sufficient.** VLM can be wrong about who's speaking in group shots. Diarization clusters can be impure (two voices merged) or over-split (one voice in two clusters).

## Core Idea

Run diarization **per scene segment** with VLM-derived speaker count hints. Shorter audio + fewer speakers = dramatically better clustering. Then map clusters to character names using VLM as a secondary signal.

## Data Analysis (ep0001)

```
Episode: 25 min, 302 subtitles, 13 unique VLM speakers, 76 scenes
Dialogue scenes: ~50 (non-silent)
Per-scene speakers: typically 2-3 (range: 0-5)

Scenes needing split (>120s):
  Scene 32: 98.8s, multiple speakers
  Scene 44: 82.7s, multiple speakers

Eyecatch: Scene 43 (13:00-13:14, 14.5s, 4 shots) — correctly detected via
subtitle-gap analysis. Eyecatch detection was rewritten to use subtitle gaps
scored by duration * proximity-to-midpoint, replacing a broken dissolve-based
fallback that previously flagged 96s of story content as eyecatch.
```

2-3 speaker diarization on 30-90s audio chunks is nearly a solved problem. 13-speaker diarization on 25 minutes is not.

## Architecture

### Phase 1: Audio Preparation (once per episode)

1. **Extract full episode audio** from MKV via ffmpeg (already done for voice samples)
2. **Vocal isolation** via Demucs (music/BGM removal) — run once on full audio, then slice
3. **Slice vocals by segment** — ffmpeg `-ss`/`-to` on the isolated vocals track

### Phase 2: Segmentation Strategy

Primary split: **scene boundaries** (natural dialogue boundaries, already computed).

Rules:
- **Skip** scenes with 0 subtitle lines
- **Keep as-is** scenes ≤120s — diarize the whole scene as one chunk
- **Sub-split** scenes >120s at the largest subtitle gap (≥3s silence between lines) near the midpoint. Recurse if sub-segments still >120s
- **Merge** adjacent very short scenes (<10s) with ≤3 lines that share the same speaker set — avoids too-short audio for voice profiling
- **Special case**: eyecatch scenes are skipped (no dialogue). Scenes with 0 VLM speakers but >0 subs — don't pass speaker hints, let diarization auto-detect

For each segment, compute:
```python
vlm_speakers = count unique VLM speakers in text_elements within segment time range
min_speakers = max(1, vlm_speakers - 1)  # buffer for VLM over-counting
max_speakers = vlm_speakers + 2          # buffer for VLM under-counting
```

### Phase 3: Per-Segment Diarization

**Engine: pyannote 3.1** (local, free, GPU, already in legacy code)
- Supports `min_speakers` and `max_speakers` kwargs
- Runs on CUDA (RTX 2080 Ti available)
- No API cost, can iterate freely

For each segment:
```python
diarization = pipeline(
    segment_audio_path,
    min_speakers=min_speakers,
    max_speakers=max_speakers,
)
```

Returns speaker turns: `[(start, end, "SPEAKER_00"), ...]` relative to segment start.

**Optional: AssemblyAI as quality check**
- Run on full episode with `min_speakers_expected`/`max_speakers_expected` from total VLM speaker count
- Compare cluster purity against per-segment pyannote results
- More expensive ($0.65/hr) — use for validation, not primary

### Phase 4: Cluster-to-Subtitle Mapping

Map each diarization turn to subtitle lines by timestamp overlap:
```
For each subtitle line in the segment:
    Find diarization turn with max overlap
    Assign that turn's speaker label (SPEAKER_00, etc.)
```

Result: each subtitle line gets a `diarization_speaker` label (e.g., "seg10_SPEAKER_00").

### Phase 5: Cluster-to-Character Resolution

Within each segment, cross-reference diarization clusters with VLM speaker guesses:

```
Segment (scene 21, 92s):
  SPEAKER_00: 18 subs → VLM says "Koby" ×16, "Monkey D. Luffy" ×2  →  Koby (89%)
  SPEAKER_01: 11 subs → VLM says "Monkey D. Luffy" ×10, "Koby" ×1  →  Luffy (91%)
```

Classification:
- **Clean** (>80% agreement): auto-assign character name
- **Likely** (60-80%): assign with review flag
- **Impure** (<60%): flag for manual review — cluster may contain multiple characters

Also use **scene character list** as a constraint: if VLM says the scene has [Koby, Luffy], diarization should produce ≤2 clusters. If it produces 3, one may be noise/music bleed.

### Phase 6: Sub-splitting Long Scenes — Details

For a >120 scene:

1. Find all subtitle gaps >3s within the scene
2. Pick the gap closest to the midpoint
3. Split into two sub-segments
4. For each sub-segment, count VLM speakers active in that time range
   - Sub-segment A (0-120s): VLM text_elements in this range say Alvida ×8, Koby ×6 → 2 speakers
   - Sub-segment B (120-244s): Koby ×20, Luffy ×16 → 2 speakers
5. Diarize each sub-segment independently with its own speaker count
6. No need to link clusters across sub-segments (VLM names are global)

**When NOT to split**: If a scene is ≤120s, even with 4-5 speakers, keep it whole. pyannote handles 4-5 speakers in 60-90s reasonably well. The quality gain from splitting doesn't justify the complexity.

### Phase 7: Integration with Voice Tagger UI

New pre-triage step: **Speaker Group Resolution** screen.

```
┌─────────────────────────────────────────────────────────┐
│  Speaker Group Resolution — ep0001                       │
│                                                          │
│  Scene 21 (2 clusters, 29 lines)                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │ SPEAKER_00 (18 lines, 45s)                       │   │
│  │   VLM: Koby (89%)                                │   │
│  │   [▶ Play sample]  [Koby ✓]  [Change...]        │   │
│  │                                                   │   │
│  │ SPEAKER_01 (11 lines, 32s)                       │   │
│  │   VLM: Monkey D. Luffy (91%)                     │   │
│  │   [▶ Play sample]  [Luffy ✓]  [Change...]       │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  Scene 23 (3 clusters, 50 lines) — SPLIT into 2 parts  │
│  ┌──────────────────────────────────────────────────┐   │
│  │ SPEAKER_00 (22 lines, 55s)                       │   │
│  │   VLM: Koby (82%)                                │   │
│  │   [▶ Play sample]  [Koby ✓]  [Change...]        │   │
│  │                                                   │   │
│  │ SPEAKER_01 (20 lines, 48s)                       │   │
│  │   VLM: Monkey D. Luffy (85%)                     │   │
│  │   [▶ Play sample]  [Luffy ✓]  [Change...]       │   │
│  │                                                   │   │
│  │ SPEAKER_02 (8 lines, 18s) ⚠ IMPURE              │   │
│  │   VLM: Alvida (62%), Koby (25%)                  │   │
│  │   [▶ Play sample]  [Alvida?]  [Split...]        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  [Accept All Clean (24 groups)] [Skip to Triage →]      │
└─────────────────────────────────────────────────────────┘
```

Workflow:
1. Run diarization (background, ~2-3 min per episode with GPU)
2. Show mapping table with VLM-suggested names + play buttons
3. User confirms clean groups (one click each), fixes impure ones
4. Bulk-tag all confirmed groups
5. Remaining unresolved lines → fall through to existing line-by-line triage

### Phase 8: Handling Impure Clusters

When a cluster is flagged impure (<80% VLM agreement):

**Option A: Accept majority, triage remainder**
- Assign the majority character (e.g., Alvida at 62%)
- Send the minority lines (Koby 25%) to line-by-line triage
- Simple, pragmatic

**Option B: Sub-split by VLM**
- Use VLM speaker labels to split the cluster into sub-groups
- Present each sub-group as a separate row in the mapping table
- User confirms each

**Option C: Re-diarize with higher speaker count**
- The cluster may be impure because `max_speakers` was too low
- Re-run just that segment with `max_speakers += 1`
- See if the merged cluster splits

**Recommend Option A as default**, with Option B available via "Split..." button.

## Implementation Modules

### `voice/diarize_segments.py` (implemented)

Core diarization module — 7-phase pipeline:

```python
prepare_episode_audio(episode_id)   # Phase 1: ffmpeg extract + Demucs vocal isolation (cached)
compute_segments(episode_id)        # Phase 2: scene boundaries + VLM speaker count hints + sub-split/merge
_slice_segment(vocals_path, segment, seg_dir)  # Phase 3: ffmpeg slice per segment (16 kHz mono)
diarize_segment(seg_audio, segment, pipeline)  # Phase 4: pyannote 3.1 with min/max speaker hints
map_turns_to_subtitles(turns, ...)  # Phase 5: assign diarization labels to subtitles by overlap
resolve_clusters(segment, ...)      # Phase 6: VLM vote purity → character + tier (clean/likely/impure)
diarize_episode(episode_id)         # Phase 7: full pipeline, caches result.json
```

### Modifications to `voice/voice_tagger_backend.py`

```python
def get_diarization_groups(episode_id) -> list[SpeakerGroup]:
    """Load cached diarization results. Returns groups with:
    - cluster_id, segment/scene, character (resolved), confidence
    - subtitle_ids, total_duration, sample_audio_path
    - purity_score, vlm_agreement breakdown"""

def bulk_accept_diarization(episode_id, accepts) -> int:
    """Batch-tag all subtitle lines in accepted diarization groups."""
```

### New API routes in `voice/voice_tagger.py`

```
GET  /api/episode/<id>/diarization_groups   → speaker group mapping table data
POST /api/episode/<id>/diarization_run      → trigger diarization (async)
GET  /api/episode/<id>/diarization_status   → poll progress
POST /api/episode/<id>/diarization_accept   → bulk-tag accepted groups
```

### UI addition

New "Speaker Groups" tab/mode in the tagger topbar, shown when diarization data exists. Contains the mapping table described in Phase 7. "Accept All Clean" button + per-group confirm/change/split controls.

## Dependencies

- **pyannote.audio 3.1+** (already used in legacy, needs re-integration)
- **Demucs** (vocal isolation, already used in legacy)
- **torch + torchaudio** (already installed for GPU features)
- **ffmpeg** (already used throughout)

No new external services. Everything runs locally on GPU.

## Cost & Performance

- **Diarization**: ~2-3 min/episode on GPU (pyannote on RTX 2080 Ti)
- **Demucs vocal isolation**: ~3-5 min/episode
- **Total preprocessing**: ~5-8 min/episode (one-time, cached)
- **API cost**: $0 (all local)
- **Tagging time target**: resolve ~20 speaker groups in 1-2 min → bulk-tag 80-90% of lines → triage remaining 10-20% in 2-3 min → total ~3-5 min/episode

## Open Questions

1. **Demucs model choice**: `htdemucs` (fast) vs `htdemucs_ft` (better quality, slower). Test both on anime audio.
2. **Minimum segment duration**: Below what duration is diarization unreliable? Probably ~10s. Merge very short segments with neighbors.
3. **Cross-segment speaker linking**: Not needed if VLM names are correct, but could use speaker embeddings (ECAPA-TDNN) to verify consistency across scenes — "does SPEAKER_00 in scene 21 sound like SPEAKER_01 in scene 23?" Both mapped to Koby — embedding similarity confirms.

