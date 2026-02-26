# Stage 2 — Anime Extraction

## Purpose

Extract structured semantic representations from anime episodes, mirroring the manga extraction schema closely enough that Stage 3 alignment can correlate them. The output must capture not just what is in the episode but the production decisions Toei made — what they added, expanded, skipped, and how they reinterpreted each manga moment. This is the training signal for the generation system.

## Input

- A single episode video file
- A subtitle file for the episode (SRT or equivalent)
- `episode_id` string
- Optionally: the corresponding chapter manifest(s) from Stage 1, used for location vocabulary grounding and content origin classification
- Optionally: a previous episode manifest, used for recap detection

## Prerequisites

- **ffmpeg** on PATH (for subtitle and audio extraction)
- **OPENAI_API_KEY** environment variable (for GPT-4o VLM calls)
- **GPU recommended** — TransNetV2 benefits from CUDA. CPU works but is slower for shot detection.
- **Internet access** (optional) — for wiki episode context fetching via `one_piece_wiki_api.py`. Pipeline works without it but VLM quality is lower.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on a single episode
python anime_metadata.py "path/to/episode.mkv" --episode-id 0001

# With manga chapter manifest for vocabulary grounding
python anime_metadata.py "path/to/episode.mkv" \
    --episode-id 0001 \
    --chapter-manifest output/manga_metadata/0001/chapter_manifest.json

# With previous episode manifest for recap detection
python anime_metadata.py "path/to/episode.mkv" \
    --episode-id 0002 \
    --chapter-manifest output/manga_metadata/0002/chapter_manifest.json \
    --prev-episode-manifest output/anime_metadata/0001/episode_manifest.json

# With optional GPU models
python anime_metadata.py "path/to/episode.mkv" --episode-id 0001 \
    --use-yolo --use-clip --use-audio

# Resume interrupted run (skips completed stages)
python anime_metadata.py "path/to/episode.mkv" --episode-id 0001 --resume
```

## Startup: Context Gathering

Before any stage runs, `run_pipeline` loads up to three sources of external context:

1. **Wiki episode data** — `fetch_wiki_episode_context()` calls `one_piece_wiki_api.py` to get: closed character list with annotations (e.g. "Gol D. Roger (flashback)"), episode title, long narrative summary (~1900 chars for ep1), and location debuts from the wiki infobox. Free, no API key.
2. **Manga chapter manifest** (optional, via `--chapter-manifest`) — provides location vocabulary and a flat dialogue sequence for cross-referencing in stages 5 and 6.
3. **Previous episode manifest** (optional, via `--prev-episode-manifest`) — used for recap detection in stage 4.

## Output

All files written to `output/anime_metadata/{episode_id}/`.

---

## Implementation — Run in This Order

**1. Subtitles first.** Parse the episode subtitle file (SRT or equivalent) to get the full dialogue transcript with per-line timestamps. This is the dialogue backbone — subtitles are already segmented, timed, and accurate. Assign subtitle lines to shots later by timestamp overlap.

**2. TransNetV2 second.** Run TransNetV2 on the full video to get all shot boundaries (hard cuts and soft transitions). Outputs a list of shots with start/end frame indices and times in seconds.

**3. Per-shot feature extraction.** Extract features from each shot (see below).

**4. Scene boundary classification.** Group shots into scenes using the extracted features (see below).

**5. VLM calls — one per scene, multiple keyframes.** Send multiple per-shot keyframes (duration-scaled selection, see below), the scene's subtitle transcript, and the chapter manifest as context. Do not call the VLM per shot.

**6. Assign subtitle segments to shots and scenes** by timestamp overlap after all grouping is complete.

Keep `scene_detector.py` as a separate module from `anime_metadata.py` so scene grouping logic can be tuned independently without re-running the full pipeline.

**Speaker attribution**: assign `speaker_id` to each subtitle line using the VLM. For each shot, pass the keyframe, the subtitle lines whose timestamps fall within the shot, and the chapter character list to the VLM. Ask it to return structured JSON with `speaker_id` and `speaker_confidence` per line. The VLM self-reports its certainty — confidence is high when one character is clearly on screen speaking, lower when multiple characters are present or the speaker is off-screen. Do not attempt speaker diarization — it adds pipeline complexity with marginal gain over VLM-based attribution for this use case.

---

## Scene Detection Architecture

### What a scene is

A scene is a semantic unit — same location, same time, continuous dramatic action — containing multiple shots. A shot is a single continuous camera take ending at a cut. A typical One Piece scene contains 5–30 shots and lasts 30 seconds to 3 minutes. A typical 18-minute episode has 200–400 shots and 15–40 scenes.

Never treat every camera cut as a scene boundary.

### Step 1 — Shot boundary detection

TransNetV2 detects both hard cuts and soft transitions (dissolves, fades). Output is a list of shots with start/end frame indices and start/end times in seconds.

Additionally, explicitly flag fade-to-black and fade-to-white frames as hard scene boundary overrides. Detect by checking if average frame brightness drops below or rises above a threshold over 3+ consecutive frames spanning a shot boundary. Fade-to-black and fade-to-white in anime are deliberate studio transitions and always mark a scene break regardless of other signals.

### Step 2 — Per-shot feature extraction

For each shot, extract and store:

**Background embedding**: take the middle frame, run YOLOv8-seg character segmentation (via `--use-yolo`) to blank out characters via inpainting, embed the background-only frame using CLIP ViT-B/32 (via `--use-clip`) producing a 512-dim L2-normalized vector. This is the strongest scene-grouping signal — shots with similar background embeddings are almost certainly in the same scene. Store the embedding vector in the shot index. When CLIP embeddings are available, they replace the color histogram as the primary scene break signal (cosine distance > 0.4 = strong break). Both models are optional and independently toggleable — the pipeline falls back to color histogram correlation when CLIP is not loaded. YOLOv8 saves per-character crop images to `keyframes/character_crops/` and background plates to `keyframes/shot_NNNN_bg_plate.jpg`. Character bounding boxes with confidence scores are stored as `character_crops` in the shot index.

**Audio features (subtitle-based)**: from the subtitle file, note whether the shot contains dialogue. Computes `has_dialogue`, `dialogue_density` (overlap time / shot duration, 0.0-1.0), and `silence_ratio`. Always available, no additional dependencies.

**Audio classification (librosa-based, optional via `--use-audio`)**: extract the audio track from the video as mono WAV using ffmpeg (done once, cached). For each shot, run librosa spectral analysis on the corresponding audio segment: RMS energy for silence detection (below -40 dB threshold), harmonic-percussive source separation for music detection (harmonic ratio > 0.6 = music candidate), spectral flatness for tonal vs noise discrimination (< 0.15 = tonal/music-like), and onset detection rate for rhythmic regularity (> 1.0 onsets/sec supports music). Returns `audio_classification` dict with `has_music` (bool), `music_confidence` (0.0-1.0), `is_silence` (bool), `rms_db`, `spectral_flatness`, `onset_rate`, and `audio_type` ("music" / "dialogue" / "ambient" / "silence"). Music continuity across shot boundaries is a strong anti-break signal; music cuts and silence onsets are pro-break signals. Requires librosa (optional dependency). Graceful degradation: if librosa is not installed or audio extraction fails, the pipeline continues with subtitle-based features only.

**Cast**: character_ids visible in the shot, from the VLM keyframe analysis.

**Shot type**: classify the keyframe as wide / medium / close-up / over-the-shoulder / reaction / aerial / POV. Wide or aerial shots with unfamiliar backgrounds are strong scene-opening signals.

**Motion intensity**: optical flow magnitude between the middle frame and the frame 0.5 seconds later using OpenCV Farneback. Store as float 0.0–1.0.

**Color palette**: dominant color histogram of the full frame (HSV, 48 floats) used as the fallback scene grouping signal when CLIP embeddings are not available. Additionally, dominant hex colors are extracted via k-means clustering (default 5 colors) and stored as `color_hex` on each shot — these are the generation pipeline's color grading reference (e.g. `["#2E4A7A", "#D4A374", "#1A1A2E"]`). The shot index outputs `color_palette` as the hex color list (not the raw histogram).

**Per-shot keyframes (adaptive)**: keyframe count per shot is adaptive based on duration:
- Shots < 3s: 1 keyframe (middle frame) — `keyframes/shot_{shot_id:04d}_keyframe.jpg`
- Shots 3–6s: 2 keyframes (1/4 and 3/4 positions) — `keyframes/shot_{shot_id:04d}_kf_00.jpg`, `_kf_01.jpg`
- Shots > 6s: 1 keyframe per ~2.5s (uniformly spaced) — `keyframes/shot_{shot_id:04d}_kf_00.jpg`, etc.

Exactly one keyframe per shot is marked "primary" (closest to middle frame). The primary keyframe is used for color palette, motion intensity, saturation/brightness, CLIP embedding, and background plate extraction. All keyframes are stored as `keyframe_paths` (list) on each shot. YOLOv8 runs on all keyframes (not just primary), with `bbox_area_fraction` (bbox pixel area / frame pixel area) stored per crop for downstream prominence filtering.

Scene-level keyframes are no longer extracted from video separately. Instead, the VLM receives per-shot primary keyframes selected via duration-based sampling (see VLM multi-image section below).

### Step 3 — Scene boundary classification

For each consecutive shot pair, compute a scene-break score using these signals in priority order:

1. Fade-to-black or fade-to-white at the boundary → automatic scene break, score = 1.0, no further checks
2. Background embedding cosine distance > 0.4 → strong scene break signal (CLIP via `--use-clip`, falls back to color histogram)
3. Audio discontinuity (subtitle-based) — dialogue to silence or silence to dialogue transition → moderate signal toward break (weight 0.20)
4. Audio classification signals (librosa via `--use-audio`): music continuing across boundary → anti-break signal (weight -0.15); music stopping at boundary → pro-break signal (weight +0.20); music starting at boundary → moderate pro-break (weight +0.10); silence onset → moderate pro-break signal (weight +0.10)
5. Dialogue gap > 2–3 seconds spanning the boundary combined with any other positive signal → scene break
6. Cast discontinuity — entirely new character set with no overlap → moderate signal toward break

Combine into a weighted score. If above threshold, mark a scene boundary. Tune threshold on a sample of episodes using `scene_detector.py` independently.

### Step 4 — Content origin classification per shot

For every shot, classify whether it is directly derived from a manga panel or is a studio addition. This classification is performed by the VLM with the chapter manifest's `dialogue_index` and `statistics` passed as context. If the shot's dialogue appears in the manga dialogue index, it is manga-derived. If the scene content appears nowhere in the chapter, it is likely anime-original.

Categories:
- `manga_derived` — shot clearly corresponds to a manga panel
- `reaction_shot` — character reaction with no panel source
- `action_extension` — studio extended a fight or action beat beyond what the manga showed
- `establishing` — location-establishing shot the studio added
- `transition` — brief transition with no narrative content
- `anime_original` — fully invented content with no manga basis in the corresponding chapter window

This field is a primary input to the alignment stage and must be as accurate as possible. Passing the chapter manifest as context at extraction time is critical — the VLM is in a much better position to make this call here than the alignment stage is later without this context.

---

## Output Schema

### Per-scene JSON — `scene_NNN.json`

```
scene_number
start_time_seconds
end_time_seconds
duration_seconds
shot_count
scene_type: "story" | "recap" | "filler" | "flashback" | "transition"
anime_original: true/false
keyframe_paths: [list]            per-shot primary keyframes sent to VLM for this scene
scene_narrative                   1–2 sentence VLM description
location                          specific and consistent, e.g. "execution platform at Loguetown"
time_context                      e.g. "22 years ago", "present day", "flashback to Roger's execution"
arc_id                            e.g. "east_blue", "alabasta"
establishing_shot_type: "wide" | "medium" | "aerial" | "none"
emotional_tone: "tense" | "comedic" | "melancholic" | "triumphant" | "dramatic" | "calm" | "action"
shot_type_sequence: ["wide", "close-up", "reaction", "close-up", ...]
                                  ordered list of shot types across all shots in scene
                                  used by generation pipeline as a control signal
motion_intensity: 0.0–1.0         average across shots in scene
reconstruction_prompt             director's reconstruction prompt — see below
characters_present: [
  character_id, name, action, emotion
]
text_elements: [
  text_id
  type: "dialog" | "narration" | "internal_monologue"
  content
  speaker, speaker_id
  speaker_confidence: 0.0–1.0
  emotion
  tone_direction                  e.g. "loud and excited", "calm and warm"
  start_time_seconds
  end_time_seconds
]
sound_effects: [
  sfx_id
  description
  intensity: 1–10
  category: "impact" | "environment" | "action" | "emotion" | "voice"
  timestamp_seconds
]
```

**`reconstruction_prompt`**: a director's prompt for recreating this scene from scratch, written purely from the anime side without reference to the manga source. Written by the VLM as part of the scene analysis call. Example: "Open on a wide shot of the execution platform, late afternoon light, thousands of people silent. Slow push-in on Roger's face — defiant, almost amused. Hold for 4 seconds. Cut to wide crowd reaction. Return to close-up as he speaks. Tone: historic and mythic." This is a description of a finished product. The generation-ready imperative delta prompt — which references the manga source and instructs what to do — is produced in Stage 3.

---

### Shot index — `shot_index.json`

First-class output. The alignment stage uses this to understand the interior structure of scenes. Required for generation pipeline's shot-level control signals.

```
shots: [
  shot_id
  scene_number
  shot_number_in_scene
  start_time_seconds
  end_time_seconds
  duration_seconds
  shot_type: "wide" | "medium" | "close-up" | "reaction" | "over-the-shoulder" | "aerial" | "POV"
  content_origin: "manga_derived" | "reaction_shot" | "action_extension" | "establishing" | "transition" | "anime_original"
  characters_present: [character_ids]
  character_crops: [
    character_id
    crop_image_path               saved bounding box crop of character in this keyframe
  ]
  motion_intensity: 0.0–1.0
  background_embedding: [512-dim CLIP float array]
  background_plate_path           saved masked background image (characters removed)
  color_palette: [dominant hex colors]
  keyframe_paths: [list]          adaptive keyframe paths — saved during Stage 3 (see per-shot keyframes section)
  dialogue_in_shot: [text_ids]
  fade_type: "none" | "fade_to_black" | "fade_to_white" | "dissolve"
  audio_classification:             only present when --use-audio is enabled
    has_music: true/false
    music_confidence: 0.0–1.0
    is_silence: true/false
    rms_db: float
    spectral_flatness: float
    onset_rate: float               onsets per second
    audio_type: "music" | "dialogue" | "ambient" | "silence"
]
```

**Character crops**: YOLOv8-seg detects person-class bounding boxes in all shot keyframes during Stage 3 feature extraction (when `--use-yolo` is enabled). Cropped character images are saved to `keyframes/character_crops/shot_NNNN_kf_KK_char_CC.jpg` (multi-keyframe) or `shot_NNNN_char_NN.jpg` (single keyframe). Each crop entry in the shot index includes `bbox` (pixel coordinates), `confidence`, `crop_image_path`, `keyframe_index` (which keyframe in the shot), and `bbox_area_fraction` (bbox pixel area / frame pixel area, for prominence filtering). Uses segmentation masks when available (more precise than bounding boxes), falling back to bounding box regions. These feed the generation pipeline's character reference conditioning system.

**Background plates**: YOLOv8-seg masks out detected characters and the masked regions are inpainted (Telea algorithm) to produce a clean background plate per shot. Saved as `keyframes/shot_NNNN_bg_plate.jpg`. The `background_plate_path` is stored in the shot index. When CLIP is also enabled, the background plate (not the full frame) is embedded — this gives a purer location signal uncontaminated by character appearance. These feed the generation pipeline's location reference conditioning system.

**Color palette**: store the dominant hex colors per shot (k-means clustering, default 5 colors). The shot index outputs `color_palette` as a list of hex strings (e.g. `["#2E4A7A", "#D4A374"]`). Used by the generation pipeline to condition on location-specific color grading (e.g. the blue-green palette of Arlong Park, the white and grey of Drum Island). The raw HSV histogram (48 floats) is computed internally for scene boundary scoring but not exported in the shot index.

---

### Episode manifest — `episode_manifest.json`

```
episode_id
episode_title
episode_number
arc_id
total_scenes
total_shots
total_duration_seconds
scenes: [
  scene_number, start_time_seconds, end_time_seconds, duration_seconds,
  scene_type, shot_count, motion_intensity, emotional_tone,
  character_count, dialog_line_count, metadata_file
]
dialogue_index: [
  seq
  content
  speaker_id
  speaker
  emotion
  scene_number
  shot_id                         included for shot-granularity alignment in Stage 3
  start_time_seconds
  end_time_seconds
  location
]
processing_metadata:
  model_used
  subtitle_source
  transnetv2_version
  processing_date
  processing_time_seconds
```

The `dialogue_index` is the primary alignment anchor. Its structure mirrors the manga chapter manifest's dialogue_index so Stage 3 can match them directly. Content comes from the subtitle file. The inclusion of `shot_id` (not just `scene_number`) allows alignment at shot granularity, which is necessary for computing accurate transformation deltas.

---

### Character registry — `character_registry.json`

```
episode_id
characters:
  {character_id}:
    name
    first_appearance_scene
    first_appearance_shot
    appearances: [scene_numbers]
    shot_appearances: [shot_ids]
    voice_profile:
      observed_emotions: [...]
```

---

## One Piece Specific Handling

**Recap sequences**: most One Piece episodes open with a 60–90 second recap of the previous episode. Detect by checking if the opening dialogue duplicates content from a prior episode manifest if available. Mark as `scene_type: "recap"` and exclude all recap scenes from the dialogue_index entirely so they do not pollute alignment.

**Eyecatch**: the mid-episode title card break is a hard cut to a static card — very short, very bright. Detect and exclude entirely. It is a hard scene boundary but not a content scene and should not appear in the output.

**Flashbacks**: usually signaled by desaturation or sepia color grading. Detect with a color saturation drop over consecutive frames. Set `time_context` and `scene_type: "flashback"` accordingly.

**Character IDs**: must use exactly the same naming convention as Stage 1 manga extraction — "luffy", "zoro", "nami", "usopp", "sanji", etc. Stage 3 joins on character_id directly.

**Location vocabulary**: when a chapter manifest is provided at runtime, extract its `locations` list and pass it to the VLM as a preferred vocabulary. The VLM should choose from this list before inventing a new location label. This keeps location strings joinable across manga and anime manifests.

**Arc ID**: set `arc_id` on both the episode manifest and each scene JSON, consistent with the arc_id used in the chapter manifest.

---

## Implementation Details

### Subtitle extraction from embedded tracks

Subtitles are extracted from embedded MKV/MP4 subtitle tracks (ASS or SRT format) using ffmpeg, not transcribed via Whisper or any speech-to-text API. This is free, faster, and more accurate for fansubbed releases which already have high-quality subtitle tracks. The pipeline probes for available subtitle streams, prefers English tracks, and supports both ASS and SRT parsing.

### Wiki episode context (independent of chapter manifest)

At startup, `fetch_wiki_episode_context()` fetches structured episode data from the One Piece Fandom Wiki via the MediaWiki API:

- **Character list with annotations**: e.g. "Gol D. Roger (flashback)". Passed as a closed set to the VLM: "use ONLY these names."
- **Episode synopsis**: Full narrative summary for context grounding.
- **Arc and title**: From the episode infobox.
- **Location debuts**: New locations introduced in this episode.

This works independently of the chapter manifest — the pipeline produces good results even without manga data. When both are available, the VLM receives a richer combined context.

### Previous scene context passing

The VLM receives the prior scene's result (narrative, location, time context, characters) with a careful disclaimer: "this scene may be a DIFFERENT location/time/storyline, so only use this if there is clear continuity." This helps narrative continuity without forcing false continuity across scene breaks. Eyecatch scenes reset the continuity chain.

### Eyecatch detection

`detect_eyecatch()` in `scene_detector.py` uses tunable heuristics to find mid-episode title cards:
- Position window: must be in the middle portion of the episode
- Duration threshold: very short shots
- Surrounding fade checks: typically surrounded by fade-to-black
- Brightness analysis: eyecatch cards are often very bright
- Fallback path for atypical eyecatch formats

Detected eyecatch scenes are excluded from dialogue indexing, content origin classification, and VLM analysis. They appear in the scene list as `scene_type: "eyecatch"` but are not counted as content scenes.

### Graceful degradation without chapter manifest

The pipeline works without a chapter manifest from Stage 1. Wiki context fills the gap for character identification and location vocabulary. The heuristic content origin classifier still functions (using fuzzy text matching and shot property analysis). Speaker attribution uses scene character lists from VLM analysis.

### VLM prompt engineering

The scene analysis prompt is constructed dynamically with:
- **Episode position**: "Scene X/Y in episode" to give the VLM episode structure awareness
- **Flashback candidate flags**: From saturation analysis in feature extraction
- **Combined location vocabulary**: Merges manga manifest locations with wiki location debuts
- **Character identification rules**: Prohibits generic labels ("unknown", "Character 1", "man"), requires actual character names from the wiki list, instructs to omit rather than guess
- **Manga dialogue reference**: Up to 50 manga dialogue lines with sequence numbers, speaker names, and content, framed for content_origin classification ("if a shot's dialogue closely matches a line here, its content_origin is 'manga_derived'")
- **Per-shot info**: All shots in the scene with shot_id, timing, duration, dialogue presence, and motion intensity — enables VLM to classify shot_type and content_origin per shot
- **Indexed subtitle lines**: Subtitle lines with numeric indices and timestamps, enabling VLM to return speaker attribution per line

The VLM returns an expanded JSON response with:
- Scene-level fields: narrative, location, time_context, emotional_tone, scene_type, establishing_shot_type, characters_present
- **`reconstruction_prompt`**: A director's description for recreating the scene from scratch (camera work, timing, framing, tone). Does not reference the manga source.
- **`shot_analysis`**: Per-shot `shot_type` (wide/medium/close-up/over-the-shoulder/reaction/aerial/POV) and `content_origin` classification. Falls back to "medium" for shot_type and heuristic for content_origin when VLM result is missing.
- **`dialogue_attribution`**: Per-subtitle-line `speaker` and `speaker_confidence`. Maps back to global subtitle IDs via line indices.

### VLM multi-image input (duration-scaled)

Stage 5 sends multiple per-shot keyframes to the VLM, not a single scene keyframe. The number of images is duration-scaled:

- Scene < 15s: all shots' primary keyframes (typically 2–5 images)
- Scene 15–60s: 1 image per ~5s of scene duration (uniformly sampled across shots)
- Scene > 60s: 1 image per ~8s of scene duration

When sampling, the first and last shot's keyframe are always included, with even spacing between. The prompt includes an image-to-shot mapping section so the VLM knows which image corresponds to which shot. Shots without a keyframe image are instructed to be inferred from timing, dialogue, and context.

Scene-level keyframe extraction from video is no longer performed — the `cv2.VideoCapture` that was opened in Stage 5 for this purpose has been removed. All keyframes come from the per-shot extraction in Stage 3.

Config entries (`anime_metadata.py`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vlm_short_scene_threshold` | 15.0 | Scenes below this duration: all shots get keyframes |
| `vlm_medium_scene_interval` | 5.0 | Scenes 15–60s: 1 keyframe per this many seconds |
| `vlm_long_scene_threshold` | 60.0 | Scenes above this: use longer interval |
| `vlm_long_scene_interval` | 8.0 | Scenes > 60s: 1 keyframe per this many seconds |

### VLM result caching

Raw VLM responses are cached to `vlm_cache/scene_NNN.json` within the episode output directory. On resume, the cache is checked first (preserving per-shot analysis and dialogue attribution), with fallback to legacy scene JSON files from Stage 8. This ensures per-shot VLM data (shot_type, vlm_content_origin) survives pipeline restarts.

### Cost tracking

API token usage (input and output tokens) is tracked per VLM call via `API_COSTS`. The pipeline prints a detailed cost breakdown at completion with per-call statistics and estimated dollar costs based on current GPT-4o pricing.

### Content origin — dual VLM + heuristic classification

Content origin classification runs as a separate Stage 6, after VLM scene analysis. The VLM classifies each shot's content_origin as part of the scene analysis call (Stage 5), seeing the manga dialogue reference and the visual keyframe. Stage 6 then runs the heuristic classifier (fuzzy text matching + shot property analysis) as a fallback. When VLM content_origin is available, it overrides the heuristic; when missing, the heuristic result is used. Disagreements between VLM and heuristic are logged (stored as `_content_origin_heuristic` on the shot) for analysis. The heuristic stage remains independently re-runnable without re-running VLM calls.

### VLM-based speaker attribution

Speaker attribution uses the VLM's `dialogue_attribution` response. For each scene's VLM call, subtitle lines are included with timestamps and numeric indices. The VLM returns `speaker` and `speaker_confidence` per line. These are mapped back to global subtitle IDs and stored on the scene dict. In Stage 7 (dialogue index construction), VLM attribution is used when available (`speaker_id` resolved via `normalize_character_id`); the heuristic fallback (scene character count + shot character data) is used when VLM attribution is missing.

### Audio classifier (optional, `--use-audio`)

A lightweight librosa-based audio classifier that runs per shot during Stage 3 feature extraction. Enabled via `--use-audio`. Does not require GPU — CPU-based spectral analysis only.

**Audio extraction**: ffmpeg extracts the audio track from the video as mono 16-bit PCM WAV at 22050 Hz. Done once per episode and cached as `audio_track.wav` in the episode output directory. Subsequent runs (including `--resume`) reuse the cached file.

**Per-shot classification**: For each shot's time segment, librosa computes:
- **RMS energy** → silence detection (below -40 dB = silence)
- **Harmonic-percussive source separation** → music detection (harmonic ratio > 0.6 = music candidate)
- **Spectral flatness** → tonal vs noise (< 0.15 = tonal/music-like; > 0.3 = ambient noise)
- **Onset detection rate** → rhythmic regularity (> 1.0 onsets/sec supports music classification)

These signals are combined into a heuristic music confidence score (0.0-1.0). A shot is classified as one of: "music", "dialogue", "ambient", or "silence".

**Scene boundary integration**: Three new signals in `_compute_scene_break_score()`:
- Music continuing across boundary → anti-break signal (weight -0.15): reduces false scene breaks in music-backed sequences
- Music cut at boundary → pro-break signal (weight +0.20): studio music edits often coincide with scene changes
- Silence onset at boundary → moderate pro-break signal (weight +0.10)

These supplement (not replace) the existing subtitle-based audio features (`has_dialogue`, `dialogue_density`, `silence_ratio`). The subtitle-based features remain the baseline, available without any optional dependency. The audio classifier adds a deeper signal that subtitle timing cannot capture — notably, background music continuity.

**Graceful degradation**: If librosa is not installed, a warning is logged and the pipeline continues with subtitle-based features only. If ffmpeg audio extraction fails, same fallback. Per-shot classification errors fall back to `_default_audio_classification()` (all fields zeroed, `audio_type: "unknown"`). The `audio_classification` field is only included in the shot index when real data is available.

**Dependencies**: `librosa` (optional), `soundfile` (required by librosa). Both are commented in `requirements.txt` since they are optional.

---

## Configuration

Both `scene_detector.py` and `anime_metadata.py` have a `CONFIG` dict at the top of the file. Key tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scene_break_threshold` | 0.55 | Combined score above this = scene break |
| `color_distance_threshold` | 0.4 | Color histogram distance for strong break signal |
| `clip_distance_threshold` | 0.4 | CLIP cosine distance for strong break signal |
| `manga_dialogue_match_threshold` | 0.75 | Fuzzy match score for manga<->anime dialogue |
| `recap_max_duration` | 90.0 | Seconds to check for recap content at episode start |
| `openai_model` | gpt-4o | VLM model for scene analysis |
| `vlm_calls_per_minute` | 100 | Self-imposed rate limit (OpenAI Tier 1 allows 500+ RPM) |
| `color_hex_count` | 5 | Number of dominant hex colors to extract per shot |
| `yolo_confidence` | 0.3 | YOLOv8 detection confidence threshold |
| `keyframe_short_threshold` | 3.0 | Shots below this duration: 1 keyframe (middle) |
| `keyframe_medium_threshold` | 6.0 | Shots 3–6s: 2 keyframes (1/4 + 3/4) |
| `keyframe_long_interval` | 2.5 | Shots > 6s: 1 keyframe per this many seconds |

Tuning scene detection standalone:

```bash
python scene_detector.py "path/to/episode.mkv" \
    --threshold 0.55 \
    --bg-threshold 0.4 \
    --output output/scene_detector_debug
```

---

## Voice Tagger UI

After running the anime metadata pipeline, use the voice tagger to assign subtitles to characters:

```bash
python voice_tagger.py
# Opens at http://localhost:7861
```

The voice tagger uses pre-computed VLM scene data from `output/anime_metadata/` for:
- **Character recommendations**: subtitles in single-character scenes get 90% confidence auto-suggestions; multi-character scenes use pipeline speaker attribution
- **Frame analysis** (A key): instant lookup of which characters GPT-4o identified at the current timestamp
- **Character list**: populated from VLM scene data (no reference images or CLIP needed)

Workflow: select episode → accept/reject recommendations → right-click subtitles to quick-tag → extract audio clips → evaluate voice clone quality.

Previously used CLIP ViT-B-32 + YOLO face detection for character matching, but this was removed because CLIP similarity scores for anime faces were pure noise (Luffy ranked #779 out of 887 reference characters). The VLM approach is free (data already extracted), instant, and far more accurate.

---

## Episode 1 Test Run Results

Ran on `[Anime Time] One Piece - 0001 - I'm Luffy! The Man Who's Gonna Be King of the Pirates!.mkv`:

| Stage | Result | Time |
|-------|--------|------|
| 1. Subtitles | 302 segments (ASS) | <1s |
| 2. Shot detection | 411 shots, 111 fades | 534s |
| 3. Features | 230 dialogue shots, 41 flashback candidates | 339s |
| 4. Scene detection | 88 scenes, 1 eyecatch | <1s |
| 5. VLM analysis | 79/88 scenes analyzed ($0.39) | 1105s (at 20 RPM; now 100 RPM) |
| 6. Content origin | 363 anime_original, 31 transition, 17 action_extension | <1s |
| 7. Dialogue index | 301 entries, 21 speakers | <1s |
| 8. Output files | 88 scene JSONs + shot index + character registry | <1s |
| 9. Manifest | episode_manifest.json | <1s |

Known issues from test run (pre-wiki-grounding):
- Speaker attribution defaults to first character in scene when multiple are present (low confidence) — **addressed by VLM-based speaker attribution (Phase 2)**
- Content origin is all anime_original without manga manifest for cross-referencing — **addressed by VLM-based content origin (Phase 2)**
- ~9 scenes got empty VLM data due to OpenAI quota exhaustion mid-run (resumable)
- GPT-4o uses inconsistent character names across scenes (e.g. "Coby" vs "Koby") — **addressed by wiki character list as closed set**
- 41% of scenes had no named characters, only generic labels ("unknown", "Character 1") — **addressed by anti-generic-name rules in prompt**
- Character hallucinations (e.g. "Albedo" instead of "Alvida") — **addressed by closed character list**
- OP theme sequence not detected as opening — **addressed by adding `opening`/`ending` to scene_type enum**

---

## Deliverables

- `anime_metadata.py` — main pipeline orchestrator, runs all steps in order
- `scene_detector.py` — shot→scene grouping as a standalone tunable module
- `requirements.txt`
