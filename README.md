# Manga-Anime Pipeline

AI-driven pipeline for One Piece (and eventually other series) that extracts structured metadata from both manga and anime, aligns them, and uses that data for two end goals:

1. **Manga-to-Anime generation** -- alignment + video generation from manga source
2. **Manga Player** -- voiced manga reader with cloned character voices, music, and short anime clips

## Two Sub-Projects

### Sub-Project 1: Extraction + Alignment + Generation

Extract structured metadata from manga chapters and anime episodes independently, then align them. The alignment maps manga panels to anime timestamps, enabling training data for manga-to-anime generation.

```
manga pages ----> kumiko_panels.py ----> panel bboxes (OpenCV pre-pass)
                        |
                        v
manga pages ----> manga_metadata.py ----> structured JSON (panels, dialogue, scenes)
                                                |
anime .mkv  ----> anime_metadata.py ----> structured JSON (shots, scenes, dialogue)
                                                |
                        +----------+------------+
                        |
                        v
                  alignment (planned)
                        |
                        v
                  video generation (planned)
```

### Sub-Project 2: Manga Player

A voiced manga reader that uses anime reference audio to clone character voices via XTTS v2. Reads manga pages aloud with per-character voices, emotion-aware prosody, pacing between panels, and eventually background music and short anime clips at key moments.

```
manga_metadata.py ----> script_builder.py ----> tts_synthesize.py ----> voiced audio
                              |                        ^
                              v                        |
                     voice_tagger.py ----> voice_clone.py (XTTS v2)
                     (tag subs to chars)   (reference audio per character)
                              |
                              v
                         reader.py (web manga reader with TTS playback)
```

Both sub-projects share the extraction layer (manga_metadata, anime_metadata, scene_detector), the wiki data layer (one_piece_wiki_api), and the voice pipeline (voice_tagger, voice_clone).

---

## Scripts

### Wiki Data Layer

**`one_piece_wiki_api.py`** -- On-demand API client for the One Piece Fandom Wiki (MediaWiki API).
Provides structured episode, character, and arc data without bulk scraping. Each function makes
one or two HTTP GET requests to `https://onepiece.fandom.com/api.php` and parses the response.
No API key required. Zero dependencies beyond Python stdlib (`urllib`, `json`, `html.parser`).

Episode functions:
- `get_episode_characters(episode_num)` -- List of characters with names, wiki pages, and annotations (e.g. "flashback"). Parses `*[[Character Name]]` wikitext from the "Characters in Order of Appearance" section.
- `get_episode_summary(episode_num)` -- Full plain-text narrative summary (Long Summary section).
- `get_episode_short_summary(episode_num)` -- One-sentence summary (Short Summary section).
- `get_episode_infobox(episode_num)` -- Episode metadata: titles (kanji/romaji/English), air dates, production credits, character/location/technique/devil fruit debuts, TV ratings.
- `get_episode_data(episode_num)` -- Convenience: returns all of the above in a single dict.

Character functions:
- `get_character_info(name)` -- Structured infobox data from rendered HTML (`portable-infobox`). Returns affiliations, devil fruit, age, height, bounty, voice actors, image URL, etc. Works for both simple characters (direct `{{Char Box}}`) and major characters with tabbed pages (Luffy, Zoro).
- `get_character_image(name, thumb_size=300)` -- Direct CDN thumbnail URL via the `pageimages` query property.

Arc functions:
- `get_arc_summary(arc_name)` -- Arc narrative summary broken into named story beats (sub-sections). Returns list of `{"heading": str, "text": str}`.
- `get_arc_ranges(arc_name)` -- Chapter/episode/volume ranges from the arc infobox. Returns structured dicts with `start`, `end`, `count` fields, plus `prev_arc`/`next_arc` navigation.
- `get_story_arcs()` -- Full saga-to-arc-to-episode/chapter mapping from the Story Arcs index page. Returns `[{"saga": str, "arcs": [{"name", "wiki_page", "chapters", "episodes"}, ...]}]`.
- `get_arc_for_episode(episode_num)` -- Reverse lookup: episode number to arc name, saga, episode range, and chapter range.

Pipeline integration points:
- **anime_metadata.py Stage 5** (VLM): Episode character list constrains GPT-4o identification ("pick from this list" vs "guess who this is"). Episode summary provides narrative context.
- **anime_metadata.py Stage 6** (content origin): Episode summary and chapter manifest provide manga-vs-anime context.
- **anime_metadata.py Stage 7** (speaker attribution): Known character list improves attribution confidence.
- **voice_tagger_backend.py**: Character list + images improve tagging suggestions.
- **character_registry.json**: Enriched with wiki data (affiliations, devil fruit, description).
- **Alignment module** (planned): Arc ranges provide direct chapter-to-episode mapping.

### Extraction Layer (shared)

**`manga_metadata.py`** -- Extracts structured metadata from manga chapter images using GPT-4o vision.
Processes each page to identify panels, dialogue bubbles, narration boxes, sound effects, characters,
locations, and time context. Produces per-page JSONs, a character registry, and a chapter manifest
with a flat `dialogue_index` used by both alignment and the TTS script builder. Handles reading
order (right-to-left), speaker attribution, and emotion detection. Integrates Kumiko panel detection
as a pre-pass: when `output/kumiko_panels/{chapter_id}/kumiko_panels.json` exists, injects precise
OpenCV-detected panel boundaries into the VLM prompt and overrides VLM bboxes via IOU matching.
Panels are always sorted into guaranteed RTL reading order after validation. Use `--no-kumiko` to disable.

**`kumiko_panels.py`** -- OpenCV-based panel detection pre-pass using the vendored Kumiko library.
Detects panel bounding boxes from manga page images via contour analysis. Produces normalized
`[x1,y1,x2,y2]` bboxes per page in RTL reading order. Run before `manga_metadata.py` for
more precise panel crops. Takes ~2 seconds per chapter. Output: `output/kumiko_panels/{chapter_id}/kumiko_panels.json`.

**`anime_metadata.py`** -- 9-stage pipeline that extracts structured metadata from anime episodes.
Stages: subtitle extraction from embedded ASS/SRT tracks, TransNetV2 shot boundary detection,
per-shot OpenCV feature extraction (optical flow, color histograms, dialogue overlay), scene grouping
via multi-signal scoring, GPT-4o vision analysis per scene (narrative, characters, location, emotion),
content origin classification (manga-derived vs anime-original), speaker attribution, and output
assembly. Stage 5 (VLM) is grounded with wiki episode context: the VLM prompt receives a closed
character list ("use ONLY these names"), the full episode long summary (~1900 chars), and location
vocabulary from `one_piece_wiki_api`, plus previous scene summary for continuity (framed as
optional — may be a different scene). Currently sends 1 keyframe per scene; multiple keyframes
planned for better coverage of action/battle scenes. Produces per-scene JSONs, shot index,
character registry, dialogue index, and episode manifest.

**`scene_detector.py`** -- Standalone module for shot detection and scene grouping, used by anime_metadata.py.
Runs TransNetV2 (CNN) for shot boundary detection, detects fades (fade-to-black, fade-to-white, dissolve)
via frame brightness analysis, extracts per-shot features using OpenCV (Farneback optical flow, HSV color
histograms), and groups shots into scenes using weighted multi-signal scoring (color distance, audio
discontinuity, dialogue gaps, cast changes). Also detects mid-episode eyecatch title cards and flashback
candidates via saturation analysis. All thresholds tunable via CONFIG dict.

### Voice Pipeline (shared)

**`voice_tagger.py`** -- Flask web UI (port 7861) for tagging anime subtitles to characters.
Serves the episode video with HTTP range request support for seeking, displays subtitles
on a timeline, and lets the user assign each subtitle line to a character. Uses pre-computed
GPT-4o scene data from anime_metadata for VLM-powered character recommendations (single-character
scenes get high-confidence auto-suggestions). Replaces the earlier CLIP + YOLO approach which
produced pure noise on anime faces.

**`voice_tagger_backend.py`** -- Backend for the voice tagger web UI, handles all non-HTTP logic.
Loads pre-computed VLM scene analysis from anime_metadata output, provides character identification
at any timestamp, manages background jobs for audio extraction (ffmpeg), and tracks processing
state via threading. Extracts per-character audio clips from MKV files targeting ~90s of reference
audio per character for voice cloning.

**`voice_clone.py`** -- Extracts per-character reference audio clips and manages voice synthesis via XTTS v2.
Builds a voice registry mapping character IDs to reference audio files with emotional labels.
Handles reference clip selection (preferring clean, emotion-diverse samples), model loading,
and per-line synthesis with emotion-aware voice conditioning. Shares CHARACTER_ALIASES with
the manga pipeline for consistent character ID normalization.

**`script_builder.py`** -- Converts manga metadata into a TTS-ready chapter script with prosody hints.
Reads per-page JSONs and voice registry, resolves each text element to a voice key, computes
speed/volume/pause values based on bubble type (shout, whisper, thought), emotion modifiers,
and panel/page position. Preprocesses text for TTS (normalize caps, add punctuation cues,
clean manga artifacts). Outputs a chapter_script.json with every line ready for tts_synthesize.

**`tts_synthesize.py`** -- Orchestrates XTTS v2 synthesis for an entire chapter script.
Reads the chapter script from script_builder, loads the TTS model and voice registry,
synthesizes each line with the assigned voice and prosody hints (speed, volume), applies
post-processing (normalization, silence trimming), and saves individual WAV files per line.
Can optionally concatenate all lines into a single chapter audio file with appropriate pauses.

### Manga Player

**`reader.py`** -- Flask web app (port 5000) that serves manga pages with integrated TTS playback.
Discovers all available chapters from `manga/` directory and presents a chapter selector dropdown
in the top bar. Loads manga images and their metadata JSONs per-chapter (cached for fast switching),
renders pages in a browser with panel navigation, and uses browser-based text-to-speech (or
pre-synthesized audio from tts_synthesize) to read dialogue aloud. Chapter titles are read from
manifests when available. Supports `--chapter 0002` CLI arg for startup default. Natural sort for
page filenames. Intended end-product of Sub-Project 2.

### Utilities / Legacy

These scripts were part of the original approach before we discovered that GPT-4o VLM scene
analysis is both cheaper and far more accurate than the CLIP + pyannote stack for anime.
The original plan was: extract audio -> diarize speakers -> identify characters via CLIP face
matching against a scraped reference database. This failed because (a) CLIP ViT-B-32 similarity
scores for anime faces are pure noise -- all characters scored within a 0.26 spread, with Luffy
ranking #779 out of 887 reference characters, and (b) pyannote speaker diarization on anime
audio mislabels speakers frequently due to music, sound effects, and voice acting range.
The current approach extracts subtitles directly from the MKV (free, instant, accurate) and uses
GPT-4o VLM to identify characters per scene from keyframe images (already paid for in stage 5).

**`ground_truth.py`** -- Generates speaker diarization ground truth using AssemblyAI API.
Sends anime audio to AssemblyAI for high-quality speaker diarization ($0.006/min, covered by
$50 free credits). Used to validate and benchmark the open-source pyannote pipeline. Produces
labeled speaker segments with timestamps for comparison against local diarization output.
*Legacy: was useful for benchmarking pyannote; no longer needed since we use subtitles + VLM.*

**`speaker_diarization.py`** -- Speaker diarization pipeline using Demucs + pyannote 3.3.2.
Extracts audio from MKV, separates vocals via Demucs, runs pyannote for speaker segmentation,
applies VAD filtering and energy thresholding. Originally the anime-side audio pipeline.
*Legacy: replaced by subtitle extraction in anime_metadata.py (embedded ASS/SRT tracks are
free and more accurate than Whisper). The Demucs vocal separation is still potentially useful
for extracting clean voice-only tracks for voice cloning reference audio.*

**`identify.py`** -- Visual character identification using CLIP ViT-B-32 + YOLOv8 face detection.
Matches speaker segments to character names by extracting face crops from video frames at
speaker timestamps, computing CLIP embeddings, and comparing against reference character images.
*Legacy: replaced by VLM-based identification in voice_tagger_backend.py. CLIP similarity for
anime faces was pure noise -- all 887 characters scored within a 0.26 spread, making ranking
meaningless. GPT-4o correctly names characters from a single keyframe.*

**`scrape_references.py`** -- Scrapes character reference images from AnimeCharactersDatabase.
Downloads one thumbnail per character for the CLIP matching pipeline in identify.py. Builds
a character_catalog.json with metadata (name, image URL, anime association). Includes retry
logic and image filtering.
*Legacy: only needed by the CLIP pipeline in identify.py. The VLM approach needs no reference
images -- GPT-4o recognizes One Piece characters from its training data.*

**`ui.py`** -- Gradio web UI for pipeline validation and interactive speaker tagging.
Loads speaker frame/crop data, character references, and identification results.
Provides an interface for reviewing and correcting automated speaker assignments.
*Legacy: superseded by voice_tagger.py which has better UX, video streaming with seeking,
and VLM-powered character recommendations instead of CLIP scores.*

---

## Data Flow

```
Input:
  0001/              Manga chapter pages (001.png - 052.png)
  anime/*.mkv        Anime episode files

Output:
  output/kumiko_panels/0001/
    kumiko_panels.json                 OpenCV panel bboxes per page (pre-pass)

  output/manga_metadata/0001/
    page_001.json ... page_052.json    Per-page extracted metadata (bbox_source: kumiko/vlm)
    character_registry.json            Character database
    chapter_manifest.json              Manifest with dialogue_index

  output/anime_metadata/0001/
    episode_manifest.json              Top-level summary + dialogue index
    shot_index.json                    Every shot with origin classification
    character_registry.json            All characters with appearances
    dialogue_index.json                Flat dialogue sequence
    scene_001.json ... scene_088.json  Per-scene metadata
    keyframes/                         Establishing shot keyframes
    subtitles.ass                      Extracted subtitle file
    subtitles_raw.json                 Parsed subtitle segments
    shots_raw.json                     TransNetV2 shot boundaries
    shots_features.json                OpenCV features per shot
    scenes_raw.json                    Scene groupings
    scenes_vlm.json                    Scenes with VLM analysis

  output/voice_tags/{episode_id}/
    tags.json                          Subtitle-to-character tags
    voice_samples/{character}/*.wav    Per-character audio clips

  output/tts_scripts/0001/
    chapter_script.json                TTS-ready script

  output/tts_output/0001/
    *.wav                              Synthesized audio per line
```

## API Costs

- **Manga extraction** (GPT-4o vision): ~$1.50 per chapter (52 pages)
- **Anime extraction** (GPT-4o vision): ~$0.40 per episode (88 scenes)
- **Subtitles**: Free (extracted from embedded MKV tracks)
- **Wiki data**: Free (MediaWiki API, no key, no rate limit concerns at pipeline scale)
- **1000 episodes**: ~$400 total for anime extraction

## Current Status

- Manga metadata extraction: complete (with Kumiko panel detection integration)
- Anime metadata extraction: complete (tested on ep1: 88 scenes, 411 shots, 301 dialogue lines)
- Wiki API client: complete (episodes, characters, arcs -- on-demand, no scraping)
- Voice tagger UI: working (VLM-powered recommendations)
- Script builder: complete
- Voice cloning: infrastructure ready, needs tagged audio
- Wiki integration into anime_metadata: Stage 5 VLM grounding complete (character list, synopsis, locations, previous scene context)
- Alignment: planned
- Manga player: working (reader.py -- chapter selector, 3-panel UI, TTS, panel overlays)
- Video generation: planned
