# Project: Manga-Anime Pipeline

AI pipeline for One Piece that extracts structured metadata from manga chapters and anime episodes, aligns them, and builds two products: (1) manga-to-anime generation training data, (2) a voiced manga reader with cloned character voices.

## Project Structure

```
wiki/                   Wiki API clients (foundation, no local deps)
extraction/             Core extraction pipelines + debug scripts
alignment/              3-pass alignment pipeline
voice/                  Voice pipeline (TTS, cloning, tagging)
viewers/                Web UIs (Flask apps)
utils/                  Validation & comparison tools
legacy/                 Superseded modules
manga/                  Input manga page images (by chapter ID)
anime/                  Input anime video files (MKV)
output/                 All pipeline outputs
vendor/                 Vendored dependencies (kumiko)
docs/                   Research notes
generation_research/    Generation stage research plans
```

## Two Sub-Projects

### 1. Extraction + Alignment + Generation
- `extraction/manga_metadata.py` extracts panels/dialogue/scenes from manga pages via GPT-4o vision
- `extraction/anime_metadata.py` is a 9-stage pipeline (subtitles, TransNetV2 shots, OpenCV features, scene grouping, GPT-4o VLM scene analysis, content origin, dialogue attribution, output files, manifest)
- `extraction/scene_detector.py` is the shot/scene detection module used by anime_metadata.py
- Alignment (planned): correlate manga `dialogue_index` with anime `dialogue_index` via fuzzy text matching + location/time anchoring
- Video generation (planned): use aligned data to generate anime from manga

### 2. Manga Player (voiced reader)
- `voice/script_builder.py` converts manga metadata into TTS-ready scripts with prosody hints
- `voice/voice_tagger.py` + `voice/voice_tagger_backend.py` is a Flask UI for tagging anime subtitles to characters (3-panel layout with VLM scene metadata, video, and character list; recommendations below subtitles; tag-based learning loop)
- `voice/voice_clone.py` manages XTTS v2 voice cloning with per-character reference audio
- `voice/tts_synthesize.py` synthesizes voiced audio for each script line
- `viewers/reader.py` is the web-based manga reader with chapter selector, TTS playback, panel bbox overlays, narrative sidebar, and page metadata display

## Architecture

```
Wiki data (on-demand):
  wiki/one_piece_wiki_api.py --> episode characters, summaries, anime notes
                              --> character info, images, infobox data
                              --> arc summaries, ranges, manga differences
                              --> episode-to-chapter mapping (hardcoded, from listfist.com)
  wiki/manga_wiki_api.py     --> chapter characters (grouped), summaries, notes
                              --> chapter infobox, cover page, arc lookup
                              (both use MediaWiki API, no scraping, no API key needed)

Extraction (shared):
  manga pages --> extraction/manga_metadata.py --> page JSONs + chapter_manifest.json
                  (uses wiki/manga_wiki_api.py for VLM grounding)
  anime/*.mkv --> extraction/anime_metadata.py --> scene JSONs + episode_manifest.json
                  (uses extraction/scene_detector.py + wiki/one_piece_wiki_api.py for VLM grounding)

Voice pipeline (shared):
  voice/voice_tagger.py --> tags subtitles to characters (uses VLM scene data)
  voice/diarize_segments.py --> VLM-guided per-segment pyannote diarization (Demucs vocals, speaker groups)
  voice/voice_clone.py  --> extracts reference audio, manages XTTS v2

Alignment:
  manga metadata + anime metadata --> alignment/alignment.py --> alignment_manifest.json
                                      (uses alignment/pass1_dialogue_anchor.py + alignment/pass2_gap_fill.py + alignment/reconciliation.py)

Sub-project 1 (generation):
  alignment dataset --> video generation
  image sequence --> generation_research/generate_sequence.py --> output/generation/{timestamp}/
                     (uses Wan2GP via subprocess for FLF2V/I2V, GPT-4o for AI prompts)

Sub-project 2 (manga player):
  manga metadata -----> viewers/reader.py (panels, bbox, characters, locations, narratives, dialogue)
  manga metadata --> voice/script_builder.py --> voice/tts_synthesize.py --> viewers/reader.py (TTS audio)
```

## Key Files

| File | Role | Status |
|------|------|--------|
| `extraction/manga_metadata.py` | Manga extraction (GPT-4o vision, wiki-grounded) | Complete |
| `extraction/anime_metadata.py` | Anime extraction (9-stage pipeline) | Complete |
| `extraction/scene_detector.py` | Shot/scene detection (TransNetV2 + OpenCV + optional GPU models + optional audio classifier) | Complete |
| `extraction/kumiko_panels.py` | OpenCV-based panel detection (Kumiko), pre-pass for manga extraction | Complete |
| `wiki/one_piece_wiki_api.py` | Wiki API client for episodes/characters/arcs | Complete |
| `wiki/manga_wiki_api.py` | Wiki API client for chapters (characters, summaries, notes) | Complete |
| `alignment/alignment.py` | Alignment pipeline orchestrator (manga-to-anime transformation dataset) | Complete |
| `alignment/pass1_dialogue_anchor.py` | Dialogue matching via dual embeddings (OpenAI + BOW) | Complete |
| `alignment/pass2_gap_fill.py` | LLM gap filling and delta extraction | Complete |
| `alignment/reconciliation.py` | Post-processing: normalize, validate, compute stats | Complete |
| `voice/voice_tagger.py` | Subtitle-to-character tagging UI (3-panel: scene metadata + video + characters, inline recommendations, overlap details, confidence badges, port 7861) | Working |
| `voice/voice_tagger_backend.py` | VLM-powered tagging backend (scene lookup, recommendations with dismiss/learning loop, tag corrections via process of elimination) | Working |
| `voice/diarize_segments.py` | VLM-guided per-segment speaker diarization (Demucs vocal isolation + pyannote 3.1 + VLM speaker count hints + cluster-to-character resolution) | Working |
| `voice/voice_clone.py` | XTTS v2 voice cloning | Infrastructure ready |
| `voice/script_builder.py` | Manga-to-TTS script conversion | Complete |
| `voice/tts_synthesize.py` | TTS synthesis orchestration | Complete |
| `viewers/reader.py` | Web manga reader (chapter selector, 3-panel UI: narrative + image with bbox overlays + dialogue, TTS, page jump, `--chapter` CLI arg) | Working |
| `viewers/anime_viewer.py` | Web anime metadata viewer (video + scene/shot timeline + keyframes/crops + detail panels, `--episode` CLI arg, port 7862) | Working |
| `viewers/alignment_viewer.py` | Web alignment review UI (side-by-side manga/anime, `--episode` CLI arg, port 7863) | Working |
| `utils/validate_rerun.py` | Baseline vs new anime metadata comparison | Utility |
| `utils/validate_multi_keyframe.py` | Single-keyframe vs multi-keyframe comparison | Utility |
| `utils/compare_manga_output.py` | Baseline vs new manga metadata comparison (panels, bboxes, IOU, Kumiko stats) | Utility |
| `utils/ground_truth.py` | AssemblyAI diarization ground truth | Utility |
| `utils/diarize_compare.py` | Multi-service diarization comparison | Utility |
| `generation_research/generate_sequence.py` | Image sequence -> video generation with `--backend wan2gp` (local GPU, subprocess) or `--backend kling` (Kling AI cloud API, JWT auth, polling). GPT-4o AI prompts via `--aiprompt`, ffmpeg concat, retry, resume | Working |
| `generation_research/WAN2GP_SETUP.md` | Wan2GP setup guide, GPU compatibility, pipeline usage docs | Documentation |
| `legacy/speaker_diarization.py` | Legacy pyannote diarization | Superseded |
| `legacy/identify.py` | Legacy CLIP character identification | Superseded |
| `legacy/scrape_references.py` | Legacy character image scraper | Superseded |
| `legacy/ui.py` | Legacy Gradio UI | Superseded |

## Important Technical Details

- **No Whisper**: Subtitles are extracted from embedded MKV tracks (ASS/SRT), not transcribed. Free, faster, more accurate for fansubs.
- **No CLIP for character ID**: GPT-4o VLM scene analysis replaced CLIP+YOLO face detection because CLIP similarity was pure noise on anime faces. However, CLIP is used for background embeddings (scene grouping signal) via `--use-clip`, and YOLOv8-seg is used for character segmentation/background plates via `--use-yolo` — both optional GPU features.
- **Character IDs**: All components use the same snake_case normalization via CHARACTER_ALIASES (e.g. "Red-Haired Shanks" -> "shanks", "Monkey D. Luffy" -> "luffy").
- **Wiki grounding**: Both manga and anime pipelines fetch character lists, summaries, and metadata from the One Piece Fandom Wiki at startup. Character lists are passed as closed sets ("use ONLY these names") to eliminate hallucinated/generic labels.
- **Arc IDs**: Standardized snake_case arc identifiers (e.g. "romance_dawn", "alabasta", "water_7") via `normalize_arc_id()` in `wiki/one_piece_wiki_api.py`. Present on manga chapter manifests, anime episode manifests, and per-scene JSONs. Anime pipeline resolves arc via `get_arc_for_episode()` (Story Arcs page), since episode infoboxes never contain an `arc` field.
- **Arc data (future enrichment)**: `wiki/one_piece_wiki_api.py` exposes rich per-arc data not yet used in VLM prompts:
  - `get_arc_for_episode(N)` — saga name, arc name, wiki page, episode range, chapter range
  - `get_arc_summary("Romance Dawn Arc")` — full narrative summary broken into headed sections
  - `get_arc_ranges("Romance Dawn Arc")` — structured start/end for volumes, chapters, episodes + next arc
- **Location vocabulary**: Manga chapter manifests include a deduplicated `locations` list aggregated from all panels. The anime pipeline reads this list as controlled vocabulary for VLM location labeling.
- **Resume support**: Both pipelines cache intermediate outputs and skip completed work with `--resume`.
- **Rate limiting**: VLM calls are rate-limited to 20/minute to stay under OpenAI API limits.
- **Optional GPU models**: Both pipelines support optional GPU features. Anime: `--use-yolo` (character crops + background plates), `--use-clip` (background embeddings for scene grouping). Manga: `--use-depth` (MiDaS depth maps on panel crops), `--use-segmentation` (DeepLabV3 segmentation masks on panel crops). All GPU features are independently toggleable and the pipelines work without them.
- **Audio classifier**: Anime pipeline supports `--use-audio` for librosa-based per-shot audio classification (music, ambient, silence detection). Extracts audio via ffmpeg once, then classifies each shot using spectral analysis (harmonic ratio, spectral flatness, onset rate, RMS energy). Music cuts and silence onsets are pro-break signals. Music continuity is a weak anti-break signal (`weight_music_continuity=0.03`) — reduced from 0.15 because anime BGM routinely spans scene changes, causing over-merging (88→32 scenes at 0.15). Optional — requires librosa + numba compatible with installed numpy (numba 0.64.0+ for numpy 2.x). Pipeline works without it (falls back to subtitle-based audio features).
- **Multi-keyframe extraction**: Adaptive per-shot keyframe count based on duration — shots <3s get 1 keyframe, 3-6s get 2, >6s get 1 per ~2.5s. Config: `keyframe_short_threshold` (3.0), `keyframe_medium_threshold` (6.0), `keyframe_long_interval` (2.5) in `extraction/scene_detector.py`. Schema uses `keyframe_paths` (list), not singular `keyframe_path`. YOLOv8 runs on all keyframes with `bbox_area_fraction` per crop. CLIP stays single-frame (primary only).
- **Multi-image VLM calls**: Stage 5 sends multiple per-shot keyframes to GPT-4o, duration-scaled: scenes <15s get all shots' keyframes, 15-60s get 1 per ~5s, >60s get 1 per ~8s. Config: `vlm_short_scene_threshold`, `vlm_medium_scene_interval`, `vlm_long_scene_threshold`, `vlm_long_scene_interval` in `extraction/anime_metadata.py`. Scene keyframe extraction from video removed — all keyframes come from per-shot Stage 3 extraction.
- **VLM dialogue attribution**: `text_elements` in scene JSONs are NOT independently detected from keyframes — they are 1:1 with subtitle lines. Stage 5 sends each scene's keyframes + the full subtitle list to GPT-4o, which returns `dialogue_attribution` (speaker guess per subtitle line). Stage 7 (`build_dialogue_index()`) combines this: each subtitle becomes a dialogue_index entry with the subtitle's own timestamps (`start_time_seconds = seg_start`) and the VLM's speaker guess. Stage 8 writes these as `text_elements` in scene JSONs. So VLM "speaker" = "GPT-4o guessed who said each subtitle line from a few keyframes + scene character list" — a visual inference, not an audio signal. Coverage is ~90% (subtitles in non-recap/non-eyecatch scenes). This is complementary to audio-based speaker diarization (AssemblyAI), which groups lines by voice similarity without knowing character names.
- **Adaptive synopsis windowing**: Manga VLM prompts include only the synopsis paragraphs relevant to the current page, not the full chapter synopsis. `get_relevant_synopsis()` in `extraction/manga_metadata.py` splits the wiki synopsis on `\n\n` paragraph boundaries and uses proportional mapping (page position → paragraph position) with an adaptive window radius based on the pages-per-paragraph ratio: dense synopses (many paragraphs, e.g. ch 1: 18 paras / 53 pages → radius 1, 3 paragraphs shown) get narrow windows, sparse synopses (few paragraphs, e.g. ch 200: 3 paras / 19 pages → radius 2) naturally degrade to sending the full text. Short synopses (<=1500 chars or <=2 paragraphs) are always sent in full.
- **Kumiko panel detection**: `extraction/kumiko_panels.py` runs OpenCV-based panel detection (vendored from `vendor/kumiko/`) as a pre-pass before manga extraction. Produces `output/kumiko_panels/{chapter_id}/kumiko_panels.json` with normalized `[x1,y1,x2,y2]` bboxes per page. `manga_metadata.py` auto-detects this data at startup and integrates it in two ways: (1) injects exact panel boundaries into the VLM prompt so GPT-4o knows panel count and positions, (2) always overrides VLM-returned bboxes with Kumiko bboxes in post-validation via IOU matching (`match_panels_by_iou()`, greedy, threshold 0.2). Each panel is tagged with `bbox_source: "kumiko"` or `"vlm"`. Kumiko uses RTL reading order by default (manga convention). Falls back to VLM-only detection when Kumiko data is absent. Disable with `--no-kumiko`.
- **Voice tagger UI**: Triage-first layout — topbar (episode select, progress bar, tier counts, Accept All Tier 1, filter toggles, S/C panel toggles), main area (toggleable scene panel + video 30% + vertical subtitle list + toggleable character panel). Subtitles grouped by scene with sticky headers. Each row shows inline signals: tagged badge, VLM suggestion with Accept/Chg buttons, or scene character chips. Three tiers: Tier 1 (auto-confirmable, VLM confidence >= 0.8 or single-char scene), Tier 2 (needs review, 0.4-0.79), Tier 3 (manual, no match). `get_enriched_subtitles()` pre-matches every subtitle to VLM text_elements by timestamp proximity (<1.5s), computes tiers, handles OP/ED auto-labeling. `bulk_accept_tags()` batch-tags all Tier 1 in one API call. Keyboard-driven: Y accept, J/K navigate untagged, 1-9 quick-assign scene chars, N open picker, F filter untagged, S/C toggle panels. Inline character picker with scene chars, recent, and searchable all-chars sections. Process of elimination in 2-3 character scenes via `_apply_enriched_corrections()` — only overrides weak/absent suggestions (score < 0.5); strong VLM attributions are preserved, not replaced. `_apply_tag_corrections()` in recommendations endpoint follows the same rule. Scene/char panels hidden by default, state in localStorage.
- **VLM-guided diarization**: `voice/diarize_segments.py` runs pyannote 3.1 speaker diarization per scene segment with VLM-derived speaker count hints (`min_speakers`, `max_speakers`). Pipeline: (1) Demucs vocal isolation from episode MKV, (2) segment computation from scene boundaries (long scenes >120s sub-split at subtitle gaps, short <10s merged), (3) ffmpeg slice of vocals per segment (16 kHz mono), (4) pyannote diarization with speaker bounds, (5) turn-to-subtitle mapping by timestamp overlap, (6) cluster-to-character resolution by VLM vote purity (clean >=80%, likely >=60%, impure <60%), (7) result.json output. All local GPU, no API cost. Requires pyannote.audio 3.1+, Demucs, torch, HuggingFace token. See `voice/DIARIZATION_PLAN.md` for design rationale.
- **Eyecatch detection**: `detect_eyecatch()` in `extraction/scene_detector.py` identifies the mid-episode eyecatch (title card break). Primary method: subtitle-gap analysis — finds the most prominent dialogue-free gap near the episode midpoint, scored by `gap_duration * (1 - distance_to_midpoint / window)`. Longer gaps closer to midpoint score highest. All shots fully contained within the gap are flagged as eyecatch. Fallback (standalone mode, no subtitles): strict per-shot brightness + fade_to_black/fade_to_white heuristic. Config: `eyecatch_min_gap_duration` (4s), `eyecatch_max_gap_duration` (30s), `eyecatch_midpoint_window` (±15%). In `anime_metadata.py`, scenes containing both eyecatch and non-eyecatch shots are split into separate sub-scenes to prevent contamination. Eyecatch scenes skip VLM analysis (Stage 5) and dialogue index (Stage 7).
- **Panel reading order enforcement**: `sort_panels_reading_order()` in `extraction/manga_metadata.py` runs at the end of `validate_page_json()` on every page. Groups panels into rows by vertical overlap (>50% of shorter panel height), sorts right-to-left within each row (manga RTL convention), and updates `reading_order` fields to match. Guarantees downstream consumers (crops, video generation) receive panels in correct reading sequence regardless of VLM output order.

## Output Directories

```
output/kumiko_panels/{chapter_id}/    Kumiko panel detection JSON (pre-pass for manga extraction)
output/manga_metadata/{chapter_id}/   Per-page JSONs + chapter_manifest + character_registry + panels/ (crops, depth maps, seg masks)
output/anime_metadata/{episode_id}/   Scene JSONs + shot_index + episode_manifest + keyframes/ (shot keyframes, bg plates, character crops)
output/alignment/{episode_id}/        Alignment manifest + index + pass1 anchors + llm_cache/
output/voice_tags/{episode_id}/       Subtitle tags + voice samples per character
output/voice_tags/{episode_id}/diarization/  Demucs vocals + per-segment WAVs + diarization result.json
output/tts_scripts/{chapter_id}/      TTS-ready chapter scripts
output/tts_output/{chapter_id}/       Synthesized WAV files per line
output/generation/{timestamp}/        Generated video clips + settings + final_sequence.mp4
```

## APIs Used

- **OpenAI GPT-4o** (vision): manga page extraction, anime scene analysis (~$1.00-$1.50/episode with multi-keyframe, ~$1.50/chapter)
- **One Piece Fandom Wiki** (MediaWiki API): episode/chapter/character/arc metadata, on-demand via `wiki/one_piece_wiki_api.py` (episodes) and `wiki/manga_wiki_api.py` (chapters). No API key, no scraping -- stdlib only (`urllib`, `json`, `html.parser`).
- **AssemblyAI** (optional): speaker diarization ground truth ($50 free credits)

## Conventions

- Episode/chapter IDs are zero-padded 4-digit strings: "0001", "0002", etc.
- All JSON output uses 2-space indent, UTF-8, no ASCII escaping.
- Manga reading order is right-to-left (Japanese convention).
- Pipeline progress prints ASCII progress bars (no Unicode -- Windows cp1252 compatibility).
- Temporary test scripts are prefixed with `_run_` (e.g. `extraction/_run_stage5.py`).
- Manga chapter source images live under `manga/{chapter_id}/` (e.g. `manga/0001/`).
- **All scripts must be run from the project root directory** (not from within subfolders). Data paths (output/, manga/, anime/) resolve relative to CWD.
- Each subfolder module includes a `sys.path` preamble for import resolution. All cross-module imports use absolute package-qualified names (e.g. `from wiki.one_piece_wiki_api import ...`).
