# Extraction Research Implementation Plan

Goal: incrementally bring the anime and manga pipelines up to the generation research spec, while preserving implementation strengths the spec missed. After all phases, the research MDs should reflect the best of both.

Scope: Stage 1 (manga extraction) and Stage 2 (anime extraction) only. Stage 3 (alignment) and Generation are out of scope.

---

## Section 1 — Comparison: Spec vs Implementation

### Stage 1 — Manga Extraction

**Implemented and matching spec:**
- Per-page JSON with full schema (page_type, page_narrative, panels, text_elements, sound_effects)
- Character registry with visual_description, appearances, voice_profile
- Chapter manifest with dialogue_index, statistics, processing_metadata
- Wiki grounding via manga_wiki_api.py
- Consistent character ID normalization via CHARACTER_ALIASES
- Resume support

**Gaps (spec has, implementation lacks):**

| Field | Purpose | Needed by |
|---|---|---|
| `panel_image_path` | Cropped panel images | Alignment visual conditioning, generation |
| `depth_map_path` | Depth estimation per panel crop | Generation (ControlNet spatial control) |
| `segmentation_map_path` | Segmentation mask per panel crop | Generation (ControlNet spatial control) |
| `arc_id` (standardized snake_case) | Conditioning signal, e.g. `"east_blue"` | All downstream stages |
| `locations` list on manifest | Controlled vocabulary for anime VLM | Anime extraction Stage 2 |
| `panel_image_path` in dialogue_index | Colorization reference pairs | Alignment Stage 3 |

### Stage 2 — Anime Extraction

**Implemented and matching spec:**
- 9-stage pipeline in correct order
- Subtitles from embedded tracks (no Whisper)
- TransNetV2 shot boundaries with fade detection (black/white/dissolve)
- Optical flow motion intensity (Farneback, 0.0-1.0)
- Color histogram per shot (HSV)
- Flashback detection via saturation drop
- Multi-signal scene boundary classification
- VLM scene analysis (1 call per scene)
- Content origin classification with all 6 categories
- Speaker attribution
- Dialogue index with shot_id
- Recap detection (heuristic + prior-episode comparison)
- Eyecatch detection and exclusion
- scene_detector.py as standalone tunable module

**Gaps (spec has, implementation lacks):**

| Field/Feature | Spec says | Implementation does | Needed by |
|---|---|---|---|
| Background embedding | CLIP 512-dim on YOLOv8-masked background | Color histogram correlation | Scene grouping accuracy, generation retrieval |
| Background plates | Save masked background per shot | Not implemented | Generation location reference |
| Character crops | Save per-character crop per shot | Not implemented | Generation identity reference |
| Shot type per shot | VLM classifies wide/medium/close-up/etc | All shots default to "medium" | Alignment, generation shot policy |
| Content origin | VLM with chapter manifest context | Heuristic (fuzzy match + rules) | Alignment accuracy |
| Speaker attribution | VLM per shot with keyframe | Heuristic (scene character count) | Dialogue index accuracy |
| Audio classifier | Music/ambient/silence detection | Subtitle overlap only | Scene boundary accuracy |
| `reconstruction_prompt` | Director's prompt per scene | Not implemented | Generation retrieval, director policy |
| `arc_id` | On manifest + every scene JSON | Not present | All downstream stages |
| Per-shot keyframes | keyframe_path per shot | Only per scene | Alignment at shot granularity |
| Color palette format | Dominant hex colors | Raw 48-float HSV histogram | Generation color grading |
| Spec terminology | Says "Whisper" in steps 5-6 | Correctly uses subtitles | Spec accuracy |

---

## Section 2 — Implementation Strengths to Preserve

These are things the implementation does better than the spec describes. Each phase that touches related code must preserve these. When updating the spec MDs, add these.

### Stage 1 — Manga Extraction Strengths

1. **Faction-grouped character lists.** Wiki character lists are grouped by faction/location (e.g. "Red Hair Pirates: Shanks, Benn Beckman, Lucky Roux" / "Foosha Village: Makino, Woop Slap"). Groups are passed directly to the VLM prompt, giving it structural context — if it sees pirates on a ship, the faction grouping narrows identification. The spec only says "closed character set" without describing the grouping. (manga_metadata.py:236-246)

2. **Two-tier character alias system.** Three-layer cascade: (a) CORE_ALIASES — hardcoded for recurring characters across all chapters, covers edge cases like "Sogeking" → `usopp`, "Cutty Flam" → `franky`, "Red-Haired Shanks" → `shanks`; (b) wiki-derived aliases built at runtime via `build_character_aliases()` — maps both display name and wiki page title to canonical snake_case ID, so every chapter automatically gets correct aliases for its cast; (c) regex fallback for names in neither dict. The spec just says "character_id: lowercase slug." (manga_metadata.py:62-118, 352-357)

3. **Bubble type / text type validation.** `validate_page_json()` catches a common VLM error: confusing `speech_bubble_type` values (like "shout" or "whisper") with `type` values. Automatically reclassifies to "dialog". The prompt also includes an explicit `IMPORTANT` block explaining the distinction. The spec lists both fields but doesn't acknowledge the VLM confuses them. (manga_metadata.py:139-145, 392-402)

4. **Sliding context window for previous pages.** Each page extraction receives summaries from the previous N pages (default N=2), including locations, time contexts, panel narratives, and key dialogue. Helps the VLM maintain narrative continuity across pages without re-sending all prior page images. The spec doesn't mention inter-page context. (manga_metadata.py CONFIG context_window)

5. **`saga` field.** The implementation stores both `arc` and `saga` from the wiki (e.g. arc="Romance Dawn Arc", saga="East Blue Saga"). The spec only has `arc_id`. Saga is a useful higher-level grouping — multiple arcs belong to one saga. Phase 1 added `arc_id` but `saga` must also be preserved. (manga_metadata.py:835-837)

6. **`wiki_grounded` in processing_metadata.** Tracks whether the extraction was wiki-grounded or not. Distinguishes high-quality extractions (with closed character set, synopsis context) from low-quality ones (generic prompt fallback). The spec doesn't track this. (manga_metadata.py:830)

7. **`--no-wiki` mode.** Explicitly disables wiki fetching for offline environments or non-One-Piece manga. Falls back to a generic prompt with no character grounding — VLM still works, just less accurately. Any phase modifying the manga VLM prompt must produce valid output in both wiki and no-wiki modes. (manga_metadata.py CLI)

8. **Manga pipeline graceful degradation (3 layers).** (a) `manga_wiki_api.py` not importable → `HAS_MANGA_WIKI_API = False`, skipped silently; (b) chapter ID not a valid number → returns None; (c) any wiki API call fails (network, 404, parse error) → catches exception, prints warning, returns None. Each layer fails independently. The spec doesn't discuss failure modes. (manga_metadata.py:14-26, 273-289)

### Stage 2 — Anime Extraction Strengths

9. **Wiki episode context is independent of chapter manifest.** `fetch_wiki_episode_context()` fetches characters with annotations, synopsis, arc, location debuts via `one_piece_wiki_api.py`. Works well even without a chapter manifest. When both are available, the VLM receives a richer combined context. (anime_metadata.py)

10. **Previous scene context passing.** VLM receives prior scene result with careful disclaimer: "this scene may be a DIFFERENT location/time/storyline." Helps narrative continuity without forcing false continuity. Eyecatch scenes reset the continuity chain. (anime_metadata.py:862-878)

11. **Anime VLM prompt engineering.** Episode position ("Scene X/Y"), flashback candidate flags from saturation analysis, combined location vocabulary from manga+wiki, character identification rules prohibiting generic labels ("unknown", "Character 1", "man"), manga dialogue reference with framing ("this scene may or may not correspond to these lines"). (anime_metadata.py)

12. **Eyecatch detection is concrete.** Proper detection with tunable heuristics: mid-episode position window, duration threshold, surrounding fade checks, brightness analysis, fallback path. Detected eyecatch scenes are excluded from dialogue indexing, content origin, and VLM analysis. (scene_detector.py:589-659)

13. **Graceful degradation without chapter manifest.** Wiki context fills the gap for character identification and location vocabulary. Heuristic content origin classifier still functions. Speaker attribution uses scene character lists from VLM analysis. (anime_metadata.py)

14. **Cost tracking.** API token usage tracking per VLM call with detailed cost breakdown at pipeline end. Manga pipeline also tracks per-page tokens aggregated in manifest. (anime_metadata.py:50-82, manga_metadata.py:825-830)

15. **Modular content origin stage.** Runs as separate Stage 6, re-tunable without re-running expensive VLM calls. Uses fuzzy text matching (Jaccard word overlap) against manga dialogue index plus shot property analysis. (anime_metadata.py)

---

## Section 3 — Implementation Phases

### Phase 1 — Spec updates + trivial metadata [STATUS: complete]

**Goal:** Update research MDs with implementation strengths. Add simple metadata fields. Zero risk to existing pipeline behavior.

**Completed:** All 5 deliverables implemented. `normalize_arc_id()` added to `one_piece_wiki_api.py` as shared function. `arc_id` and `locations` fields added to manga chapter manifest output. `arc_id` added to anime episode manifest and per-scene JSONs. `_extract_location_vocabulary()` updated to read from `locations` field. Both spec MDs updated with implementation strengths sections. Whisper references fixed.

**Deliverables:**

1. Fix "Whisper" -> "subtitle" in STAGE2_ANIME_EXTRACTION.md (steps 5 and 6)

2. Update STAGE1_MANGA_EXTRACTION.md to document:
   - Wiki grounding details (chapter notes, synopsis, cover page, grouped character lists)
   - Dynamic alias building from wiki
   - Bubble type / text type validation
   - Context window for previous pages
   - Cost tracking

3. Update STAGE2_ANIME_EXTRACTION.md to document:
   - Wiki episode context fetching (independent of chapter manifest)
   - Previous scene context passing with continuity disclaimer
   - Eyecatch detection heuristics
   - Graceful degradation without chapter manifest
   - VLM prompt engineering details
   - Cost tracking

4. Add `arc_id` standardization:
   - Create an `arc_id` mapping function (wiki arc name string -> snake_case id)
   - Add `arc_id` field to manga chapter_manifest.json output
   - Add `arc_id` field to anime episode_manifest.json and per-scene JSONs
   - Both pipelines use the same mapping

5. Add `locations` list to manga chapter manifest:
   - Aggregate deduplicated location strings from all panels
   - Store as top-level field on chapter_manifest.json
   - Update anime pipeline's _extract_location_vocabulary() to read from this field

**Validation:** Run both pipelines on existing test data. Diff outputs — only new fields should appear, no existing fields changed.

---

### Phase 2 — VLM prompt enhancements [STATUS: complete]

**Goal:** Improve accuracy of existing outputs by enriching the VLM scene analysis call. Add new output fields. Keep heuristic fallbacks.

**Completed:** All 4 deliverables implemented. VLM prompt expanded to return `shot_analysis` (per-shot shot_type + content_origin), `reconstruction_prompt` (director's scene description), and `dialogue_attribution` (per-subtitle speaker + confidence). Raw VLM results cached to `vlm_cache/` for accurate resume. Stage 6 uses VLM content_origin as primary with heuristic fallback, logging disagreements. Stage 7 uses VLM speaker attribution as primary with heuristic fallback. Scene JSONs include `reconstruction_prompt`. `_apply_vlm_scene_result()` and `_apply_vlm_dialogue_attribution()` helper functions added for clean separation. STAGE2_ANIME_EXTRACTION.md updated with implementation details.

**Deliverables:**

1. Per-shot `shot_type` classification:
   - Modify VLM scene analysis prompt to return shot_type per shot_id in the scene
   - Parse VLM response, assign shot_type to each shot
   - Keep "medium" as fallback if VLM doesn't return a type for a shot
   - Update shot_type_sequence in scene JSONs from real data

2. `reconstruction_prompt` per scene:
   - Add to VLM scene analysis prompt: "Write a director's reconstruction prompt..."
   - Store in scene JSON output
   - This is a new field, purely additive

3. Content origin via VLM:
   - Add content origin question to VLM scene analysis prompt for each shot in the scene
   - VLM receives manga dialogue context (already passed) and classifies per shot
   - Keep heuristic classification as fallback when VLM result is missing/unclear
   - Compare VLM vs heuristic results on test episodes before switching default

4. Speaker attribution via VLM:
   - For each scene's VLM call, include the subtitle lines in that scene with timestamps
   - Ask VLM to return speaker_id + speaker_confidence per subtitle line
   - Keep heuristic attribution as fallback
   - Compare VLM vs heuristic results on test episodes before switching default

**Validation:** Run on 2-3 test episodes. Compare VLM-attributed content_origin and speaker_id against heuristic versions. Log disagreements. VLM is primary when available; heuristic is fallback. Disagreements logged on shots as `_content_origin_heuristic` for post-run analysis.

**Note:** All four changes modify the same VLM prompt. Implemented together since they share the same API call — no additional VLM cost. Heuristic fallbacks preserved for all fields.

---

### Phase 3 — Per-shot keyframes [STATUS: complete]

**Goal:** Save a keyframe image for every shot, not just every scene.

**Completed:** All 3 deliverables implemented. `extract_shot_features()` in `scene_detector.py` accepts an optional `keyframes_dir` parameter. When provided (by `anime_metadata.py` Stage 3), the middle frame of each shot is saved as `shot_{shot_id:04d}_keyframe.jpg` and `keyframe_path` is set on the shot dict. This is efficient — the middle frame is already read for color/motion features, so no redundant video reads. The `_set_default_features()` fallback sets `keyframe_path` to None for shots whose frames can't be read. Scene keyframes (Stage 5) remain unchanged — they use `scene_NNN_keyframe.jpg` naming and coexist in the same `keyframes/` directory. `build_shot_index()` already included `keyframe_path` in the shot index output. STAGE2_ANIME_EXTRACTION.md updated with per-shot keyframe details.

**Deliverables:**

1. During feature extraction (Stage 3 of pipeline), save the middle frame of each shot as a JPEG
   - Store in `keyframes/` subdirectory
   - Path: `keyframes/shot_{shot_id:04d}_keyframe.jpg`

2. Add `keyframe_path` to each shot in the shot index

3. Scene keyframes continue to work as before (first wide/aerial shot in scene)

**Validation:** Check keyframe count matches shot count. Check file sizes are reasonable. Verify shot_index.json has keyframe_path populated.

**Note:** This is a prerequisite for Phase 5 (character crops need per-shot keyframes). Also needed for Phase 2's VLM enhancements if we want to send multiple keyframes per scene.

---

### Phase 4 — Panel cropping (manga pipeline) [STATUS: complete]

**Goal:** Extract panel bounding boxes and save cropped panel images.

**Completed:** All 5 deliverables implemented. VLM prompt expanded with `PANEL BOUNDING BOXES` instruction and `bbox` field in the JSON schema. `validate_bounding_box()` validates and clamps fractional coordinates, rejecting degenerate boxes (<1% area). `crop_and_save_panels()` reads the page image with OpenCV, crops each panel by its bbox, and saves as `panels/{panel_id}.png`. `panel_image_path` is set in-place on panel dicts before the per-page JSON is written. `build_manifest()` reads `panel_image_path` from saved page JSONs and includes it in dialogue_index entries. All preservation constraints verified: sliding context window, `--no-wiki` mode, bubble type validation, `saga` field, `wiki_grounded` tracking, and graceful degradation (invalid/missing bbox -> `panel_image_path = null`, no crash). STAGE1_MANGA_EXTRACTION.md updated with panel cropping details.

**Deliverables:**

1. Add bounding box fields to manga VLM prompt:
   - Ask VLM to return approximate bounding box per panel (top-left x,y and bottom-right x,y as fractions 0.0-1.0 of page dimensions)
   - Parse and validate bounding boxes

2. Crop and save panel images:
   - Use bounding boxes to crop from page image
   - Save to `output/manga_metadata/{chapter_id}/panels/`
   - Path: `panels/{panel_id}.png`

3. Store `panel_image_path` in per-page JSON panel entries

4. Add `panel_image_path` to dialogue_index entries in chapter manifest

5. Update STAGE1_MANGA_EXTRACTION.md with panel cropping details

**Validation:** Visual inspection of panel crops on test chapters. Check bounding boxes don't overlap excessively or miss content. Check dialogue_index has panel_image_path populated.

**Note on depth/segmentation:** depth_map_path and segmentation_map_path are deferred to Phase 5 (they require GPU models and are generation-only dependencies). Panel cropping itself is a prerequisite.

**Manga pipeline preservation constraints:** This phase modifies the manga VLM prompt (adding bounding box extraction). The following existing strengths must be preserved:
- Sliding context window (strength #4): previous-page summaries must still be included in the modified prompt
- `--no-wiki` mode (strength #7): bounding box extraction must work with both wiki-grounded and generic fallback prompts
- Bubble type / text type validation (strength #3): `validate_page_json()` must still run on the response — adding bbox fields must not break existing validation
- `saga` field (strength #5): manifest building must continue to include `saga` alongside `arc_id`
- `wiki_grounded` tracking (strength #6): must remain in processing_metadata
- Graceful degradation (strength #8): all 3 fallback layers must still work — panel cropping should degrade gracefully (skip crops if bbox parsing fails, not crash the pipeline)

---

### Phase 5 — GPU model additions [STATUS: complete]

**Goal:** Add CLIP, YOLOv8, and depth/segmentation models as optional pipeline features.

**Prerequisite:** Phase 3 (per-shot keyframes) and Phase 4 (panel crops) must be complete.

**Completed:** All 5 deliverables implemented. Anime pipeline: `load_gpu_models()` in `scene_detector.py` lazy-loads YOLOv8-seg and CLIP ViT-B/32. `extract_shot_features()` accepts optional `gpu_models` dict — when YOLOv8 is loaded, runs `_run_yolo_segmentation()` per keyframe to extract character crops (saved to `keyframes/character_crops/`), background plates (inpainted via Telea, saved as `keyframes/shot_NNNN_bg_plate.jpg`), and character bounding boxes with confidence. When CLIP is loaded, runs `_compute_clip_embedding()` on the background plate (or full frame if no YOLOv8), storing 512-dim L2-normalized vector. `_compute_scene_break_score()` uses CLIP cosine distance as primary scene break signal when embeddings are available, falling back to color histogram correlation. `_extract_dominant_hex_colors()` uses k-means clustering (default 5 colors) on each keyframe, stored as `color_hex` on shots. `build_shot_index()` outputs `character_crops`, `background_embedding`, `background_plate_path`, and `color_palette` (hex colors). CLI flags: `--use-yolo`, `--use-clip`. Manga pipeline: `load_manga_gpu_models()` lazy-loads MiDaS (via torch.hub) and DeepLabV3 (via torchvision). `process_panel_gpu_models()` runs depth/segmentation per panel crop after `crop_and_save_panels()`, saving depth maps as `panels/{panel_id}_depth.png` (uint8) and segmentation masks as `panels/{panel_id}_seg.png` (uint8 class indices). Both paths stored in per-page JSON and propagated to `dialogue_index` in chapter manifest. CLI flags: `--use-depth`, `--use-segmentation`. All GPU features gracefully degrade (skip with null paths on failure, no crash). Resume support for all GPU outputs. STAGE1_MANGA_EXTRACTION.md and STAGE2_ANIME_EXTRACTION.md updated.

**Deliverables:**

1. YOLOv8 character segmentation on anime keyframes:
   - Run YOLOv8-seg on each shot keyframe
   - Extract character bounding boxes -> save character crops
   - Store `character_crops` in shot index (bbox + confidence + crop_image_path)
   - Mask out characters via segmentation masks + inpainting -> save background plate
   - Store `background_plate_path` in shot index

2. CLIP background embeddings:
   - Run CLIP on background plates (characters masked out)
   - Store 512-dim embedding in shot index
   - Add CLIP cosine distance as primary scene grouping signal in scene_detector.py
   - Keep color histogram as secondary/fallback signal
   - Make CLIP optional via --use-clip flag (pipeline works without it)

3. Depth estimation on manga panel crops:
   - Run MiDaS depth estimator on each saved panel crop
   - Save depth map image as `panels/{panel_id}_depth.png`
   - Store `depth_map_path` in per-page JSON and dialogue_index

4. Segmentation on manga panel crops:
   - Run DeepLabV3 segmentation model on each saved panel crop
   - Save segmentation mask as `panels/{panel_id}_seg.png`
   - Store `segmentation_map_path` in per-page JSON and dialogue_index

5. Color palette as hex colors:
   - Extract dominant colors from shot keyframes (k-means on pixels)
   - Store as `color_hex` list on shots, output as `color_palette` in shot index
   - Existing HSV histogram retained internally for scene boundary fallback

**Validation:** Each model is independently toggleable. Pipeline works without any GPU models. Graceful degradation on per-item failures. Resume skips existing outputs.

**Flags:** Anime: `--use-yolo`, `--use-clip`. Manga: `--use-depth`, `--use-segmentation`.

---

### Phase 6 — Audio classifier [STATUS: complete]

**Goal:** Add lightweight audio analysis for music, ambient sound, and silence detection.

**Completed:** All 4 deliverables implemented. `extract_audio_track()` in `scene_detector.py` uses ffmpeg to extract mono WAV audio (cached per episode). `_classify_audio_for_shot()` performs librosa-based spectral analysis per shot: RMS energy for silence detection, harmonic-percussive source separation for music detection, spectral flatness for tonal vs noise classification, and onset rate for rhythmic regularity. Classification returns `audio_classification` dict with `has_music`, `music_confidence`, `is_silence`, `rms_db`, `spectral_flatness`, `onset_rate`, and `audio_type` ("music"/"dialogue"/"ambient"/"silence"). `_compute_scene_break_score()` uses three new signals: music continuity across boundary (anti-break, weight -0.15), music cut at boundary (pro-break, weight +0.20), and silence onset (moderate pro-break, weight +0.10). `build_shot_index()` includes `audio_classification` in shot index when real data is available. `_extract_scene_sfx()` uses audio classification to refine SFX category. CLI flag: `--use-audio`. All existing features preserved — subtitle-based `audio_features` unchanged. Graceful degradation: librosa ImportError caught, ffmpeg failures handled, per-shot classification errors fall back to defaults. STAGE2_ANIME_EXTRACTION.md and CLAUDE.md updated.

**Deliverables:**

1. Extract audio track from video (ffmpeg, done once)

2. Per-shot audio classification:
   - Detect music presence
   - Detect ambient environment type
   - Detect silence
   - Flag audio events at shot boundaries (music cut, silence onset)

3. Update scene boundary scoring in scene_detector.py:
   - Music continuity across boundary = anti-break signal
   - Music cut at boundary = pro-break signal
   - Silence onset = moderate pro-break signal

4. Store audio features alongside existing subtitle-based features

**Validation:** Run scene detection with and without audio features on test episodes. Compare scene counts and boundaries. Audio should reduce false scene breaks where music continues.

---

### Phase 7 — Multi-keyframe support [STATUS: complete]

**Goal:** Extract multiple keyframes per shot (adaptive by duration), send multi-image VLM calls, migrate schema from `keyframe_path` (singular) to `keyframe_paths` (list).

**Completed:** All deliverables implemented. `_compute_keyframe_positions()` in `scene_detector.py` computes adaptive keyframe positions based on shot duration (<3s: 1 frame, 3-6s: 2 frames, >6s: every ~2.5s). YOLOv8 runs on all keyframes with `bbox_area_fraction` and `keyframe_index` per crop. CLIP stays single-frame (primary only). `_select_vlm_keyframes()` in `anime_metadata.py` performs duration-based selection of per-shot primary keyframes for VLM input (no hard cap). `_call_vlm_scene()` sends multiple base64-encoded images. Scene keyframe extraction from video removed — all keyframes come from per-shot Stage 3 extraction. Schema migrated: `keyframe_paths` (list) everywhere, backward compat shim for old cache files. `validate_multi_keyframe.py` created for baseline comparison. Docs updated.

**Deliverables:**

1. Adaptive multi-keyframe extraction in `scene_detector.py`:
   - Duration-based: <3s → 1 frame, 3-6s → 2 frames, >6s → every ~2.5s
   - YOLOv8 on all keyframes (not just primary)
   - `bbox_area_fraction` on character crops
   - CLIP unchanged (primary keyframe only)

2. Multi-image VLM calls in `anime_metadata.py`:
   - Duration-scaled keyframe selection (scenes <15s: all shots, 15-60s: 1/5s, >60s: 1/8s)
   - Remove scene keyframe extraction from video
   - Prompt includes image-to-shot mapping

3. Schema migration (`keyframe_path` → `keyframe_paths`):
   - Clean break in output writers (scene JSONs, shot index)
   - Backward compat shim for old cached `shots_features.json`
   - `validate_rerun.py` updated

4. Validation script (`validate_multi_keyframe.py`):
   - 8-section comparison report: keyframe stats, VLM coverage, shot types, content origin, speaker attribution, characters, cost, spot checks

5. Documentation updates

**Validation:** Run `python validate_multi_keyframe.py 0001` after pipeline rerun on episode 0001. Expect: increased total keyframes, more VLM images per scene, improved shot type variety, improved speaker attribution confidence.

---

## Execution Protocol

1. Start each phase in a new conversation
2. At the start of the conversation, read this plan file to see current status
3. Implement the phase deliverables
4. After implementation, update:
   - This plan file (mark phase STATUS: complete, add any notes)
   - The relevant research MDs (STAGE1_*.md, STAGE2_*.md) to reflect new reality
   - CLAUDE.md if pipeline capabilities changed
5. Validate before marking complete
