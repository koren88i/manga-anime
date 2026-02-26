# Stage 1 — Manga Extraction

## Status

Complete. Implemented in `manga_metadata.py`.

## Purpose

Extract structured semantic representations from manga pages. The output is the primary input to Stage 3 alignment and provides vocabulary grounding (locations, character IDs, dialogue) to Stage 2 anime extraction.

## Input

A directory of manga chapter page images (colorized English scanlation), one image per page, named sequentially (001.png, 002.png, ...).

## Prerequisites

- **OPENAI_API_KEY** environment variable (for GPT-4o vision calls)
- **Internet access** (optional) — for wiki chapter context fetching via `manga_wiki_api.py`. Pipeline works without it but VLM quality is lower.
- Chapter page images as numbered files (e.g. `001.jpg`, `002.png`) in a directory.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Kumiko panel detection first (recommended, ~2s per chapter)
python extraction/kumiko_panels.py --chapter-id 0001

# Run manga extraction (auto-detects Kumiko data if present)
python extraction/manga_metadata.py ./manga/0001 --chapter-id 0001

# Resume interrupted run (skips already-processed pages)
python extraction/manga_metadata.py ./manga/0001 --chapter-id 0001 --resume

# Without wiki context (generic prompt, no character grounding)
python extraction/manga_metadata.py ./manga/0001 --chapter-id 0001 --no-wiki

# Without Kumiko panel detection (VLM-only bboxes)
python extraction/manga_metadata.py ./manga/0001 --chapter-id 0001 --no-kumiko

# With optional GPU models for panel crops
python extraction/manga_metadata.py ./manga/0001 --chapter-id 0001 \
    --use-depth --use-segmentation

# Custom output directory
python extraction/manga_metadata.py ./manga/0001 --chapter-id 0001 --output-dir ./custom_output

# Compare two runs (e.g. before/after Kumiko integration)
python utils/compare_manga_output.py output/manga_metadata/0001_baseline output/manga_metadata/0001
```

## Startup: Context Gathering

Before any page is processed, `main()` fetches external context from the One Piece Fandom Wiki:

1. **Wiki chapter data** — `fetch_wiki_chapter_context()` calls `manga_wiki_api.py` to get:
   - Closed character list with annotations (e.g. "Gold Roger (flashback)", "Nami (cover)"), grouped by faction/location (e.g. "Red Hair Pirates", "Foosha Village")
   - Chapter title from the infobox (VIZ title preferred)
   - Long narrative summary (~15k chars for Ch1)
   - Chapter notes (key plot points and revelations)
   - Cover page description
   - Arc and saga lookup
   Free, no API key needed.

2. **Dynamic prompt building** — the system prompt and character aliases are built at runtime from wiki data:
   - `build_system_prompt(wiki_context)` generates the VLM prompt with a closed character set ("use ONLY these names") grouped by faction, plus all structural extraction rules
   - `build_character_aliases(wiki_context)` merges core recurring-character aliases with wiki-derived name-to-ID mappings

If `--no-wiki` is passed, or the wiki fetch fails, the pipeline falls back to a generic prompt with no character grounding. The VLM still works but is more likely to use inconsistent or generic character labels.

## Output

All files written to `output/manga_metadata/{chapter_id}/`.

```
output/manga_metadata/{chapter_id}/
  chapter_manifest.json    # Top-level summary + flat dialogue index
  character_registry.json  # All characters with appearance tracking + voice profiles
  page_001.json            # Per-page detailed metadata
  page_002.json
  ...
  panels/                  # Cropped panel images + optional depth/segmentation maps
    001_panel_1.png
    001_panel_1_depth.png  # Only with --use-depth
    001_panel_1_seg.png    # Only with --use-segmentation
    ...
```

---

## Pipeline Stages

| Stage | Description | Module |
|-------|-------------|--------|
| Pre-pass | Kumiko OpenCV panel detection (run separately, auto-detected) | `kumiko_panels.py` |
| 1/4 | Discover page images in chapter directory, sort numerically | `manga_metadata.py` |
| 2/4 | Load progress — find already-processed pages for resume support. Load Kumiko data if available. | `manga_metadata.py` |
| 3/4 | GPT-4o vision extraction per page, grounded with wiki context + Kumiko panel boundaries | `manga_metadata.py` + `manga_wiki_api.py` |
| 4/4 | Build chapter manifest with flat dialogue index | `manga_metadata.py` |

### Stage 1: Discover Pages

Scans the chapter directory for image files (`.png`, `.jpg`, `.jpeg`), extracts the page number from the filename stem (must be numeric, e.g. `001.jpg`), and sorts numerically. Non-numeric filenames are skipped with a warning.

### Stage 2: Load Progress

When `--resume` is passed, loads previously processed `page_*.json` files from the output directory. Already-processed pages are skipped. The character registry and page context summaries are restored so the pipeline can continue where it left off.

### Pre-pass: Kumiko Panel Detection

Run `extraction/kumiko_panels.py --chapter-id NNNN` before manga extraction. This uses the vendored Kumiko library (`vendor/kumiko/`) to detect panel bounding boxes via OpenCV contour analysis. Output is saved to `output/kumiko_panels/{chapter_id}/kumiko_panels.json` with normalized `[x1,y1,x2,y2]` bboxes per page, keyed by filename. Kumiko uses RTL reading order by default (`--no-rtl` to disable). Takes ~2 seconds per chapter.

`manga_metadata.py` auto-detects Kumiko data at startup — no CLI flag needed. If `output/kumiko_panels/{chapter_id}/kumiko_panels.json` exists, it is loaded and used. Use `--no-kumiko` to skip.

### Stage 3: GPT-4o Vision Extraction

The core AI stage. For each unprocessed page:

1. **Encodes the page image** as base64 JPEG/PNG
2. **Looks up Kumiko bboxes** for the current page (if available)
3. **Builds the prompt** with:
   - System prompt (dynamically built from wiki context or generic fallback)
   - Chapter synopsis from wiki (adaptively windowed to relevant paragraphs)
   - Chapter notes from wiki (up to 8 key events)
   - Context from previous pages (sliding window of 2 pages)
   - Known characters discovered so far (from the evolving character registry)
   - **Kumiko panel boundaries** (if available): exact panel count and bbox coordinates injected as "PRE-DETECTED PANEL BOUNDARIES" with an instruction to return exactly that many panels and copy the bboxes verbatim
4. **Calls GPT-4o** with the image (high detail) + text prompt, using JSON response mode
5. **Validates the JSON response** — ensures required fields exist, normalizes character IDs via `CHARACTER_ALIASES`, fixes misclassified text types
6. **Overrides bboxes with Kumiko** (if available) — IOU-based matching (`match_panels_by_iou()`, greedy, threshold 0.2) pairs each VLM panel with the best-overlapping Kumiko bbox. Matched panels get `bbox_source: "kumiko"`, unmatched keep `bbox_source: "vlm"`. This always runs via IOU matching (never blind index mapping) to handle cases where VLM and Kumiko return panels in different order.
7. **Sorts panels into reading order** — `sort_panels_reading_order()` groups panels into rows by vertical overlap (>50% of shorter panel height), sorts right-to-left within each row (manga RTL convention), and updates `reading_order` fields. This guarantees downstream consumers receive panels in correct sequence regardless of VLM output order.
8. **Crops panels** — validates bounding boxes, crops panel images from the page, saves to `panels/`
9. **Runs GPU models** (optional) — depth estimation and segmentation on each panel crop
10. **Updates the character registry** — adds new characters, tracks page appearances, enriches voice profiles from tone_direction and emotion hints
11. **Saves outputs** — per-page JSON + updated character registry

Rate-limited by `delay_between_requests` (default 2.0s). Retries on JSON parse errors, rate limits, and API errors (3 attempts with exponential backoff).

Typical cost for one chapter (~53 pages for Ch1): ~$1.50

### Stage 4: Build Chapter Manifest

Aggregates all `page_*.json` files into `chapter_manifest.json`:
- Builds the flat `dialogue_index` — one entry per dialogue line across the entire chapter, with sequential numbering, speaker attribution, emotion, location, and time context
- Aggregates deduplicated `locations` list from all panels (vocabulary for anime extraction)
- Aggregates statistics (panels, dialogue lines, narration lines, SFX, unique characters)
- Includes chapter title, arc, saga, and arc_id from wiki context (or fallback)
- Records processing metadata (model, date, tokens, wiki grounding status)

---

## Output Schema

### Per-page JSON — `page_NNN.json`

One file per page. Contains:

```
page_number
source_file
page_type: "cover" | "credits" | "title_spread" | "story"
page_narrative: 1–3 sentence summary of what happens on this page
panels: [
  panel_id                        e.g. "3_panel_2"
  reading_order                   integer, left-to-right top-to-bottom
  position_description            e.g. "top-right", "full-page", "middle-left"
  bbox                            [x1, y1, x2, y2] fractional coords (0.0-1.0 of page)
  bbox_source                     "kumiko" (OpenCV pre-detected) or "vlm" (GPT-4o estimated)
  panel_image_path                path to saved cropped panel image (null if bbox invalid)
  scene_description               brief visual description
  panel_narrative                 what happens — action, reaction, story beat
  location                        specific, e.g. "execution platform at Loguetown"
  time_context                    e.g. "present day", "22 years ago", "flashback to Roger's execution"
  depth_map_path                  path to saved depth estimation image for this panel crop
  segmentation_map_path           path to saved segmentation mask for this panel crop
  characters_present: [
    character_id                  lowercase slug, e.g. "luffy", "zoro"
    name
    action
  ]
  text_elements: [
    text_id                       e.g. "3_panel_2_text_1"
    type                          "dialog" | "narration" | "internal_monologue" | "sfx" | "caption"
    content
    speaker
    speaker_id
    speaker_confidence            0.0–1.0
    speech_bubble_type            "speech" | "thought" | "shout" | "whisper" | "narration_box"
    emotion
    tone_direction                TTS voice direction, e.g. "loud and excited", "calm and warm"
    reading_order
  ]
  sound_effects: [
    sfx_id
    content                       English text or transliteration
    description                   what the sound is, e.g. "glass shattering"
    intensity                     1–10
    category                      "impact" | "environment" | "action" | "emotion" | "voice"
    japanese_text_visible         true/false
  ]
]
```

Example:

```json
{
  "page_number": 5,
  "source_file": "005.png",
  "page_type": "story",
  "page_narrative": "Luffy declares he will become King of the Pirates while Shanks laughs at him.",
  "panels": [
    {
      "panel_id": "005_panel_1",
      "reading_order": 1,
      "position_description": "top-right",
      "bbox": [0.5, 0.0, 1.0, 0.5],
      "bbox_source": "kumiko",
      "panel_image_path": "panels/005_panel_1.png",
      "scene_description": "Luffy standing on a barrel at Partys Bar, fist raised",
      "panel_narrative": "Luffy boldly announces his dream to the entire bar",
      "location": "Partys Bar",
      "time_context": "10 years ago",
      "depth_map_path": null,
      "segmentation_map_path": null,
      "characters_present": [
        {
          "character_id": "luffy",
          "name": "Luffy",
          "action": "standing on barrel, fist raised, shouting"
        }
      ],
      "text_elements": [
        {
          "text_id": "005_panel_1_text_1",
          "type": "dialog",
          "content": "I'm gonna be King of the Pirates!",
          "speaker": "Luffy",
          "speaker_id": "luffy",
          "speaker_confidence": 0.95,
          "speech_bubble_type": "shout",
          "emotion": "excited",
          "tone_direction": "loud and determined, youthful voice",
          "reading_order": 1
        }
      ],
      "sound_effects": [
        {
          "sfx_id": "005_panel_1_sfx_1",
          "content": "BANG",
          "description": "fist slamming on barrel",
          "intensity": 6,
          "category": "impact",
          "japanese_text_visible": false
        }
      ]
    }
  ]
}
```

### Field Reference

**Panel-level alignment signals:**

| Field | Type | Alignment role |
|---|---|---|
| `panel_id` | string | Reference key |
| `location` | string | **Strong correlation signal** — matches anime scenes by setting |
| `time_context` | string | **Disambiguation signal** — distinguishes flashback vs present |
| `panel_narrative` | string | Scene matching |

**Text element alignment signals:**

| Field | Type | Alignment role |
|---|---|---|
| `content` | string | **Primary correlation signal** — fuzzy matched against anime subtitles |
| `speaker_id` | string | Speaker matching |
| `emotion` | string | Soft matching signal |
| `tone_direction` | string | TTS pipeline |
| `speech_bubble_type` | enum | Delivery style for TTS |

**Panel image crops**: The VLM returns an approximate `bbox` per panel as fractional coordinates `[x1, y1, x2, y2]` (0.0-1.0 of page dimensions). These bounding boxes are validated (`validate_bounding_box()` — clamped to [0,1], must have at least 1% area in each dimension) and used to crop the panel from the page image via OpenCV. Crops are saved to `output/manga_metadata/{chapter_id}/panels/{panel_id}.png` and the path is stored in `panel_image_path` on both the per-page JSON panel entry and the dialogue_index in the chapter manifest. If bounding box parsing fails for a panel, `panel_image_path` is set to null and the pipeline continues — graceful degradation, no crash. Works in both wiki-grounded and `--no-wiki` modes.

**Depth maps and segmentation**: Optional GPU models run on each saved panel crop when enabled via CLI flags. `--use-depth` loads MiDaS (via torch.hub) and estimates a depth map per panel crop, saved as `panels/{panel_id}_depth.png` (uint8, 0-255, higher = closer). `--use-segmentation` loads DeepLabV3 (via torchvision) and produces a semantic segmentation mask per panel crop, saved as `panels/{panel_id}_seg.png` (uint8 class indices). Both paths are stored in `depth_map_path` and `segmentation_map_path` on the panel dict and propagated to the dialogue_index in the chapter manifest. Graceful degradation: if a model fails on a specific panel, the path is set to null and the pipeline continues. Resume support: skips if output files already exist. Used by the generation pipeline for ControlNet-style spatial conditioning.

---

### Character registry — `character_registry.json`

```
chapter_id
characters:
  {character_id}:
    character_id
    name
    visual_description
    first_appearance_page
    appearances: [page numbers]
    voice_profile:
      suggested_voice_style
      observed_emotions: [...]
```

Example:

```json
{
  "chapter_id": "0001",
  "characters": {
    "luffy": {
      "character_id": "luffy",
      "name": "Luffy",
      "visual_description": "standing on barrel, fist raised, shouting",
      "first_appearance_page": 3,
      "appearances": [3, 4, 5, 7, 8, 10],
      "voice_profile": {
        "suggested_voice_style": "loud and excited, youthful",
        "observed_emotions": ["excited", "angry", "determined", "crying"]
      }
    }
  }
}
```

---

### Chapter manifest — `chapter_manifest.json`

The primary reference document for downstream stages. Provides vocabulary grounding for anime extraction and the dialogue anchor list for alignment.

```
chapter_id
chapter_title
series
arc_id                            e.g. "east_blue", "alabasta", "drum_island", "marineford"
arc                               e.g. "Romance Dawn Arc" (raw wiki string)
saga                              e.g. "East Blue Saga" (raw wiki string)
source_type                       e.g. "colorized_english_scanlation"
total_pages
pages: [
  page_number, source_file, page_type, metadata_file,
  panels, dialog_lines, narration_lines, sfx_count
]
dialogue_index: [
  seq                             global sequence number across chapter
  content
  speaker_id
  speaker
  emotion
  page
  panel_id
  panel_image_path
  location
  time_context
]
character_registry_file
statistics:
  total_panels
  total_dialog_lines
  total_narration_lines
  total_sfx
  unique_characters
locations: [distinct location strings found in chapter]   <- vocabulary for anime extraction
processing_metadata:
  model_used
  processing_date
  total_tokens_used
  processing_time_seconds
  wiki_grounded                   true/false
```

Example:

```json
{
  "chapter_id": "0001",
  "chapter_title": "Romance Dawn",
  "series": "One Piece",
  "arc_id": "romance_dawn",
  "arc": "Romance Dawn Arc",
  "saga": "East Blue Saga",
  "source_type": "colorized_english_scanlation",
  "total_pages": 53,
  "pages": [{"page_number": 1, "page_type": "cover", "panels": 0}],
  "dialogue_index": [
    {
      "seq": 1,
      "content": "Wealth... Fame... Power...",
      "speaker_id": "narrator",
      "speaker": null,
      "emotion": "dramatic",
      "page": 2,
      "panel_id": "002_panel_1",
      "panel_image_path": "panels/002_panel_1.png",
      "location": "Loguetown execution platform",
      "time_context": "22 years ago"
    }
  ],
  "character_registry_file": "character_registry.json",
  "statistics": {
    "total_panels": 210,
    "total_dialog_lines": 145,
    "total_narration_lines": 23,
    "total_sfx": 67,
    "unique_characters": 15
  },
  "locations": ["Partys Bar", "Foosha Village harbor", "Loguetown execution platform"],
  "processing_metadata": {
    "model_used": "gpt-4o",
    "processing_date": "2025-01-15T14:30:00",
    "total_tokens_used": 450000,
    "processing_time_seconds": 320.5,
    "wiki_grounded": true
  }
}
```

**`locations` field**: a deduplicated list of all location strings found across all panels in the chapter. Passed to the anime extraction VLM as a controlled vocabulary so location labels stay consistent and joinable across stages.

**`arc_id` field**: identifies which One Piece story arc this chapter belongs to (snake_case via `normalize_arc_id()`). Must be set consistently — the same arc_id will appear in the episode manifest and alignment records and is used as a conditioning signal in generation.

**`arc` and `saga` fields**: raw wiki strings preserved alongside `arc_id`. `saga` is a higher-level grouping — multiple arcs belong to one saga (e.g. Romance Dawn Arc + Orange Town Arc + ... = East Blue Saga). Only present when wiki context is available.

---

## Implementation Details

### Wiki grounding via `manga_wiki_api.py`

At startup, the pipeline fetches chapter context from the One Piece Fandom Wiki via the MediaWiki API (no API key, no scraping — stdlib only). This provides:

| Data | Source Section | Usage |
|------|---------------|-------|
| Character list (grouped by faction) | Quick Reference > Characters | Closed set in VLM prompt: "use ONLY these names" |
| Long summary | Long Summary | Injected as chapter synopsis context |
| Chapter notes | Quick Reference > Chapter Notes | Key events for VLM grounding |
| Chapter title | Infobox (VIZ title) | Manifest + prompt header |
| Arc/saga | Story Arcs page lookup | Manifest metadata + arc_id |
| Cover page description | Cover Page | Informational |

#### Character list format

Chapter wiki pages use a `CharTable` wikitext table that groups characters by faction:

```
Red Hair Pirates: Shanks, Benn Beckman, Lucky Roux, Yasopp, ...
Foosha Village: Makino, Woop Slap, Gyoru, Chicken
Others: Gold Roger (flashback), Monkey D. Luffy, ...
```

Annotations like `(flashback)`, `(cover)`, `(mentioned only)`, `(wanted poster)` are parsed and preserved.

#### Graceful degradation

The wiki fetch has three layers of fallback:
1. If `manga_wiki_api.py` is not importable: `HAS_MANGA_WIKI_API = False`, skipped silently
2. If chapter ID is not a valid number: returns None
3. If any wiki API call fails (network, 404, parse error): catches exception, prints warning, returns None

Without wiki context, the VLM prompt falls back to generic character identification instructions ("Identify characters by their visual appearance and any in-panel name mentions").

#### `--no-wiki` flag

Explicitly disables wiki fetching. Useful for:
- Offline environments
- Non-One Piece manga (the wiki API is One Piece specific)
- Testing the generic prompt behavior

### Character ID normalization

Character names from VLM output are normalized to stable snake_case IDs via a three-tier alias system:

1. **Core aliases** (`CORE_ALIASES`) — hardcoded mappings for recurring characters across all chapters. Covers the Straw Hats, Roger, Shanks, and common variations (e.g. "Monkey D. Luffy" -> `luffy`, "Sogeking" -> `usopp`, "Cutty Flam" -> `franky`, "Red-Haired Shanks" -> `shanks`).

2. **Wiki-derived aliases** (`build_character_aliases()`) — at runtime, for each character in the wiki's chapter character list, adds `display_name -> snake_case(wiki_page)` and `wiki_page -> snake_case(wiki_page)` mappings. This covers chapter-specific characters without manual alias entries.

3. **Fallback** — names not in either alias dict are converted to snake_case via regex: `re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")`.

### Bubble type / text type validation

`validate_page_json()` catches a common VLM error: confusing `speech_bubble_type` values (like "shout" or "whisper") with `type` values (which must be one of "dialog", "narration", "internal_monologue", "sfx", "caption"). When a bubble type is found in the `type` field, it is automatically reclassified to "dialog". This prevents downstream TTS pipeline errors. The VLM prompt also includes an explicit `IMPORTANT` block explaining the distinction.

### Context window for previous pages

Each page extraction receives a sliding window of summaries from the previous N pages (default N=2). Summaries include locations, time contexts, panel narratives, and key dialogue. This helps the VLM maintain narrative continuity across pages without re-sending all prior page images.

### Panel bounding boxes: Kumiko + VLM

Panel bounding boxes come from two sources, with Kumiko taking priority when available:

**Kumiko (OpenCV, preferred)**: `extraction/kumiko_panels.py` runs contour-based panel detection on the raw page images and produces precise normalized bboxes. Kumiko is run as a separate pre-pass and its output is auto-detected by `manga_metadata.py`. Kumiko panels are returned in correct RTL reading order via a spatial sort in the Kumiko library (`Panel.__lt__()` implements top-to-bottom, right-to-left comparison).

**VLM (GPT-4o, fallback)**: The VLM prompt includes a `PANEL BOUNDING BOXES` instruction asking it to return a `bbox` field per panel as `[x1, y1, x2, y2]` fractional coordinates. VLM bboxes tend to snap to a coarse grid and are less precise than Kumiko's.

**Integration flow**:
1. When Kumiko data exists for a page, the exact panel count and bboxes are injected into the VLM prompt as "PRE-DETECTED PANEL BOUNDARIES" so GPT-4o knows the panel layout
2. After the VLM returns its response, `validate_page_json()` overrides VLM bboxes with Kumiko bboxes using IOU-based matching (`match_panels_by_iou()`, greedy, threshold 0.2). This pairs each VLM panel with the spatially closest Kumiko bbox regardless of array order
3. Each panel is tagged with `bbox_source: "kumiko"` or `"vlm"`
4. Panels are then sorted into guaranteed manga reading order by `sort_panels_reading_order()` (row grouping + RTL sort within rows)

**Why IOU matching, not index mapping**: The VLM does not guarantee panels are returned in reading order (the `reading_order` field is metadata, not a guarantee of array order). Kumiko returns panels in RTL reading order. Blind `panel[i] = kumiko[i]` would silently assign wrong bboxes if the orders differ. IOU matching is always safe.

**Validation**: `validate_bounding_box()` clamps coordinates to [0,1] range and rejects boxes smaller than 1% of the page in either dimension. Invalid or missing bboxes result in `panel_image_path = null` — the pipeline never crashes due to bad bounding boxes.

**Output directory**: `output/manga_metadata/{chapter_id}/panels/`

**Naming**: `{panel_id}.png` (e.g. `003_panel_1.png`, `003_panel_2.png`)

**Disable**: `--no-kumiko` skips Kumiko loading entirely, reverting to VLM-only bboxes.

### Cost tracking

API token usage (input and output) is tracked per page and aggregated in the manifest's `processing_metadata.total_tokens_used`. The pipeline prints a cost summary at completion.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `gpt-4o` | OpenAI model for vision extraction |
| `max_tokens` | 4096 | Max output tokens per page |
| `temperature` | 0.1 | Low temperature for consistent extraction |
| `context_window` | 2 | Number of previous pages included as context |
| `retry_attempts` | 3 | Retries on API errors |
| `retry_delay_base` | 2.0 | Exponential backoff base (seconds) |
| `delay_between_requests` | 2.0 | Delay between page API calls (seconds) |
| `--no-kumiko` | false | Disable Kumiko panel detection even if data exists |

---

## One Piece Specific Handling

- **Reading order**: Right-to-left, top-to-bottom (traditional manga convention). The VLM prompt explicitly instructs this.
- **Wiki grounding**: Chapter character list, synopsis, and notes fetched from One Piece Fandom Wiki via `manga_wiki_api.py` at pipeline startup. Character list is passed as a closed set ("use ONLY these names") grouped by faction.
- **Character IDs**: Same snake_case convention as anime pipeline (`luffy`, `zoro`, `nami`). `CORE_ALIASES` covers the main recurring cast.
- **Colorized scanlations**: The prompt identifies the input as colorized English fan translations, which affects visual analysis (colors present, text is English).
- **Speech bubble types**: VLM is instructed to distinguish bubble types (speech, thought, shout, whisper, narration_box) for TTS prosody. These are separate from text element types (dialog, narration, etc.).

---

## Alignment Anchors

The output schema is designed for a third alignment stage to correlate manga and anime:

- **Primary**: `manga.dialogue_index[n].content <-> anime.dialogue_index[n].content` matched by text similarity within a bounded chapter/episode window
- **Secondary**: Panel narrative/scene descriptions, location strings, character presence, time context
- **Page granularity**: `dialogue_index` entries include `page` and `panel_id` so alignment can operate at panel level against anime shots
- **Content origin**: Anime shots classified as `manga_derived` are the alignment targets; `anime_original` shots have no manga counterpart

---

## What downstream stages need from this output

**Stage 2 (anime extraction)** uses the chapter manifest's `locations` list and `character_registry` for vocabulary grounding when the VLM classifies content origin and labels locations in anime shots.

**Stage 3 (alignment)** uses the `dialogue_index` as the primary anchor list for Pass 1 matching. Uses `panel_image_path` fields to construct colorization reference pairs. Uses panel descriptions and character_ids for Pass 2 gap filling.

**Generation** uses `panel_image_path` as visual conditioning input, `depth_map_path` and `segmentation_map_path` for spatial conditioning, and the `dialogue_index` for TTS input.

---

## Deliverables

- `extraction/manga_metadata.py` — main pipeline orchestrator, runs all stages in order
- `extraction/kumiko_panels.py` — OpenCV panel detection pre-pass (Kumiko)
- `wiki/manga_wiki_api.py` — wiki API client for chapter metadata
- `utils/compare_manga_output.py` — structural comparison tool (panels, bboxes, IOU, Kumiko source stats)
- `requirements.txt`
