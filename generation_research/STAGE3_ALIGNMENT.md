# Stage 3 — Alignment

## Status

Implemented. Pipeline files: `alignment.py`, `pass1_dialogue_anchor.py`, `pass2_gap_fill.py`, `reconciliation.py`.

## Purpose

Produce a transformation record for every manga panel or panel group showing exactly what Toei Animation did with it — which anime shots it became, how long it lasted, what shot types were used, what was added or skipped, and what emotional shift occurred. This transformation dataset is the supervised training signal for the generation system. It captures the director policy: the creative decisions a studio makes when adapting manga to anime.

## Input

- Chapter manifest(s) from Stage 1, including `dialogue_index`, panel descriptions, `panel_image_path` fields, and `locations`
- Episode manifest(s) from Stage 2, including `dialogue_index` and `shot_index`
- The coarse chapter/episode mapping (which chapters correspond to which episodes)

## Output

All files written to `output/alignment/{episode_id}/`.

---

## Two-Pass Architecture

The alignment problem is not solved in one sweep. Trying to match, classify, and compute deltas simultaneously produces compounding errors — a misattributed action sequence pulls surrounding dialogue matches off by several panels in both directions. The two passes solve fundamentally different problems with different tools and different confidence levels. Pass 1's output hard-constrains Pass 2.

### Pass 1 — Anchor Establishment

**What it does**: dialogue matching only. Automatic, cheap, high confidence. Its output is a sparse set of fixed pins in both timelines.

**How it works**: take the manga `chapter_manifest.dialogue_index` and the anime `episode_manifest.dialogue_index` for a given chapter/episode pair. Match them using semantic similarity (sentence embeddings, not string equality — scanlation and dub/sub translations differ). Apply the matching within the bounded window defined by the known chapter/episode mapping.

Each confirmed match becomes an anchor: a panel_id and anime shot_id that are known to correspond, with high confidence. A typical chapter/episode pair will anchor 30–60% of manga panels via dialogue. The remaining gaps — especially action-heavy sequences — are left unmatched and passed to Pass 2.

**Gap analysis**: after anchoring, for each gap between consecutive anchors, record exactly how many manga panels are unaccounted for and how many anime shots fall in the corresponding time window. This bounds the Pass 2 sub-problems precisely.

**Anime-original flagging**: identify anime content that falls outside any manga-bounded window — before the first anchor, after the last anchor, or in a gap too large to plausibly correspond to the available manga panels. Flag these as `anime_original: true` before Pass 2 begins. Pass 2 does not try to match these.

---

### Pass 2 — Bounded Gap Filling and Delta Extraction

**What it does**: fills unanchored gaps and computes the transformation delta for every aligned pair — both anchored and interpolated. This is where the generation training signal is produced.

**How it works**: for each gap between consecutive anchors, the LLM receives a fully constrained sub-problem:
- The prior anchor (manga panel + anime shot, confirmed)
- The next anchor (manga panel + anime shot, confirmed)
- The manga structured descriptions for unanchored panels in the gap
- The anime shot data for the corresponding time window, including `content_origin` classifications from Stage 2
- The chapter manifest `dialogue_index` and narrative context

Within this bounded window, the LLM produces:
- A best-effort panel-to-shot mapping with confidence score
- A content origin confirmation or correction for each anime shot in the gap
- The transformation delta for each aligned pair (see schema below)
- The imperative delta prompt for each aligned pair (see below)

**Transformation delta**: for each aligned manga panel (or panel group) to anime shot (or shot group), record:
- `duration_expansion_ratio` — anime seconds divided by manga panel count
- `shot_type_sequence` — ordered list of shot types the studio chose
- `emotional_tone_shift` — preserved / heightened / softened / changed
- `added_content` — list of anime-original shots inserted and their content_origin categories
- `skipped_panels` — manga panel_ids with no anime counterpart in this window

**Imperative delta prompt**: for each aligned pair, the LLM produces a shooting script grounded in the manga source material. This is the primary generation conditioning text. It is written in the imperative voice — instructions for what to do — not the analytical voice describing what happened.

The LLM should reason in two steps internally. First, analytical: describe what the studio did and why — "the studio stretched two silent panels into 18 shots over 40 seconds, inserting Garp and Sengoku reaction shots to build historic weight." This reasoning is not stored as an output field but is essential scaffolding — it forces the LLM to account for the actual evidence before writing instructions. Second, imperative: rewrite that reasoning as a shooting script starting from the manga panels. The analytical step ensures the imperative is grounded in the real alignment rather than fabricated.

The prompt references the actual manga panels as source material and describes the studio's adaptation decisions as actions to execute:

> "You have two manga panels as source. Panel 1: Roger stands on the scaffold, smiling, no dialogue. Panel 2: the crowd below, silent, thousands of faces.
> Expand into 18 shots over 40 seconds. Open with a wide establishing shot of the scaffold platform. Cut to a slow push-in on Roger's face — hold for 4 seconds before he speaks. Insert a reaction shot of Garp in the crowd (not in manga, add him). Insert a reaction shot of Sengoku beside him (not in manga, add him). Return to Roger close-up. Let silence hold for 8 seconds before the first dialogue line begins. Tone: historic, mythic, weightless."

This prompt is directly actionable as a conditioning input to a video generation model. A model fine-tuned on thousands of these across 60 episodes learns to produce them for new manga panels it has never seen.

---

### Post-Processing — Consistency Reconciliation

After both passes, run a lightweight reconciliation step with no LLM calls:

- Normalize location labels across the full alignment manifest — if the same location appears with slight label variations, standardize to the most common form
- Validate all character_ids against the shared character registries from both stages
- Compute aggregate statistics per alignment record — anchor density, average confidence, percentage of anime shots marked anime-original
- Flag low-confidence regions (long action gaps with tier 3 confidence) for optional human review

---

## Confidence Tiers

Every alignment record receives a confidence tier. The generation system uses this to weight training examples — a certain dialogue-anchored alignment and a loosely matched 40-shot fight sequence should not contribute equally to training.

- **Tier 1** — dialogue-anchored. Treat as ground truth.
- **Tier 2** — interpolated in short gaps (1–3 manga panels → 1–5 anime shots) with strong narrative coherence from surrounding anchors. Medium confidence.
- **Tier 3** — interpolated in long gaps (action sequences with many shots and few panels, weak narrative constraint). Low confidence. Useful for learning expansion ratios and shot type patterns but not fine-grained panel-to-shot correspondence.

---

## Output Schema

### Alignment manifest — `alignment_manifest.json`

One record per manga panel or panel group:

```
alignment_id
arc_id
manga_side:
  chapter_id
  panel_ids: [...]
  page_range: [start_page, end_page]
  panel_image_paths: [...]
  dialogue_line_ids: [...]
  panel_descriptions: [...]
  manga_context_window:
    before: [3 panels before this group — panel_id, panel_image_path, panel_narrative, characters_present]
    after:  [3 panels after this group — same fields]
anime_side:
  episode_id
  scene_numbers: [...]
  shot_ids: [...]
  time_range: [start_seconds, end_seconds]
  dialogue_line_ids: [...]
anchor_type: "dialogue_anchored" | "interpolated" | "unanchored"
confidence: 0.0–1.0
confidence_tier: 1 | 2 | 3
transformation_delta:
  duration_expansion_ratio        anime seconds / manga panel count
  shot_count
  shot_type_sequence: [...]
  emotional_tone_shift: "preserved" | "heightened" | "softened" | "changed"
  added_content: [
    content_origin
    shot_id
    description
  ]
  skipped_panels: [panel_ids]
anime_original_segments: [shot_ids with no manga source in this window]
imperative_delta_prompt           the shooting script — see above
alignment_explanation             brief LLM rationale for the matching decision
```

**`manga_context_window`**: the 3 manga panels before and after the aligned panel group, with their descriptions and image paths. The generation system does not generate shots from single panels in isolation — it needs surrounding context to make coherent pacing and transition decisions.

**`arc_id`**: present on every record. Used as a conditioning signal by the generation system — visual style, pacing norms, and tonal conventions vary significantly across One Piece arcs.

---

### Alignment index — `alignment_index.json`

A flat summary of all records for fast lookup without loading individual records:

```
records: [
  alignment_id
  arc_id
  chapter_id, episode_id
  panel_ids, shot_ids
  time_range
  anchor_type
  confidence_tier
  duration_expansion_ratio
  emotional_tone_shift
]
statistics:
  total_records
  tier_1_count, tier_2_count, tier_3_count
  anime_original_shot_count
  average_expansion_ratio
  average_confidence
```

---

## What This Dataset Enables for Generation

The full alignment dataset across 60 episodes and 103 chapters gives the generation system:

**Learned expansion ratios**: how many seconds of anime does each type of manga moment typically expand into — dialogue panels, action panels, reaction panels, establishing panels. These ratios are arc-dependent and character-dependent.

**Learned shot type policies**: given a manga panel of type X (single character reaction, two-character dialogue, wide action, environmental), what shot type sequence does the studio typically choose?

**Learned addition patterns**: what does the studio add around manga-derived shots — reaction shots, establishing shots, extended action? Where in the sequence do they insert them?

**Learned tone shifts**: which types of manga moments get heightened, softened, or reinterpreted emotionally?

**Fine-tuning data for the director LLM**: the imperative delta prompts, paired with their source manga panel descriptions and images, form a supervised dataset for fine-tuning a model that produces shooting scripts from manga panels.

---

## Quick Start

```bash
# Pass 1 only (no LLM cost, tests dialogue anchoring)
python alignment.py --episode-id 0001 --pass1-only \
    --episode-manifest output/anime_metadata/0001_baseline/episode_manifest.json \
    --chapter-manifest output/manga_metadata/0001/chapter_manifest.json \
    --shot-index output/anime_metadata/0001_baseline/shot_index.json

# Full pipeline (Pass 1 + Pass 2 gap filling + reconciliation)
python alignment.py --episode-id 0001 \
    --episode-manifest output/anime_metadata/0001_baseline/episode_manifest.json \
    --chapter-manifest output/manga_metadata/0001/chapter_manifest.json \
    --shot-index output/anime_metadata/0001_baseline/shot_index.json

# Auto-discover from episode ID (uses wiki API for chapter mapping)
python alignment.py --episode-id 0001

# Resume interrupted run
python alignment.py --episode-id 0001 --resume
```

---

## Implementation Details

### Dual Embedding for Dialogue Matching (Pass 1)

Both methods run at runtime. Each anchor stores both scores for comparison.

**OpenAI embeddings** (`text-embedding-3-small`): Batch-embeds all dialogue texts in one API call. 1536-dim vectors, cosine similarity. ~$0.00007 per chapter/episode pair. Primary matching method.

**Bag-of-words cosine** (numpy, zero external deps): Term-frequency vectors, L2-normalized, cosine similarity via dot product. Fallback when OpenAI is unavailable. Known weakness: short generic lines ("Alright!", "Me?") can false-match because they share common words.

The pipeline prints a comparison summary showing agreement rate, disagreements, and score correlation between the two methods.

### Chapter-Episode Mapping

Uses `get_episode_chapters()` and `get_chapter_episodes()` from `one_piece_wiki_api.py` (hardcoded mapping for episodes 1-1121).

- **Auto-discovery**: given `--episode-id`, finds which chapters to load
- **Filler detection**: `is_filler: true` → outputs 100% anime-original manifest, exits early
- **Explicit paths**: `--chapter-manifest` bypasses wiki lookup

### Transformation Analysis (new field)

Each alignment record stores two complementary descriptions:

- **`transformation_analysis`**: What the studio did and why (analytical, past tense). Covers creative decisions — what was expanded, compressed, added, reinterpreted.
- **`imperative_delta_prompt`**: Shooting script from manga panels (imperative voice). Direct conditioning input for generation.

Both are produced by the same LLM call in Pass 2. The analytical step ensures the imperative prompt is grounded in actual evidence.

### Resume Support

- Pass 1 results cached to `pass1_anchors.json` and `pass1_gaps.json`
- Pass 2 LLM responses cached to `llm_cache/gap_NNN.json`
- `--resume` flag skips completed work

---

## Deliverables

- `alignment.py` — main alignment pipeline orchestrator
- `pass1_dialogue_anchor.py` — dialogue matching as a standalone module (dual embeddings)
- `pass2_gap_fill.py` — bounded LLM gap filling and delta extraction as a standalone module
- `reconciliation.py` — post-processing consistency step
