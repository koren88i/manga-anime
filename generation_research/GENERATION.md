# Generation System

## Status

Future phase. Not currently in implementation scope. Documented here so that Stages 1–3 are built with the right outputs to support it.

## Goal

Given a new manga chapter that was not in the training set, produce a corresponding anime episode in the visual style of One Piece.

---

## What the Generation Pipeline Receives

At inference time the system has:

- The manga chapter manifest from Stage 1 (panels, dialogue, locations, characters)
- The panel image crops and depth/segmentation maps per panel
- The transformation dataset from Stage 3 (all alignment records with delta prompts)
- A retrieval index over the transformation dataset

It does not have any anime frames for the new chapter. It must produce them.

---

## Core Principle

The system does not attempt to generate video directly from manga pixels. It first learns what decisions to make — duration, shot types, pacing, what to add — and then executes those decisions using video generation infrastructure. The alignment dataset is the supervised training set for the decision-making step.

---

## Pipeline Sketch

```
New manga chapter
        ↓
[1] Manga extraction (Stage 1 pipeline)
        ↓
[2] Director policy model
    — for each panel or panel group, predict production decisions
    — retrieves similar aligned records from transformation dataset
    — produces an imperative delta prompt per panel group
        ↓
[3] Conditioning assembly
    — imperative delta prompt (text)
    — manga panel crops (images)
    — depth maps and segmentation masks (spatial control)
    — character reference crops from retrieval (identity consistency)
    — background plate references from retrieval (location consistency)
    — structured control signals: duration, shot_type_sequence, motion_intensity
    — arc_id (style conditioning)
        ↓
[4] Video generation
    — one shot at a time, conditioned on all signals above
    — assembled into scenes, then episode
        ↓
[5] Audio track (parallel, separate pipeline)
    — TTS dialogue using dialogue_index tone_direction fields
    — SFX from sound_effects fields
    — music selection
        ↓
Final episode
```

---

## The Two Conditioning Types

**Text conditioning** — the imperative delta prompt tells the model what to do. It is the output of the director policy model and is written in the same format as the prompts produced during Stage 3 alignment. Example:

> "You have three manga panels as source. Panel 1: Luffy stands at the ship's prow, wind in his hat. Panel 2: the Going Merry sails toward a storm. Panel 3: Nami watching from the deck, expression serious.
> Expand into 12 shots over 35 seconds. Open wide on the Merry from distance, storm visible on horizon. Cut to Luffy close-up — determined, not worried. Slow pan across the crew on deck. End on a wide shot of the Merry entering the storm front. Tone: adventurous, forward momentum, slightly ominous."

**Visual conditioning** — the model also receives the actual manga panel images, character reference crops from similar past scenes, and background plates from similar locations. This is what makes the output look like One Piece rather than generic anime. Text alone cannot enforce character consistency, art style, or location accuracy.

The two conditioning types are complementary. Text drives intent and structure. Images drive identity and style.

---

## The Director Policy Model

This is the learned core of the generation system. It takes manga panel descriptions and images as input and produces the imperative delta prompt as output.

It is fine-tuned on the transformation dataset from Stage 3 — specifically the pairs of (manga panel descriptions + panel images) → (imperative delta prompt). After seeing thousands of examples of how Toei adapted One Piece panels across 60 episodes, it learns:

- How much to expand each type of manga moment
- What shot type sequences fit each narrative beat
- What the studio typically adds that isn't in the manga
- How tone shifts between manga and anime for different scene types
- Arc-specific conventions (Alabasta has different pacing norms than East Blue)

At inference time, for a panel group it has never seen, it retrieves the most similar alignment records from the transformation dataset and uses them as few-shot context or produces a prompt via learned generation.

---

## Retrieval for Consistency

At inference time, for each panel group, the system retrieves from the alignment dataset:

**Similar past scenes** — using text embedding similarity over panel_narrative and character interaction fields (not raw image CLIP embeddings, which cluster poorly on manga). "Find previous Zoro fights" or "find Going Merry departure scenes" — structured semantic retrieval over manifest fields.

**Character reference crops** — for each character in the current panel, retrieve high-confidence character crops from the anime side of the alignment dataset, filtered by character_id, emotion, and shot_type. These are fed as reference images to enforce visual identity consistency across generated shots.

**Location reference images** — retrieve background plates from scenes with the same location label. Feed as visual conditioning to maintain location consistency.

**Color palettes** — retrieve the canonical color palette for the current location and arc. Use as a color grading conditioning signal.

---

## What Is Not in Scope

**Audio generation**: TTS, SFX, and music are parallel pipelines separate from video generation. They consume the alignment dataset's dialogue_index (tone_direction fields for TTS, sound_effects fields for SFX) but are not part of the video generation pipeline.

**Full episode coherence**: maintaining visual consistency of characters and locations across an entire 18-minute episode is a hard unsolved problem. The initial generation system targets scene-level coherence. Episode-level consistency is a future problem.

**Generalization beyond One Piece**: the system is trained on One Piece data and learns One Piece's art style, characters, and production conventions. Generalizing to other manga/anime is a separate project requiring different training data.

---

## Key Dependencies on Earlier Stages

The generation pipeline requires these outputs from earlier stages to function:

From Stage 1: `panel_image_path` per panel (visual conditioning), `depth_map_path` and `segmentation_map_path` (spatial control), `arc_id` (style conditioning), `dialogue_index` with `tone_direction` (TTS input).

From Stage 2: `character_crops` per shot in shot_index (identity reference library), `background_plate_path` per shot (location reference library), `color_palette` per shot (color grading reference), `reconstruction_prompt` per scene (used as retrieval signal for finding similar past scenes).

From Stage 3: `imperative_delta_prompt` per alignment record (fine-tuning data for director policy model), `transformation_delta` fields (training signal for structured prediction), `manga_context_window` per record (context for director policy inference), `confidence_tier` per record (training sample weighting).
