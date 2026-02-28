# Manga → Anime Generation System — Research & Implementation Plan

## Goal

Given a new One Piece manga chapter, generate a corresponding anime episode — with correct voices, music, motion, and visuals that resemble the characters and their fighting styles. We are not targeting pixel-perfect character consistency. Good vibes, recognizable characters, correct emotional tone, and coherent action are the bar for v1.

---

## Overall Pipeline

```
Manga chapter (new, unseen)
        ↓
[1] Manga understanding
    — parse panels into structured representation (already done by Stage 1)
    — enrich with wiki/fan context for pacing and tone inference
        ↓
[2] Scene planning
    — infer scene boundaries, pacing, and duration from content
    — casual moment vs tense drama vs fight → drives how long each scene runs
    — no alignment data needed, this is narrative common sense from the manga itself
        ↓
[3] Anchor generation (per scene)
    — synthesize the starting frame for each new scene
    — see detail below
        ↓
[4] Video generation (per scene)
    — first shot conditioned on synthesized anchor
    — each subsequent shot conditioned on last frame of previous shot
    — this handles within-scene character consistency
        ↓
[5] Audio (parallel)
    — dialogue via voice cloning from episode audio library
    — SFX and music separately
        ↓
Final episode
```

---

## Design Notes

Pacing and duration are inferred from manga content and wiki/fan context — not learned from alignment data. A fight climax runs longer than a casual conversation. This is narrative common sense.

---

## Anime Extraction (what we mine from existing episodes)

We go through episodes to extract reusable assets and knowledge for generation. What we extract:

- **Character reference crops** — for each main character, high-quality crops per emotion (neutral, angry, determined, scared, comedic, etc.) and per shot type (close-up, medium). Used to generate per-scene character images in the anchor step. Stored as a simple lookup table: `character_id + emotion → reference image(s)`. Not a retrieval database — a curated, manually-reviewed set.
- **Background plates** — masked backgrounds (characters removed) per distinct location. Used to generate location images in the anchor step.
- **Voice reference clips** — clean audio per character for voice cloning. Multiple emotional registers per character.

---

## External Knowledge

**Character knowledge base** — fighting styles, named attacks, weapons, movement characteristics per character. Scraped from One Piece wiki data. Fed into video generation prompts to guide motion.

---

## The Synthesized Anchor — Key Design Decision

At the start of each new scene, there is no previous frame to continue from. Rather than retrieving a similar past frame, we synthesize a purpose-built starting frame calibrated to the specific scene.

**Steps:**
1. Parse the manga page for the scene — identify location, characters present, their emotions, and rough spatial composition from panel layout.
2. Generate the background plate for this location — using the stored background plate reference for this location and color palette, generate or adapt a background image.
3. For each character in the scene — take their canonical reference image for the relevant emotion, generate a new image of them with the correct pose and expression for this scene.
4. Composite characters onto the background — place them according to the spatial composition suggested by the manga panel.
5. Use the composite as the first frame of the scene — all subsequent shots continue from the last frame of the previous shot.

**Why this works:** The anchor is deliberately constructed rather than retrieved, so it's always calibrated to the scene's emotional and spatial context. The multi-character problem is hardest at this step but the manga panel gives us composition guidance and character references give us identity anchors. After the first frame, last-frame continuity carries the characters through the rest of the scene.

**The weak link:** If the anchor composite looks off, every shot in the scene inherits that. Quality checking the anchor before committing to scene generation is important.

---

## Character Consistency Model

We are not targeting 90% consistency. The approach is:

- **Within scenes** — last-frame continuity. Each shot starts from the last frame of the previous shot. Character appearance stays stable because it's literally continuing from the previous image.
- **Across scene boundaries** — synthesized anchor as described above. One purpose-built starting frame per scene.

---

## Implementation Plan

### Phase 0: Data Foundation
*No ML yet. Pure extraction and curation. Everything downstream depends on this quality.*

**Step 0.1 — Episode asset extraction pipeline**
Build a pipeline that processes raw One Piece episode video files and outputs: character crops (by character + emotion + shot type), background plates (characters masked out, keyed by location ID), and voice reference clips (clean audio, per character, per emotional register).

- **Deliverable:** A structured asset library on disk. Manually reviewable.
- **Test:** Human review — sample 20 crops per main character, check quality and correct labeling. Background plates should look clean without ghosting.

**Step 0.2 — Character knowledge base**
Scrape the One Piece Wiki for all main/recurring characters: fighting styles, named attacks, signature movement patterns, weapons. Structure it as a simple JSON lookup (`character_id → description, attacks[], movement_style`, etc.).

- **Deliverable:** `character_kb.json`, manually audited.
- **Test:** Spot-check 10 characters against known wiki entries. Check that named attacks are spelled correctly — these go into generation prompts verbatim.

**Step 0.3 — Location index**
Map episode locations to background plate IDs. Build a lookup: `location_name → background_plate_file(s)`. This is partly manual curation.

- **Deliverable:** `location_index.json` + background plate folder.
- **Test:** Given a location name ("Marineford", "Thousand Sunny deck"), confirm you can pull an appropriate plate.

---

### Phase 1: Manga Understanding (Stage 1)
*Stage 1 parsing is already done — but it needs a defined output schema for downstream stages to consume.*

**Step 1.1 — Canonicalize Stage 1 output schema**
Define and lock a structured JSON schema for a parsed chapter: list of panels, each with `{ characters[], dialogue[], emotion_tags[], location_hint, action_description, composition_hint }`. Everything downstream reads this schema.

- **Deliverable:** Schema doc + 2–3 hand-annotated chapters as gold examples.
- **Test:** Run Stage 1 parser on a held-out chapter, diff output against hand annotation. Define acceptable error rate per field.

**Step 1.2 — Wiki/fan context enrichment**
Given a chapter, fetch relevant wiki context (arc name, character relationships at this point in the story, named techniques expected to appear). Append to the structured panel representation.

- **Deliverable:** Enriched chapter JSON.
- **Test:** Human checks — does the enrichment correctly identify the arc, the key characters, and flag relevant named attacks?

---

### Phase 2: Scene Planning
*Pure reasoning/inference — no generation yet. Cheap to iterate.*

**Step 2.1 — Scene boundary detection**
From the panel sequence, group panels into scenes using location continuity, character presence, and tone. Output: list of scenes, each containing their panel subset.

- **Deliverable:** Scene list with panel assignments.
- **Test:** Human review of 3 chapters. Do scene boundaries feel correct? Check edge cases — flashbacks, intercutting between two locations.

**Step 2.2 — Pacing and duration inference**
For each scene, assign an estimated duration (in seconds). Rules: fight climax > tense dialogue > casual moment. Use panel density, action word density, and dialogue volume as signals. Rule-based first, LLM-assisted later.

- **Deliverable:** Scene list with `duration_seconds`, `pacing_tag` (combat/dialogue/transition/comedic).
- **Test:** Sum durations for a chapter → should land in the 20–24 minute range for a standard episode. Spot-check individual scenes against intuition.

---

### Phase 3: Anchor Generation
*This is the hardest and most critical phase. Build and validate this before touching video.*

**Step 3.1 — Single-character anchor generation**
For a single character with a known emotion, generate a new image using their reference crops. Test image-conditioned generation (IP-Adapter, ControlNet reference, or similar). Do not composite yet.

- **Deliverable:** Given `(character_id, emotion, pose_hint)` → output image.
- **Test:** Human evaluation — does the character look recognizably like themselves? Rate 1–5 across 50 generations for 5 main characters. Set a quality threshold before moving on.

**Step 3.2 — Background generation**
Given a location ID and background plate, generate/adapt a background image for the scene (time of day, weather, damage state if applicable).

- **Deliverable:** Given `(location_id, scene_conditions)` → background image.
- **Test:** Human review — does it look like the right place? Does the style match the show's aesthetic?

**Step 3.3 — Multi-character compositing**
Place generated character images onto generated background using the manga panel's composition hint (who's left/right, foreground/background, size relative to frame).

- **Deliverable:** Fully composited anchor frame per scene.
- **Test:** Human review of 20 anchors across varied scenes. Key failure modes to check: character scale feels wrong, characters overlap badly, composition reads as unnatural. Flag any anchor rated below threshold — these block scene generation.

**Step 3.4 — Anchor quality gate**
Build a lightweight quality-check step (human-in-the-loop for v1, auto-scored later) before any anchor is passed to video generation. Rejected anchors get regenerated with adjusted parameters.

- **Deliverable:** Anchor review UI or script. Pass/fail log.
- **Test:** Run 30 anchors through the gate, measure how often a human reviewer agrees with any auto-score you add.

---

### Phase 4: Video Generation
*Only start this after Phase 3 is stable. Bad anchors make this untestable.*

**Step 4.1 — Single-shot video from anchor**
Given an anchor frame + scene description + character movement style (from KB), generate a single 3–6 second video clip. Use a video generation model that accepts image conditioning (first frame).

- **Deliverable:** Single video clip per test scene.
- **Test:** Human review — does action look plausible? Does the character maintain recognizable appearance through the clip? Check for obvious deformation or drift.

**Step 4.2 — Multi-shot scene generation with last-frame continuity**
Chain shots within a scene. Each shot uses the last frame of the previous shot as its first-frame condition. Vary camera angle and framing per shot using scene description.

- **Deliverable:** Full generated scene (multiple chained shots, 15–60 seconds depending on pacing).
- **Test:** Watch the full scene. Does character appearance stay stable? Do scene transitions feel coherent? Rate overall scene quality.

**Step 4.3 — Cross-scene stitching**
Assemble all scenes in order. Validate that scene boundaries (anchor → first shot) are the only consistency weak points, and that within-scene continuity holds.

- **Deliverable:** Full episode rough cut (video only, no audio).
- **Test:** Watch 2–3 full episodes end to end. Log all consistency breaks and classify as anchor-origin vs. mid-scene drift.

---

### Phase 5: Audio
*Largely parallel to Phase 4. Can be developed independently.*

**Step 5.1 — Voice cloning pipeline**
Given character dialogue (from Stage 1) and voice reference clips, generate voiced lines per character. Use a voice cloning model (e.g. XTTS, Tortoise, or commercial API).

- **Deliverable:** Audio files per dialogue line.
- **Test:** Blind listening test — can a human correctly identify which character is speaking for 20 samples? Target >85% accuracy for main cast.

**Step 5.2 — SFX and music placement**
Rule-based or LLM-assisted: map `pacing_tag` and action events to SFX and music cues. For v1, a relatively simple lookup (fight → combat OST, emotional moment → piano theme, etc.) using the existing One Piece OST library.

- **Deliverable:** Timed SFX + music track.
- **Test:** Mix against the silent video. Does the music change land at the right moments? Do SFX feel roughly synchronized?

**Step 5.3 — Audio mixing and sync**
Align dialogue, SFX, and music against the video timeline. Basic sync based on scene/shot timing from Phase 2 pacing estimates.

- **Deliverable:** Full episode with audio.
- **Test:** Watch with audio. Is dialogue roughly lip-sync plausible? Are emotional music cues landing at the right beats?

---

### Phase 6: End-to-End Integration
*Wire all stages together. Find the failure modes that only appear at system scale.*

**Step 6.1 — Full pipeline run on a held-out chapter**
Run a complete, unseen chapter through every stage end-to-end with no human intervention except at the anchor quality gate.

- **Deliverable:** Generated episode.
- **Test:** Human eval panel scores the episode on: character recognizability, emotional coherence, action clarity, overall "anime vibe." Also measure wall-clock generation time.

**Step 6.2 — Failure mode catalog**
Collect and categorize all quality failures from Step 6.1. Prioritize by frequency and severity. This drives what gets iterated in v2.

- **Deliverable:** Ranked failure mode doc.
- **Test:** Is this doc specific enough to drive engineering decisions? Each failure mode should have a root cause hypothesis.

---

## Key Dependencies and Risks

**Anchor quality gate (Step 3.4)** is the highest-leverage point in the whole system — if you get this right, everything downstream benefits. If you skip validating it, you'll burn compute on Phase 4 with garbage inputs.

**Multi-character compositing (Step 3.3)** is the hardest single technical problem. Budget extra time here. The manga panel composition hint helps, but getting scale, lighting consistency, and natural placement right is genuinely difficult. Worth prototyping this in parallel with Phase 1/2 rather than waiting for the full pipeline to be ready.

**Voice cloning quality** will be highly dependent on the cleanliness of your reference clip extraction (Step 0.1). Invest early in good audio isolation — background music bleed is the most common failure mode in voice cloning pipelines.
