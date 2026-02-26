# Manga → Anime Generation System

## Goal

Given a new One Piece manga chapter, generate a corresponding anime episode. The system learns from paired manga/anime data how Toei Animation makes production decisions — what they expand, compress, skip, add, and how they reinterpret each manga moment into cinematic language. That learned director policy is the core of the generation system.

## Data

- 60 anime episodes, 18 minutes each, ~24fps → ~1.5M frames total
- 103 manga chapters → ~9,000 panels total
- Dataset characteristic: ~170 anime frames per manga panel on average (~7 seconds per panel), though individual panels range from 2 seconds to 40+ seconds depending on the moment
- Mapping is not 1-to-1: one episode can draw from several chapters, and one chapter can span multiple episodes
- The chapter/episode mapping is known at coarse level — panel/shot alignment does not yet exist
- Manga is colorized English scanlation

## Pipeline

```
Stage 1 — Manga Extraction        ← COMPLETE
Stage 2 — Anime Extraction        ← COMPLETE
Stage 3 — Alignment               ← COMPLETE
              ↓
     Transformation Dataset
              ↓
     Generation System            ← FUTURE
```

See each stage file for full specification:
- `STAGE1_MANGA_EXTRACTION.md` — what the existing pipeline produces
- `STAGE2_ANIME_EXTRACTION.md` — anime extraction spec
- `STAGE3_ALIGNMENT.md` — alignment architecture and output schema
- `GENERATION.md` — generation pipeline sketch

## Key Design Decisions

These decisions are made and should not be re-debated during implementation.

**Semantic alignment, not pixel alignment.** Align manga to anime through structured natural language representations of both sides, not through visual feature matching across the style gap.

**Dialogue is the ground truth anchor.** Spoken dialogue maps almost 1-to-1 between manga and anime. It is cheap to extract, nearly unambiguous, and provides fixed alignment points that everything else is interpolated from. Use sentence embeddings for matching, not string equality — scanlation and dub/sub translations differ.

**Shots and scenes are different things.** A scene is a semantic unit (same location, continuous dramatic action) containing many shots (single continuous camera takes). The pipeline must separate shot detection from scene grouping. Never treat every camera cut as a scene change.

**The shot index is a first-class output.** The alignment stage needs shot-level interior structure of scenes to understand what the studio did. The shot index is not a debug artifact.

**Content origin must be classified at extraction time.** The VLM during anime extraction has the chapter manifest as context and is in the best position to classify whether each shot is manga-derived or anime-original. Do not leave this inference to the alignment stage.

**Character IDs must be consistent across all stages.** Always use the same lowercase slug — "luffy", "zoro", "nami" — so manifests can be joined directly without translation.

**Location labels must use controlled vocabulary.** Pass the chapter manifest's location strings to the VLM at anime extraction time to keep labels consistent and joinable.

**Retrieval is semantic, not visual, for manga panels.** CLIP embeddings cluster poorly on manga due to low style variance and high semantic variance. Retrieval over manga panels uses text embeddings over structured fields — character_ids, location, panel_narrative — not image embeddings.

**The generation pipeline conditions on images and structure, not just text.** The imperative delta prompt is one conditioning signal. The manga panel crops, character reference crops, background plates, and structured control signals (duration, shot sequence, motion intensity) are equally important inputs. The text prompt alone will not produce consistent One Piece visuals.

**Arc identity is a conditioning signal.** One Piece visual style, pacing, and tone vary significantly across arcs. Every manifest and alignment record carries an `arc_id` field that the generation model uses as a conditioning input.

**No speaker diarization.** Dialogue content comes from episode subtitle files, not audio transcription. Speaker attribution is handled by passing subtitle lines and the shot keyframe to the VLM along with the character list. The VLM returns speaker_id and speaker_confidence per line as part of its structured output, self-reporting its own certainty. Speaker diarization adds pipeline complexity with marginal gain over this approach and is not worth the engineering cost.
