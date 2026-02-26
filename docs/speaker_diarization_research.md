# Speaker Diarization Research

## Best GitHub Projects

| Project | Stars | Why it matters |
|---------|-------|----------------|
| whisper-diarization | ~2.8k | Closest to what we need — Demucs + Whisper + NeMo TitaNet embeddings, end-to-end |
| animespeechdataset | — | Anime-specific — builds per-character embedding database, uses KNN matching on labeled data |
| WhisperX | ~19.5k | Fast transcription + diarization, but diarization quality complaints are common |

## APIs for Ground Truth (all have free tiers)

| Service | Free Credits | DER | Best for |
|---------|-------------|-----|----------|
| AssemblyAI | $50 free (~185+ hrs) | ~2.9% speaker count error | Best bang-for-buck, Japanese supported |
| pyannoteAI Precision-2 | EUR 19/mo (125 hrs) | Best-in-class (~28% better than OSS) | Direct comparison to your pyannote pipeline |
| Deepgram | $200 free | Good, 10x faster | Secondary reference |

3 One Piece episodes (3.5 hrs total) would cost **$0.60 on AssemblyAI** — fully covered by free credits.

## Recommended Strategy

### Phase 1 — Ground truth (free)
Run AssemblyAI + pyannoteAI on 3 episodes. Where both agree = high-confidence labels.

### Phase 2 — Open-source pipeline
Switch from our current pyannote 3.3.2 (with its dependency hell) to either:
- **whisper-diarization** project (already does Demucs + NeMo TitaNet + clustering)
- **pyannote Community-1** (newer model, pyannote-audio 4.0, much better accuracy than 3.1)

### Phase 3 — Scale
Compare open-source output against API ground truth using DER metric. Build character embedding database that grows per episode. Run all episodes for free.
