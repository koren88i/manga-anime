"""
Stage 3 — Alignment Pipeline

Aligns manga chapter metadata with anime episode metadata to produce a
transformation dataset. The output captures how Toei Animation adapted
each manga panel into anime — which shots it became, how long it lasted,
what was added or skipped, and the creative decisions behind each choice.

Two-pass architecture:
  Pass 1: Dialogue anchoring (semantic embedding matching)
  Pass 2: Bounded gap filling (GPT-4o)
  Post:   Reconciliation (no LLM)

Usage:
    # Explicit paths (for testing)
    python alignment.py --episode-id 0001 \\
        --episode-manifest output/anime_metadata/0001_baseline/episode_manifest.json \\
        --chapter-manifest output/manga_metadata/0001/chapter_manifest.json \\
        --shot-index output/anime_metadata/0001_baseline/shot_index.json

    # Auto-discover from episode ID
    python alignment.py --episode-id 0001

    # Pass 1 only (no LLM cost)
    python alignment.py --episode-id 0001 --pass1-only \\
        --chapter-manifest output/manga_metadata/0001/chapter_manifest.json

Output:
    output/alignment/{episode_id}/
        alignment_manifest.json
        alignment_index.json
        pass1_anchors.json
        pass1_gaps.json
        llm_cache/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import logging
import time

# ---------------------------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------------------------

# We import carefully to handle missing dependencies gracefully
try:
    from wiki.one_piece_wiki_api import get_episode_chapters, get_chapter_episodes
    HAS_WIKI_API = True
except ImportError:
    HAS_WIKI_API = False

from alignment.pass1_dialogue_anchor import run_pass1, get_embedding_cost_summary
from alignment.pass2_gap_fill import run_pass2, get_llm_cost_summary
from alignment.reconciliation import run_reconciliation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "output_base": "output/alignment",

    # OpenAI API
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "openai_model": "gpt-4o",
    "embedding_model": "text-embedding-3-small",
    "openai_max_tokens": 4096,
    "openai_temperature": 0.2,

    # Rate limiting
    "vlm_max_retries": 3,
    "vlm_retry_delay": 5,
    "vlm_calls_per_minute": 100,

    # Pass 1 matching
    "match_threshold": 0.65,
    "speaker_match_bonus": 0.10,
    "location_match_bonus": 0.05,
    "time_context_match_bonus": 0.05,

    # Pass 2 context
    "context_window_size": 3,

    # Resume
    "resume": False,
}


# ---------------------------------------------------------------------------
# JSON I/O (same pattern as anime_metadata.py)
# ---------------------------------------------------------------------------

def load_json(path):
    """Load a JSON file, return None if not found."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(data, path):
    """Save data to a JSON file with pretty printing."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def _discover_anime_dir(episode_id):
    """Find the anime metadata directory for an episode."""
    base = os.path.join("output", "anime_metadata", episode_id)
    # Try standard dir first, then {episode_id}_baseline
    for suffix in ["", "_baseline"]:
        d = f"{base}{suffix}"
        if os.path.exists(os.path.join(d, "episode_manifest.json")):
            return d
    return None


def _discover_chapter_manifests(episode_id):
    """
    Use wiki API to find chapter manifests for an episode.

    Returns list of (chapter_id_str, manifest_path) tuples.
    """
    if not HAS_WIKI_API:
        logger.warning("Wiki API not available -- cannot auto-discover chapters")
        return []

    ep_num = int(episode_id)
    result = get_episode_chapters(ep_num)

    if result is None:
        logger.warning(f"Episode {ep_num} not in wiki mapping")
        return []

    if result.get("is_filler"):
        return []  # Filler episode, no chapters

    manifests = []
    for ch_num in result.get("chapters", []):
        ch_id = f"{ch_num:04d}"
        ch_path = os.path.join(
            "output", "manga_metadata", ch_id, "chapter_manifest.json"
        )
        if os.path.exists(ch_path):
            manifests.append((ch_id, ch_path))
        else:
            logger.warning(f"Chapter manifest not found: {ch_path}")

    return manifests


def _load_scene_data(anime_dir):
    """Load all scene JSON files from an anime metadata directory."""
    scenes = []
    if not anime_dir:
        return scenes

    for fname in sorted(os.listdir(anime_dir)):
        if fname.startswith("scene_") and fname.endswith(".json"):
            path = os.path.join(anime_dir, fname)
            data = load_json(path)
            if data:
                # Extract scene number from filename
                try:
                    num = int(fname.replace("scene_", "").replace(".json", ""))
                    data["scene_number"] = num
                except ValueError:
                    pass
                scenes.append(data)

    # Also check vlm_cache directory
    vlm_cache = os.path.join(anime_dir, "vlm_cache")
    if os.path.isdir(vlm_cache):
        for fname in sorted(os.listdir(vlm_cache)):
            if fname.startswith("scene_") and fname.endswith(".json"):
                path = os.path.join(vlm_cache, fname)
                data = load_json(path)
                if data:
                    try:
                        num = int(fname.replace("scene_", "").replace(".json", ""))
                        data["scene_number"] = num
                    except ValueError:
                        pass
                    # Only add if not already loaded from main dir
                    existing_nums = {s.get("scene_number") for s in scenes}
                    if data.get("scene_number") not in existing_nums:
                        scenes.append(data)

    return scenes


# ---------------------------------------------------------------------------
# Filler episode output
# ---------------------------------------------------------------------------

def _write_filler_output(episode_id, output_dir, anime_manifest):
    """Write alignment output for a filler episode (100% anime-original)."""
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "episode_id": episode_id,
        "is_filler": True,
        "records": [],
    }
    save_json(manifest, os.path.join(output_dir, "alignment_manifest.json"))

    index = {
        "episode_id": episode_id,
        "is_filler": True,
        "100%_anime_original": True,
        "statistics": {
            "total_records": 0,
            "anime_original_shot_count": anime_manifest.get("total_shots", 0),
        },
    }
    save_json(index, os.path.join(output_dir, "alignment_index.json"))

    print(f"[Alignment] Filler episode -- 100% anime-original. "
          f"No alignment records.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(episode_id, episode_manifest_path=None,
                 chapter_manifest_paths=None, shot_index_path=None,
                 output_dir=None, pass1_only=False, config_overrides=None):
    """
    Run the full alignment pipeline.

    Args:
        episode_id: zero-padded episode ID string (e.g. "0001")
        episode_manifest_path: explicit path to episode manifest
        chapter_manifest_paths: list of explicit paths to chapter manifests
        shot_index_path: explicit path to shot index
        output_dir: output directory (default: output/alignment/{episode_id})
        pass1_only: if True, skip Pass 2 (no LLM calls)
        config_overrides: dict of config overrides
    """
    cfg = {**CONFIG, **(config_overrides or {})}
    start_time = time.time()

    if output_dir is None:
        output_dir = os.path.join(cfg["output_base"], episode_id)

    print(f"=" * 60)
    print(f"Stage 3 -- Alignment Pipeline")
    print(f"Episode: {episode_id}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    # -----------------------------------------------------------------------
    # Step 0: Validate episode-to-chapter mapping
    # -----------------------------------------------------------------------
    print(f"\n[0] Validating episode-to-chapter mapping...")

    wiki_chapters = None
    is_filler = False
    if HAS_WIKI_API:
        ep_num = int(episode_id)
        wiki_result = get_episode_chapters(ep_num)
        if wiki_result is None:
            print(f"  WARNING: Episode {ep_num} not in wiki mapping.")
        elif wiki_result.get("is_filler"):
            is_filler = True
            print(f"  Episode {ep_num} is filler (no manga source).")
        else:
            wiki_chapters = wiki_result.get("chapters", [])
            print(f"  Wiki mapping: episode {ep_num} adapts chapter(s) "
                  f"{wiki_chapters}")
    else:
        print(f"  WARNING: Wiki API not available -- cannot validate mapping.")

    # If explicit --chapter-manifest given, warn if it doesn't match wiki
    if chapter_manifest_paths and wiki_chapters:
        print(f"  Explicit chapter manifests provided -- will cross-check "
              f"against wiki mapping.")

    # -----------------------------------------------------------------------
    # Step 1: Load inputs
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Loading inputs...")

    # Episode manifest
    anime_dir = None
    if episode_manifest_path:
        anime_manifest = load_json(episode_manifest_path)
        anime_dir = os.path.dirname(episode_manifest_path)
    else:
        anime_dir = _discover_anime_dir(episode_id)
        if anime_dir:
            episode_manifest_path = os.path.join(anime_dir, "episode_manifest.json")
            anime_manifest = load_json(episode_manifest_path)
        else:
            anime_manifest = None

    if not anime_manifest:
        print(f"ERROR: Could not load episode manifest for {episode_id}")
        print(f"  Tried: {episode_manifest_path or 'auto-discovery failed'}")
        return False

    print(f"  Episode manifest: {episode_manifest_path}")
    print(f"  Scenes: {anime_manifest.get('total_scenes', '?')}, "
          f"Shots: {anime_manifest.get('total_shots', '?')}, "
          f"Dialogue lines: {len(anime_manifest.get('dialogue_index', []))}")

    # Filler episode -- write empty output and exit
    if is_filler:
        _write_filler_output(episode_id, output_dir, anime_manifest)
        return True

    # Shot index
    if shot_index_path:
        shot_index = load_json(shot_index_path)
    elif anime_dir:
        shot_index_path = os.path.join(anime_dir, "shot_index.json")
        shot_index = load_json(shot_index_path)
    else:
        shot_index = None

    if not shot_index:
        print(f"  WARNING: No shot index found. Pass 2 will have limited context.")
        shot_index = []
    else:
        print(f"  Shot index: {len(shot_index)} shots")

    # Scene data
    scene_data = _load_scene_data(anime_dir)
    print(f"  Scene data: {len(scene_data)} scenes loaded")

    # Chapter manifest(s)
    manga_manifests = []

    if chapter_manifest_paths:
        # Explicit paths provided -- load them
        for path in chapter_manifest_paths:
            manifest = load_json(path)
            if manifest:
                manifest["_manifest_path"] = path
                manga_manifests.append(manifest)
                ch_id = manifest.get("chapter_id", "?")
                dlg_count = len(manifest.get("dialogue_index", []))
                print(f"  Chapter manifest: {path} "
                      f"(chapter {ch_id}, {dlg_count} dialogue lines)")

        # Cross-check against wiki mapping
        if wiki_chapters and manga_manifests:
            loaded_ch_ids = set()
            for m in manga_manifests:
                try:
                    loaded_ch_ids.add(int(m.get("chapter_id", 0)))
                except (ValueError, TypeError):
                    pass
            expected = set(wiki_chapters)
            if loaded_ch_ids != expected:
                print(f"  WARNING: Wiki says episode {ep_num} adapts chapters "
                      f"{sorted(expected)}, but you provided chapters "
                      f"{sorted(loaded_ch_ids)}.")
                print(f"  Matches may be poor if chapters don't correspond "
                      f"to this episode.")
    else:
        # Auto-discover via wiki mapping
        if wiki_chapters is None:
            print(f"  ERROR: No wiki mapping and no --chapter-manifest given.")
            print(f"  Provide explicit chapter manifest paths with "
                  f"--chapter-manifest.")
            return False

        # Check which chapter manifests exist on disk
        found = []
        missing = []
        for ch_num in wiki_chapters:
            ch_id = f"{ch_num:04d}"
            ch_path = os.path.join(
                "output", "manga_metadata", ch_id, "chapter_manifest.json"
            )
            if os.path.exists(ch_path):
                found.append((ch_id, ch_path))
            else:
                missing.append(ch_num)

        if missing:
            print(f"  ERROR: Episode {ep_num} adapts chapters {wiki_chapters}, "
                  f"but manga metadata is missing for chapter(s): {missing}")
            print(f"  Run manga extraction first:")
            for ch in missing:
                print(f"    python manga_metadata.py --chapter-id {ch:04d}")
            return False

        for ch_id, ch_path in found:
            manifest = load_json(ch_path)
            if manifest:
                manifest["_manifest_path"] = ch_path
                manga_manifests.append(manifest)
                dlg_count = len(manifest.get("dialogue_index", []))
                print(f"  Chapter {ch_id}: {dlg_count} dialogue lines")

    if not manga_manifests:
        print(f"ERROR: No chapter manifests loaded.")
        return False

    # -----------------------------------------------------------------------
    # Step 2: Prepare dialogue indices
    # -----------------------------------------------------------------------
    print(f"\n[2/5] Preparing dialogue indices...")

    # Concatenate manga dialogue from all chapters (in order)
    manga_dialogue = []
    for manifest in manga_manifests:
        ch_id = manifest.get("chapter_id", "?")
        for d in manifest.get("dialogue_index", []):
            entry = dict(d)
            entry["_chapter_id"] = ch_id
            manga_dialogue.append(entry)

    anime_dialogue = anime_manifest.get("dialogue_index", [])

    print(f"  Manga dialogue: {len(manga_dialogue)} lines "
          f"from {len(manga_manifests)} chapter(s)")
    print(f"  Anime dialogue: {len(anime_dialogue)} lines")

    if not manga_dialogue:
        print(f"ERROR: No manga dialogue lines found.")
        return False

    # -----------------------------------------------------------------------
    # Step 3: Pass 1 — Dialogue Anchoring
    # -----------------------------------------------------------------------
    print(f"\n[3/5] Pass 1 -- Dialogue Anchoring")

    # Check for cached Pass 1 results
    anchors_path = os.path.join(output_dir, "pass1_anchors.json")
    gaps_path = os.path.join(output_dir, "pass1_gaps.json")

    if cfg.get("resume") and os.path.exists(anchors_path) and os.path.exists(gaps_path):
        print(f"  Loading cached Pass 1 results...")
        anchors = load_json(anchors_path)
        gaps = load_json(gaps_path)
        anime_original_seqs = []  # Not cached separately
        comparison = {}
        print(f"  {len(anchors)} anchors, {len(gaps)} gaps (cached)")
    else:
        anchors, gaps, anime_original_seqs, comparison = run_pass1(
            manga_dialogue, anime_dialogue, config=cfg
        )

        # run_pass1 returns (None, None, None, None) on fatal error
        if anchors is None:
            return False

        # Save intermediate results
        os.makedirs(output_dir, exist_ok=True)
        save_json(anchors, anchors_path)
        save_json(gaps, gaps_path)
        print(f"  Saved: {anchors_path}")
        print(f"  Saved: {gaps_path}")

    # -----------------------------------------------------------------------
    # Step 4: Pass 2 — Gap Filling
    # -----------------------------------------------------------------------
    if pass1_only:
        print(f"\n[4/5] Pass 2 -- SKIPPED (--pass1-only)")
        # Build records from anchors only
        from pass2_gap_fill import _build_anchor_record
        records = []
        for idx, anchor in enumerate(anchors):
            record = _build_anchor_record(
                anchor, manga_manifests[0], anime_manifest,
                shot_index, idx, cfg
            )
            records.append(record)
    else:
        print(f"\n[4/5] Pass 2 -- Gap Filling")
        records = run_pass2(
            anchors, gaps, manga_manifests[0], anime_manifest,
            shot_index, scene_data, config=cfg, output_dir=output_dir
        )

    # -----------------------------------------------------------------------
    # Step 5: Reconciliation + Output
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Reconciliation + Output")

    records, statistics, warnings = run_reconciliation(
        records, manga_manifests[0], anime_manifest,
        anime_original_seqs=anime_original_seqs if 'anime_original_seqs' in dir() else []
    )

    # Build alignment_manifest.json
    alignment_manifest = {
        "episode_id": episode_id,
        "chapter_ids": [m.get("chapter_id") for m in manga_manifests],
        "arc_id": manga_manifests[0].get("arc_id", ""),
        "records": records,
    }
    manifest_path = os.path.join(output_dir, "alignment_manifest.json")
    save_json(alignment_manifest, manifest_path)
    print(f"  Saved: {manifest_path}")

    # Build alignment_index.json
    alignment_index = {
        "episode_id": episode_id,
        "chapter_ids": [m.get("chapter_id") for m in manga_manifests],
        "arc_id": manga_manifests[0].get("arc_id", ""),
        "records": [
            {
                "alignment_id": r["alignment_id"],
                "arc_id": r.get("arc_id", ""),
                "chapter_id": r.get("manga_side", {}).get("chapter_id", ""),
                "episode_id": r.get("anime_side", {}).get("episode_id", ""),
                "panel_ids": r.get("manga_side", {}).get("panel_ids", []),
                "shot_ids": r.get("anime_side", {}).get("shot_ids", []),
                "time_range": r.get("anime_side", {}).get("time_range", []),
                "anchor_type": r.get("anchor_type"),
                "confidence_tier": r.get("confidence_tier"),
                "score_openai": r.get("score_openai"),
                "score_bow": r.get("score_bow"),
                "duration_expansion_ratio": (
                    r.get("transformation_delta", {}) or {}
                ).get("duration_expansion_ratio"),
                "emotional_tone_shift": (
                    r.get("transformation_delta", {}) or {}
                ).get("emotional_tone_shift"),
            }
            for r in records
        ],
        "statistics": statistics,
        "embedding_comparison": comparison if 'comparison' in dir() else {},
        "warnings": warnings,
    }
    index_path = os.path.join(output_dir, "alignment_index.json")
    save_json(alignment_index, index_path)
    print(f"  Saved: {index_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Alignment complete in {elapsed:.1f}s")
    print(f"  Records: {len(records)}")
    print(f"  Tier 1: {statistics.get('tier_1_count', 0)}")
    print(f"  Tier 2: {statistics.get('tier_2_count', 0)}")
    print(f"  Tier 3: {statistics.get('tier_3_count', 0)}")
    if not pass1_only:
        print(f"  {get_llm_cost_summary()}")
    print(f"  {get_embedding_cost_summary()}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 -- Alignment: manga-to-anime transformation dataset"
    )
    parser.add_argument(
        "--episode-id", required=True,
        help="Episode ID (zero-padded, e.g. 0001)"
    )
    parser.add_argument(
        "--episode-manifest",
        help="Path to episode_manifest.json (auto-discovered if not given)"
    )
    parser.add_argument(
        "--chapter-manifest", action="append", dest="chapter_manifests",
        help="Path to chapter_manifest.json (can specify multiple; "
             "auto-discovered via wiki API if not given)"
    )
    parser.add_argument(
        "--shot-index",
        help="Path to shot_index.json (auto-discovered if not given)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: output/alignment/{episode_id})"
    )
    parser.add_argument(
        "--pass1-only", action="store_true",
        help="Run only Pass 1 (dialogue anchoring, no LLM calls)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from cached results"
    )
    parser.add_argument(
        "--match-threshold", type=float,
        help="Minimum similarity for dialogue anchor (default: 0.65)"
    )
    parser.add_argument(
        "--openai-model", default="gpt-4o",
        help="OpenAI model for Pass 2 gap filling (default: gpt-4o)"
    )
    parser.add_argument(
        "--embedding-model", default="text-embedding-3-small",
        help="OpenAI embedding model for Pass 1 (default: text-embedding-3-small)"
    )
    parser.add_argument(
        "--bow-only", action="store_true",
        help="Skip OpenAI embeddings, use bag-of-words matching only "
             "(no API key needed)"
    )

    args = parser.parse_args()

    # Build config overrides from CLI args
    config_overrides = {}
    if args.resume:
        config_overrides["resume"] = True
    if args.match_threshold is not None:
        config_overrides["match_threshold"] = args.match_threshold
    if args.openai_model:
        config_overrides["openai_model"] = args.openai_model
    if args.embedding_model:
        config_overrides["embedding_model"] = args.embedding_model
    if args.bow_only:
        config_overrides["bow_only"] = True

    success = run_pipeline(
        episode_id=args.episode_id,
        episode_manifest_path=args.episode_manifest,
        chapter_manifest_paths=args.chapter_manifests,
        shot_index_path=args.shot_index,
        output_dir=args.output_dir,
        pass1_only=args.pass1_only,
        config_overrides=config_overrides,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
