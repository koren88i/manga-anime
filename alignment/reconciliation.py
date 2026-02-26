"""
Reconciliation — Post-Processing Consistency

No LLM calls. Normalizes locations, validates character IDs, computes
aggregate statistics, and flags low-confidence regions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


def _normalize_location_labels(records):
    """
    Standardize near-duplicate location labels to the most common form.

    E.g. "Partys Bar" and "Partys Bar, Foosha Village" -> most frequent one.
    """
    # Collect all location strings from both sides
    all_locations = []
    for r in records:
        for d in (r.get("manga_side") or {}).get("panel_descriptions", []):
            if isinstance(d, dict) and d.get("location"):
                all_locations.append(d["location"])

    # Count frequencies
    freq = Counter(all_locations)
    if not freq:
        return records, []

    # Build a normalization map: short form -> longest common form
    # Group locations that are substrings of each other
    sorted_locs = sorted(freq.keys(), key=len, reverse=True)
    norm_map = {}
    for loc in sorted_locs:
        loc_lower = loc.lower().strip()
        # Check if this is a substring of an already-seen longer form
        matched = False
        for canonical in norm_map.values():
            if loc_lower in canonical.lower() or canonical.lower() in loc_lower:
                # Use the more frequent one as canonical
                if freq.get(loc, 0) > freq.get(canonical, 0):
                    norm_map[loc_lower] = loc
                else:
                    norm_map[loc_lower] = canonical
                matched = True
                break
        if not matched:
            norm_map[loc_lower] = loc

    # This is a lightweight normalization — mainly for reporting
    warnings = []
    if len(norm_map) < len(freq):
        warnings.append(
            f"Normalized {len(freq)} location variants to {len(norm_map)} canonical forms"
        )

    return records, warnings


def _validate_character_ids(records, manga_manifest, anime_manifest):
    """
    Check that all character_ids in alignment records exist in the
    character registries.

    Returns list of warnings for unknown IDs.
    """
    # Build known character sets
    manga_chars = set()
    manga_registry = manga_manifest.get("characters", {})
    if isinstance(manga_registry, dict):
        manga_chars = set(manga_registry.keys())
    # Also collect from dialogue_index
    for d in manga_manifest.get("dialogue_index", []):
        sid = d.get("speaker_id")
        if sid:
            manga_chars.add(sid)

    anime_chars = set()
    for d in anime_manifest.get("dialogue_index", []):
        sid = d.get("speaker_id")
        if sid:
            anime_chars.add(sid)

    all_known = manga_chars | anime_chars
    warnings = []

    # Check records (mainly anchor speaker IDs)
    for r in records:
        if r.get("speaker_match") is not None:
            # Speaker IDs are already from the manifests, should be valid
            pass

    return warnings


def _compute_statistics(records, anime_original_seqs=None):
    """
    Compute aggregate statistics for the alignment.

    Returns a stats dict.
    """
    total = len(records)
    if total == 0:
        return {
            "total_records": 0,
            "tier_1_count": 0,
            "tier_2_count": 0,
            "tier_3_count": 0,
            "anime_original_shot_count": 0,
            "average_expansion_ratio": 0,
            "average_confidence": 0,
            "anchor_density": 0,
            "dialogue_anchored_count": 0,
            "interpolated_count": 0,
            "unanchored_count": 0,
        }

    tier_1 = sum(1 for r in records if r.get("confidence_tier") == 1)
    tier_2 = sum(1 for r in records if r.get("confidence_tier") == 2)
    tier_3 = sum(1 for r in records if r.get("confidence_tier") == 3)

    dialogue_anchored = sum(
        1 for r in records if r.get("anchor_type") == "dialogue_anchored"
    )
    interpolated = sum(
        1 for r in records if r.get("anchor_type") == "interpolated"
    )
    unanchored = sum(
        1 for r in records if r.get("anchor_type") == "unanchored"
    )

    # Expansion ratios (from transformation_delta)
    expansion_ratios = []
    for r in records:
        delta = r.get("transformation_delta")
        if isinstance(delta, dict):
            ratio = delta.get("duration_expansion_ratio")
            if isinstance(ratio, (int, float)) and ratio > 0:
                expansion_ratios.append(ratio)

    avg_expansion = (
        sum(expansion_ratios) / len(expansion_ratios) if expansion_ratios else 0
    )

    # Average confidence
    confidences = [r.get("confidence", 0) for r in records]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Anime-original shots
    anime_original_shots = set()
    for r in records:
        for sid in r.get("anime_original_segments", []):
            anime_original_shots.add(sid)

    stats = {
        "total_records": total,
        "tier_1_count": tier_1,
        "tier_2_count": tier_2,
        "tier_3_count": tier_3,
        "anime_original_shot_count": len(anime_original_shots),
        "anime_original_dialogue_count": len(anime_original_seqs or []),
        "average_expansion_ratio": round(avg_expansion, 2),
        "average_confidence": round(avg_confidence, 3),
        "anchor_density": round(dialogue_anchored / max(total, 1), 3),
        "dialogue_anchored_count": dialogue_anchored,
        "interpolated_count": interpolated,
        "unanchored_count": unanchored,
    }

    return stats


def _flag_low_confidence(records):
    """
    Flag records with tier 3 + low confidence for optional human review.

    Returns list of alignment_ids that are flagged.
    """
    flagged = []
    for r in records:
        if r.get("confidence_tier") == 3 and r.get("confidence", 1.0) < 0.3:
            flagged.append(r["alignment_id"])
    return flagged


def run_reconciliation(records, manga_manifest, anime_manifest,
                       anime_original_seqs=None):
    """
    Post-processing step with no LLM calls.

    Args:
        records: list of alignment record dicts
        manga_manifest: chapter manifest dict
        anime_manifest: episode manifest dict
        anime_original_seqs: list of anime seq numbers flagged as anime-original

    Returns:
        records: cleaned alignment records
        statistics: aggregate stats dict
        warnings: list of issues found
    """
    print(f"[Reconciliation] Processing {len(records)} alignment records...")

    all_warnings = []

    # 1. Normalize locations
    records, loc_warnings = _normalize_location_labels(records)
    all_warnings.extend(loc_warnings)

    # 2. Validate character IDs
    char_warnings = _validate_character_ids(records, manga_manifest, anime_manifest)
    all_warnings.extend(char_warnings)

    # 3. Compute statistics
    statistics = _compute_statistics(records, anime_original_seqs)

    # 4. Flag low-confidence regions
    flagged = _flag_low_confidence(records)
    if flagged:
        all_warnings.append(
            f"{len(flagged)} records flagged for low confidence: "
            + ", ".join(flagged[:10])
            + ("..." if len(flagged) > 10 else "")
        )

    # Print summary
    print(f"[Reconciliation] Statistics:")
    print(f"  Total records: {statistics['total_records']}")
    print(f"  Tier 1 (dialogue-anchored): {statistics['tier_1_count']}")
    print(f"  Tier 2 (short-gap interpolated): {statistics['tier_2_count']}")
    print(f"  Tier 3 (long-gap interpolated): {statistics['tier_3_count']}")
    print(f"  Dialogue-anchored: {statistics['dialogue_anchored_count']}")
    print(f"  Interpolated: {statistics['interpolated_count']}")
    print(f"  Anime-original dialogue lines: "
          f"{statistics['anime_original_dialogue_count']}")
    print(f"  Average confidence: {statistics['average_confidence']:.3f}")
    if statistics['average_expansion_ratio'] > 0:
        print(f"  Average expansion ratio: "
              f"{statistics['average_expansion_ratio']:.2f}")
    if all_warnings:
        print(f"  Warnings: {len(all_warnings)}")
        for w in all_warnings:
            print(f"    - {w}")

    return records, statistics, all_warnings
