"""Compare baseline vs new anime metadata output after pipeline rerun.

Usage:
    python validate_rerun.py 0001
    python validate_rerun.py 0001 --baseline-suffix _baseline

Compares output/anime_metadata/{episode_id}_baseline/ vs output/anime_metadata/{episode_id}/
Prints a human-readable report of structural checks, quantitative metrics, and spot-checks.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenes(directory):
    """Load all scene_NNN.json files from a directory."""
    scenes = {}
    for f in sorted(os.listdir(directory)):
        if f.startswith("scene_") and f.endswith(".json") and f != "scene_detector.py":
            num = int(f.split("_")[1].split(".")[0])
            scenes[num] = load_json(os.path.join(directory, f))
    return scenes


def fmt_pct(a, b):
    if b == 0:
        return "N/A"
    return f"{a/b*100:.1f}%"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_metric(label, old_val, new_val, unit=""):
    delta = ""
    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
        diff = new_val - old_val
        if old_val > 0:
            pct = diff / old_val * 100
            delta = f"  ({'+' if diff >= 0 else ''}{diff}{unit}, {'+' if pct >= 0 else ''}{pct:.1f}%)"
        else:
            delta = f"  ({'+' if diff >= 0 else ''}{diff}{unit})"
    print(f"  {label:.<40} old: {old_val}{unit}  | new: {new_val}{unit}{delta}")


def run_structural_checks(new_dir, new_scenes, new_manifest):
    """Pass/fail checks on the new output."""
    print_header("STRUCTURAL CHECKS")
    passed = 0
    failed = 0

    # 1. Scene files present and valid JSON
    scene_count = len(new_scenes)
    if scene_count > 0:
        print(f"  [PASS] {scene_count} scene JSON files present and valid")
        passed += 1
    else:
        print(f"  [FAIL] No scene JSON files found")
        failed += 1

    # 2. Episode manifest exists and has arc_id
    if new_manifest:
        print(f"  [PASS] episode_manifest.json present")
        passed += 1
        if new_manifest.get("arc_id"):
            print(f"  [PASS] arc_id present: {new_manifest['arc_id']}")
            passed += 1
        else:
            print(f"  [WARN] arc_id missing or empty (ok if no wiki/chapter context)")
    else:
        print(f"  [FAIL] episode_manifest.json missing")
        failed += 1

    # 3. New fields populated: reconstruction_prompt
    rp_count = sum(1 for s in new_scenes.values() if s.get("reconstruction_prompt"))
    rp_pct = fmt_pct(rp_count, scene_count)
    if rp_count > scene_count * 0.5:
        print(f"  [PASS] reconstruction_prompt populated on {rp_count}/{scene_count} scenes ({rp_pct})")
        passed += 1
    else:
        print(f"  [WARN] reconstruction_prompt only on {rp_count}/{scene_count} scenes ({rp_pct})")

    # 4. Shot type variety (not all "medium")
    shot_types = Counter()
    for s in new_scenes.values():
        for st in s.get("shot_type_sequence", []):
            shot_types[st] += 1
    non_medium = sum(v for k, v in shot_types.items() if k != "medium")
    total_shots_typed = sum(shot_types.values())
    if non_medium > 0:
        print(f"  [PASS] shot_type variety: {dict(shot_types)} ({fmt_pct(non_medium, total_shots_typed)} non-medium)")
        passed += 1
    else:
        print(f"  [WARN] All shots still 'medium' -- VLM shot_type not working?")

    # 5. Per-shot keyframes exist
    shot_index_path = os.path.join(new_dir, "shot_index.json")
    if os.path.exists(shot_index_path):
        shot_index = load_json(shot_index_path)
        kf_count = sum(1 for s in shot_index if len(s.get("keyframe_paths", [])) > 0)
        total_kf = sum(len(s.get("keyframe_paths", [])) for s in shot_index)
        total_shots = len(shot_index)
        if kf_count > total_shots * 0.8:
            print(f"  [PASS] Per-shot keyframes: {kf_count}/{total_shots} shots have keyframes ({total_kf} total keyframes)")
            passed += 1
        else:
            print(f"  [WARN] Per-shot keyframes: only {kf_count}/{total_shots} shots have keyframes ({total_kf} total keyframes)")
    else:
        print(f"  [FAIL] shot_index.json missing")
        failed += 1

    # 6. Character registry exists
    cr_path = os.path.join(new_dir, "character_registry.json")
    if os.path.exists(cr_path):
        cr = load_json(cr_path)
        char_count = len(cr.get("characters", {}))
        print(f"  [PASS] character_registry.json present ({char_count} characters)")
        passed += 1
    else:
        print(f"  [FAIL] character_registry.json missing")
        failed += 1

    # 7. Dialogue index exists (standalone file or embedded in manifest)
    di_path = os.path.join(new_dir, "dialogue_index.json")
    if os.path.exists(di_path):
        di = load_json(di_path)
        print(f"  [PASS] dialogue_index.json present ({len(di)} lines)")
        passed += 1
    elif new_manifest and new_manifest.get("dialogue_index"):
        di = new_manifest["dialogue_index"]
        print(f"  [PASS] dialogue_index in manifest ({len(di)} lines)")
        passed += 1
    else:
        print(f"  [FAIL] dialogue_index missing (not standalone or in manifest)")
        failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return failed == 0


def run_quantitative_comparison(old_dir, new_dir, old_scenes, new_scenes):
    """Compare numbers between baseline and new output."""
    print_header("QUANTITATIVE COMPARISON")

    # Scene count
    print_metric("Scene count", len(old_scenes), len(new_scenes))

    # Total characters identified (sum across scenes)
    def count_characters(scenes):
        return sum(len(s.get("characters_present", [])) for s in scenes.values())

    old_chars = count_characters(old_scenes)
    new_chars = count_characters(new_scenes)
    print_metric("Total character identifications", old_chars, new_chars)

    # Unique character IDs
    def unique_char_ids(scenes):
        ids = set()
        for s in scenes.values():
            for c in s.get("characters_present", []):
                ids.add(c.get("character_id", c.get("name", "?")))
        return ids

    old_ids = unique_char_ids(old_scenes)
    new_ids = unique_char_ids(new_scenes)
    print_metric("Unique character IDs", len(old_ids), len(new_ids))

    # Unknown characters
    old_unknown = sum(
        1 for s in old_scenes.values()
        for c in s.get("characters_present", [])
        if "unknown" in c.get("character_id", c.get("name", "")).lower()
    )
    new_unknown = sum(
        1 for s in new_scenes.values()
        for c in s.get("characters_present", [])
        if "unknown" in c.get("character_id", c.get("name", "")).lower()
    )
    print_metric("'unknown' character refs", old_unknown, new_unknown)

    # Character name overlap
    shared = old_ids & new_ids
    if old_ids:
        print(f"  {'Character ID overlap':.<40} {len(shared)}/{len(old_ids)} old IDs found in new ({fmt_pct(len(shared), len(old_ids))})")
        lost = old_ids - new_ids
        gained = new_ids - old_ids
        if lost:
            print(f"    Lost:   {', '.join(sorted(lost))}")
        if gained:
            print(f"    Gained: {', '.join(sorted(gained))}")

    # Scene type distribution
    def scene_type_dist(scenes):
        return Counter(s.get("scene_type", "?") for s in scenes.values())

    old_st = scene_type_dist(old_scenes)
    new_st = scene_type_dist(new_scenes)
    print(f"\n  Scene type distribution:")
    all_types = sorted(set(list(old_st.keys()) + list(new_st.keys())))
    for t in all_types:
        o = old_st.get(t, 0)
        n = new_st.get(t, 0)
        marker = " *" if o != n else ""
        print(f"    {t:.<30} old: {o:>3}  | new: {n:>3}{marker}")

    # Content origin distribution (from shot_index)
    def content_origin_dist(directory):
        si_path = os.path.join(directory, "shot_index.json")
        if not os.path.exists(si_path):
            return Counter()
        si = load_json(si_path)
        return Counter(s.get("content_origin", "?") for s in si)

    old_co = content_origin_dist(old_dir)
    new_co = content_origin_dist(new_dir)
    if old_co or new_co:
        print(f"\n  Content origin distribution (shot-level):")
        all_origins = sorted(set(list(old_co.keys()) + list(new_co.keys())))
        for o in all_origins:
            ov = old_co.get(o, 0)
            nv = new_co.get(o, 0)
            marker = " *" if ov != nv else ""
            print(f"    {o:.<30} old: {ov:>3}  | new: {nv:>3}{marker}")

    # Speaker attribution confidence
    def speaker_stats(directory):
        di = None
        di_path = os.path.join(directory, "dialogue_index.json")
        if os.path.exists(di_path):
            di = load_json(di_path)
        else:
            # Fall back to dialogue_index embedded in episode_manifest
            mp = os.path.join(directory, "episode_manifest.json")
            if os.path.exists(mp):
                m = load_json(mp)
                di = m.get("dialogue_index")
        if not di:
            return 0, 0, 0
        total = len(di)
        high_conf = sum(1 for d in di if d.get("speaker_confidence", 0) >= 0.7)
        unknown_speaker = sum(1 for d in di if "unknown" in (d.get("speaker_id") or "").lower())
        return total, high_conf, unknown_speaker

    old_total, old_hc, old_unk = speaker_stats(old_dir)
    new_total, new_hc, new_unk = speaker_stats(new_dir)
    print(f"\n  Speaker attribution:")
    print_metric("Dialogue lines", old_total, new_total)
    print_metric("High confidence (>=0.7)", old_hc, new_hc)
    print_metric("Unknown speaker", old_unk, new_unk)

    # Cost comparison (from manifest processing_metadata)
    def get_cost(directory):
        mp = os.path.join(directory, "episode_manifest.json")
        if not os.path.exists(mp):
            return None
        m = load_json(mp)
        pm = m.get("processing_metadata", {})
        return pm.get("total_cost_usd") or pm.get("estimated_cost_usd")

    old_cost = get_cost(old_dir)
    new_cost = get_cost(new_dir)
    if old_cost or new_cost:
        print(f"\n  Cost:")
        print(f"  {'API cost':.<40} old: ${old_cost or '?'}  | new: ${new_cost or '?'}")


def run_spot_checks(old_scenes, new_scenes):
    """Print side-by-side comparisons for human review."""
    print_header("SPOT CHECKS")

    # Find scenes that exist in both (by approximate time matching)
    # Since scene boundaries may have changed, match by closest start_time
    def match_scenes(old_s, new_s):
        """Match old scenes to new scenes by closest start time."""
        matches = []
        new_by_time = sorted(new_s.values(), key=lambda s: s.get("start_time_seconds", 0))
        for old_num, old_scene in sorted(old_s.items()):
            old_start = old_scene.get("start_time_seconds", 0)
            best = None
            best_diff = float("inf")
            for ns in new_by_time:
                diff = abs(ns.get("start_time_seconds", 0) - old_start)
                if diff < best_diff:
                    best_diff = diff
                    best = ns
            if best and best_diff < 10.0:  # within 10 seconds
                matches.append((old_scene, best, best_diff))
        return matches

    matches = match_scenes(old_scenes, new_scenes)

    # 1. Random sample of 5 matched scenes
    sample_size = min(5, len(matches))
    if sample_size > 0:
        print(f"\n  --- Random scene narrative comparison ({sample_size} scenes) ---")
        sample = random.sample(matches, sample_size)
        for old_s, new_s, tdiff in sample:
            old_num = old_s.get("scene_number", "?")
            new_num = new_s.get("scene_number", "?")
            time_note = f" (dt={tdiff:.1f}s)" if tdiff > 0.5 else ""
            print(f"\n  Scene {old_num} -> {new_num}{time_note} [{old_s.get('start_time_seconds', 0):.1f}s]:")
            print(f"    OLD: {old_s.get('scene_narrative', '(none)')}")
            print(f"    NEW: {new_s.get('scene_narrative', '(none)')}")

            old_chars = [c.get("name", "?") for c in old_s.get("characters_present", [])]
            new_chars = [c.get("name", "?") for c in new_s.get("characters_present", [])]
            if old_chars != new_chars:
                print(f"    OLD chars: {', '.join(old_chars) or '(none)'}")
                print(f"    NEW chars: {', '.join(new_chars) or '(none)'}")

            # Show new fields
            rp = new_s.get("reconstruction_prompt", "")
            if rp:
                print(f"    NEW reconstruction_prompt: {rp[:120]}{'...' if len(rp) > 120 else ''}")

    # 2. Scenes with biggest character count difference
    char_diffs = []
    for old_s, new_s, tdiff in matches:
        old_c = len(old_s.get("characters_present", []))
        new_c = len(new_s.get("characters_present", []))
        if old_c != new_c:
            char_diffs.append((abs(new_c - old_c), old_c, new_c, old_s, new_s))
    char_diffs.sort(key=lambda x: x[0], reverse=True)

    if char_diffs:
        print(f"\n  --- Biggest character count changes (top 3) ---")
        for diff, old_c, new_c, old_s, new_s in char_diffs[:3]:
            old_names = [c.get("name", "?") for c in old_s.get("characters_present", [])]
            new_names = [c.get("name", "?") for c in new_s.get("characters_present", [])]
            print(f"\n  Scene {old_s.get('scene_number','?')} -> {new_s.get('scene_number','?')} [{old_s.get('start_time_seconds',0):.1f}s]:")
            print(f"    OLD ({old_c}): {', '.join(old_names) or '(none)'}")
            print(f"    NEW ({new_c}): {', '.join(new_names) or '(none)'}")

    # 3. Scene type changes
    type_changes = []
    for old_s, new_s, tdiff in matches:
        old_t = old_s.get("scene_type", "?")
        new_t = new_s.get("scene_type", "?")
        if old_t != new_t:
            type_changes.append((old_s, new_s, old_t, new_t))

    if type_changes:
        print(f"\n  --- Scene type changes ({len(type_changes)} total) ---")
        for old_s, new_s, old_t, new_t in type_changes[:5]:
            print(f"    Scene {old_s.get('scene_number','?')} -> {new_s.get('scene_number','?')}: {old_t} -> {new_t}")
    else:
        print(f"\n  No scene type changes detected (among matched scenes)")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs new anime metadata output")
    parser.add_argument("episode_id", help="Episode ID (e.g. 0001)")
    parser.add_argument("--baseline-suffix", default="_baseline",
                        help="Suffix for baseline directory (default: _baseline)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for spot-check sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    base_dir = os.path.join("output", "anime_metadata")
    old_dir = os.path.join(base_dir, f"{args.episode_id}{args.baseline_suffix}")
    new_dir = os.path.join(base_dir, args.episode_id)

    if not os.path.isdir(old_dir):
        print(f"Baseline directory not found: {old_dir}")
        sys.exit(1)
    if not os.path.isdir(new_dir):
        print(f"New output directory not found: {new_dir}")
        sys.exit(1)

    print(f"Comparing: {old_dir}  vs  {new_dir}")

    # Load data
    old_scenes = load_scenes(old_dir)
    new_scenes = load_scenes(new_dir)

    new_manifest_path = os.path.join(new_dir, "episode_manifest.json")
    new_manifest = load_json(new_manifest_path) if os.path.exists(new_manifest_path) else None

    # Run checks
    structural_ok = run_structural_checks(new_dir, new_scenes, new_manifest)
    run_quantitative_comparison(old_dir, new_dir, old_scenes, new_scenes)
    run_spot_checks(old_scenes, new_scenes)

    print(f"\n{'='*60}")
    if structural_ok:
        print("  All structural checks passed. Review metrics above.")
    else:
        print("  STRUCTURAL CHECK FAILURES -- investigate before batch rerun.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
