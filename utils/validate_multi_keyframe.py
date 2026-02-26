"""Compare single-keyframe baseline vs multi-keyframe anime metadata output.

Usage:
    python validate_multi_keyframe.py 0001
    python validate_multi_keyframe.py 0001 --baseline-suffix _baseline

Compares output/anime_metadata/{episode_id}_baseline/ vs output/anime_metadata/{episode_id}/
Focuses on metrics specific to the multi-keyframe change: keyframe extraction stats,
VLM input coverage, shot type/content origin quality, speaker attribution, cost.
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
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_metric(label, old_val, new_val, unit=""):
    delta = ""
    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
        diff = new_val - old_val
        if old_val > 0:
            pct = diff / old_val * 100
            delta = f"  ({'+' if diff >= 0 else ''}{diff}{unit}, {'+' if pct >= 0 else ''}{pct:.1f}%)"
        else:
            delta = f"  ({'+' if diff >= 0 else ''}{diff}{unit})"
    print(f"  {label:.<45} old: {old_val}{unit}  | new: {new_val}{unit}{delta}")


def get_shot_index(directory):
    si_path = os.path.join(directory, "shot_index.json")
    if os.path.exists(si_path):
        return load_json(si_path)
    return []


def get_keyframe_paths(shot):
    """Get keyframe paths from a shot, handling both old and new schema."""
    paths = shot.get("keyframe_paths")
    if paths is not None:
        return paths if isinstance(paths, list) else [paths]
    old = shot.get("keyframe_path")
    if old:
        return [old]
    return []


def get_dialogue_index(directory):
    di_path = os.path.join(directory, "dialogue_index.json")
    if os.path.exists(di_path):
        return load_json(di_path)
    mp = os.path.join(directory, "episode_manifest.json")
    if os.path.exists(mp):
        m = load_json(mp)
        return m.get("dialogue_index", [])
    return []


# -- Section 1: Keyframe extraction stats --

def report_keyframe_stats(old_shots, new_shots):
    print_header("1. KEYFRAME EXTRACTION STATS")

    old_total_kf = sum(len(get_keyframe_paths(s)) for s in old_shots)
    new_total_kf = sum(len(get_keyframe_paths(s)) for s in new_shots)
    old_shots_with_kf = sum(1 for s in old_shots if len(get_keyframe_paths(s)) > 0)
    new_shots_with_kf = sum(1 for s in new_shots if len(get_keyframe_paths(s)) > 0)

    print_metric("Total keyframes", old_total_kf, new_total_kf)
    print_metric("Shots with keyframes", old_shots_with_kf, new_shots_with_kf)
    print_metric("Shots with 0 keyframes", len(old_shots) - old_shots_with_kf,
                 len(new_shots) - new_shots_with_kf)

    # Distribution by shot duration bucket
    buckets = {"<3s": (0, 3), "3-6s": (3, 6), ">6s": (6, float("inf"))}
    print(f"\n  Keyframe count distribution by shot duration:")
    print(f"  {'Bucket':>8} | {'Old shots':>10} | {'Old KF/shot':>12} | {'New shots':>10} | {'New KF/shot':>12}")
    print(f"  {'-'*62}")

    for label, (lo, hi) in buckets.items():
        old_bucket = [s for s in old_shots if lo <= s.get("duration_seconds", 0) < hi]
        new_bucket = [s for s in new_shots if lo <= s.get("duration_seconds", 0) < hi]
        old_kf = sum(len(get_keyframe_paths(s)) for s in old_bucket)
        new_kf = sum(len(get_keyframe_paths(s)) for s in new_bucket)
        old_avg = f"{old_kf/len(old_bucket):.1f}" if old_bucket else "-"
        new_avg = f"{new_kf/len(new_bucket):.1f}" if new_bucket else "-"
        print(f"  {label:>8} | {len(old_bucket):>10} | {old_avg:>12} | {len(new_bucket):>10} | {new_avg:>12}")


# -- Section 2: VLM input coverage --

def report_vlm_coverage(old_scenes, new_scenes):
    print_header("2. VLM INPUT COVERAGE")

    def scene_kf_count(scene):
        paths = scene.get("keyframe_paths")
        if paths and isinstance(paths, list):
            return len(paths)
        if scene.get("keyframe_path"):
            return 1
        return 0

    old_total_vlm_images = sum(scene_kf_count(s) for s in old_scenes.values())
    new_total_vlm_images = sum(scene_kf_count(s) for s in new_scenes.values())

    print_metric("Total VLM images across episode", old_total_vlm_images, new_total_vlm_images)

    # Images per scene distribution
    old_counts = [scene_kf_count(s) for s in old_scenes.values()]
    new_counts = [scene_kf_count(s) for s in new_scenes.values()]

    old_avg = sum(old_counts) / len(old_counts) if old_counts else 0
    new_avg = sum(new_counts) / len(new_counts) if new_counts else 0
    print_metric("Avg images per scene", round(old_avg, 2), round(new_avg, 2))

    if new_counts:
        print(f"\n  New run images-per-scene distribution:")
        dist = Counter(new_counts)
        for k in sorted(dist.keys()):
            bar = "#" * min(dist[k], 40)
            print(f"    {k:>3} images: {dist[k]:>4} scenes  {bar}")

    # Scenes where VLM saw all shots vs sampled
    new_all_covered = 0
    new_sampled = 0
    for s in new_scenes.values():
        shot_count = s.get("shot_count", 0)
        kf_count = scene_kf_count(s)
        if shot_count > 0 and kf_count >= shot_count:
            new_all_covered += 1
        elif shot_count > 0:
            new_sampled += 1

    total_new_story = new_all_covered + new_sampled
    if total_new_story > 0:
        print(f"\n  VLM saw all shots:  {new_all_covered}/{total_new_story} scenes ({fmt_pct(new_all_covered, total_new_story)})")
        print(f"  VLM sampled shots:  {new_sampled}/{total_new_story} scenes ({fmt_pct(new_sampled, total_new_story)})")


# -- Section 3: Shot type classification quality --

def report_shot_types(old_scenes, new_scenes, old_shots, new_shots):
    print_header("3. SHOT TYPE CLASSIFICATION")

    old_types = Counter(s.get("shot_type", "medium") for s in old_shots)
    new_types = Counter(s.get("shot_type", "medium") for s in new_shots)

    old_non_medium = sum(v for k, v in old_types.items() if k != "medium")
    new_non_medium = sum(v for k, v in new_types.items() if k != "medium")
    old_total = sum(old_types.values())
    new_total = sum(new_types.values())

    print_metric("% non-medium shot types",
                 f"{old_non_medium}/{old_total} ({fmt_pct(old_non_medium, old_total)})",
                 f"{new_non_medium}/{new_total} ({fmt_pct(new_non_medium, new_total)})")

    # Distribution comparison
    all_types = sorted(set(list(old_types.keys()) + list(new_types.keys())))
    print(f"\n  Shot type distribution:")
    print(f"  {'Type':.<25} {'Old':>6}  {'New':>6}  {'Delta':>6}")
    print(f"  {'-'*50}")
    for t in all_types:
        o = old_types.get(t, 0)
        n = new_types.get(t, 0)
        d = n - o
        sign = "+" if d > 0 else ""
        marker = " ***" if abs(d) > 5 else ""
        print(f"  {t:.<25} {o:>6}  {n:>6}  {sign}{d:>5}{marker}")

    # Shots that changed from "medium" to something else
    # Match by shot_id or index
    old_by_id = {s.get("shot_id", i): s for i, s in enumerate(old_shots)}
    new_by_id = {s.get("shot_id", i): s for i, s in enumerate(new_shots)}
    common_ids = set(old_by_id.keys()) & set(new_by_id.keys())

    medium_to_other = []
    for sid in common_ids:
        old_t = old_by_id[sid].get("shot_type", "medium")
        new_t = new_by_id[sid].get("shot_type", "medium")
        if old_t == "medium" and new_t != "medium":
            medium_to_other.append((sid, new_t, new_by_id[sid].get("start_time_seconds", 0)))

    if medium_to_other:
        print(f"\n  Shots reclassified from 'medium' ({len(medium_to_other)} total, showing first 10):")
        for sid, new_t, start in medium_to_other[:10]:
            print(f"    Shot {sid} @ {start:.1f}s -> {new_t}")


# -- Section 4: Content origin classification --

def report_content_origin(old_shots, new_shots):
    print_header("4. CONTENT ORIGIN CLASSIFICATION")

    old_origins = Counter(s.get("content_origin", "unknown") for s in old_shots)
    new_origins = Counter(s.get("content_origin", "unknown") for s in new_shots)

    all_origins = sorted(set(list(old_origins.keys()) + list(new_origins.keys())))
    print(f"  {'Origin':.<25} {'Old':>6}  {'New':>6}  {'Delta':>6}")
    print(f"  {'-'*50}")
    for o in all_origins:
        ov = old_origins.get(o, 0)
        nv = new_origins.get(o, 0)
        d = nv - ov
        sign = "+" if d > 0 else ""
        print(f"  {o:.<25} {ov:>6}  {nv:>6}  {sign}{d:>5}")

    # Shots that flipped between manga_derived <-> anime_original
    old_by_id = {s.get("shot_id", i): s for i, s in enumerate(old_shots)}
    new_by_id = {s.get("shot_id", i): s for i, s in enumerate(new_shots)}
    common_ids = set(old_by_id.keys()) & set(new_by_id.keys())

    flips = []
    for sid in common_ids:
        old_o = old_by_id[sid].get("content_origin", "unknown")
        new_o = new_by_id[sid].get("content_origin", "unknown")
        if old_o != new_o and {old_o, new_o} & {"manga_derived", "anime_original"}:
            flips.append((sid, old_o, new_o, new_by_id[sid].get("start_time_seconds", 0)))

    if flips:
        print(f"\n  manga_derived <-> anime_original flips ({len(flips)} total, showing first 10):")
        for sid, old_o, new_o, start in flips[:10]:
            print(f"    Shot {sid} @ {start:.1f}s: {old_o} -> {new_o}")
    else:
        print(f"\n  No manga_derived <-> anime_original flips detected")


# -- Section 5: Speaker attribution quality --

def report_speaker_attribution(old_dir, new_dir):
    print_header("5. SPEAKER ATTRIBUTION")

    old_di = get_dialogue_index(old_dir)
    new_di = get_dialogue_index(new_dir)

    if not old_di and not new_di:
        print("  No dialogue index found in either run.")
        return

    print_metric("Total dialogue lines", len(old_di), len(new_di))

    # Mean speaker confidence
    def mean_conf(di):
        confs = [d.get("speaker_confidence", 0) for d in di if d.get("speaker_confidence") is not None]
        return round(sum(confs) / len(confs), 3) if confs else 0

    print_metric("Mean speaker_confidence", mean_conf(old_di), mean_conf(new_di))

    # High confidence (>= 0.7)
    old_hc = sum(1 for d in old_di if d.get("speaker_confidence", 0) >= 0.7)
    new_hc = sum(1 for d in new_di if d.get("speaker_confidence", 0) >= 0.7)
    print_metric("High confidence (>=0.7)", f"{old_hc} ({fmt_pct(old_hc, len(old_di))})",
                 f"{new_hc} ({fmt_pct(new_hc, len(new_di))})")

    # Unknown speaker
    old_unk = sum(1 for d in old_di if "unknown" in (d.get("speaker_id") or "").lower())
    new_unk = sum(1 for d in new_di if "unknown" in (d.get("speaker_id") or "").lower())
    print_metric("Unknown speaker", f"{old_unk} ({fmt_pct(old_unk, len(old_di))})",
                 f"{new_unk} ({fmt_pct(new_unk, len(new_di))})")

    # Lines where speaker changed
    changes = []
    min_len = min(len(old_di), len(new_di))
    for i in range(min_len):
        old_spk = old_di[i].get("speaker_id", "")
        new_spk = new_di[i].get("speaker_id", "")
        if old_spk != new_spk:
            text = old_di[i].get("text", "")[:60]
            changes.append((i, old_spk, new_spk, text))

    if changes:
        print(f"\n  Speaker changed on {len(changes)}/{min_len} matched lines (showing first 10):")
        for idx, old_spk, new_spk, text in changes[:10]:
            print(f"    Line {idx}: {old_spk or '(none)'} -> {new_spk or '(none)'}  \"{text}...\"")


# -- Section 6: Character identification --

def report_characters(old_scenes, new_scenes):
    print_header("6. CHARACTER IDENTIFICATION")

    def total_char_ids(scenes):
        return sum(len(s.get("characters_present", [])) for s in scenes.values())

    def unique_ids(scenes):
        ids = set()
        for s in scenes.values():
            for c in s.get("characters_present", []):
                ids.add(c.get("character_id", c.get("name", "?")))
        return ids

    def unknown_scene_count(scenes):
        count = 0
        for s in scenes.values():
            for c in s.get("characters_present", []):
                cid = c.get("character_id", c.get("name", ""))
                if "unknown" in cid.lower():
                    count += 1
                    break
        return count

    print_metric("Total character identifications", total_char_ids(old_scenes), total_char_ids(new_scenes))

    old_ids = unique_ids(old_scenes)
    new_ids = unique_ids(new_scenes)
    print_metric("Unique character IDs", len(old_ids), len(new_ids))

    gained = new_ids - old_ids
    lost = old_ids - new_ids
    if gained:
        print(f"  Characters gained: {', '.join(sorted(gained))}")
    if lost:
        print(f"  Characters LOST:   {', '.join(sorted(lost))}")
    if not gained and not lost:
        print(f"  Character set unchanged ({len(old_ids)} IDs)")

    old_unk_scenes = unknown_scene_count(old_scenes)
    new_unk_scenes = unknown_scene_count(new_scenes)
    print_metric("Scenes with 'unknown' characters", old_unk_scenes, new_unk_scenes)


# -- Section 7: Cost comparison --

def report_cost(old_dir, new_dir):
    print_header("7. COST COMPARISON")

    def get_manifest_metadata(directory):
        mp = os.path.join(directory, "episode_manifest.json")
        if not os.path.exists(mp):
            return {}
        m = load_json(mp)
        return m.get("processing_metadata", {})

    old_pm = get_manifest_metadata(old_dir)
    new_pm = get_manifest_metadata(new_dir)

    old_cost = old_pm.get("total_cost_usd") or old_pm.get("estimated_cost_usd")
    new_cost = new_pm.get("total_cost_usd") or new_pm.get("estimated_cost_usd")

    if old_cost is not None or new_cost is not None:
        print_metric("API cost (USD)", f"${old_cost or '?'}", f"${new_cost or '?'}")

    old_in = old_pm.get("total_input_tokens", old_pm.get("input_tokens"))
    new_in = new_pm.get("total_input_tokens", new_pm.get("input_tokens"))
    if old_in is not None or new_in is not None:
        print_metric("Input tokens", old_in or "?", new_in or "?")

    old_out = old_pm.get("total_output_tokens", old_pm.get("output_tokens"))
    new_out = new_pm.get("total_output_tokens", new_pm.get("output_tokens"))
    if old_out is not None or new_out is not None:
        print_metric("Output tokens", old_out or "?", new_out or "?")

    old_calls = old_pm.get("vlm_call_count") or old_pm.get("vlm_calls")
    new_calls = new_pm.get("vlm_call_count") or new_pm.get("vlm_calls")
    if old_calls is not None or new_calls is not None:
        print_metric("VLM call count", old_calls or "?", new_calls or "?")

    # Cost per scene
    if new_cost and isinstance(new_cost, (int, float)):
        old_scenes = load_scenes(old_dir)
        new_scenes = load_scenes(new_dir)
        old_scene_count = len(old_scenes)
        new_scene_count = len(new_scenes)
        if new_scene_count > 0:
            new_cps = new_cost / new_scene_count
            old_cps = old_cost / old_scene_count if old_cost and old_scene_count else None
            if old_cps is not None:
                print_metric("Cost per scene", f"${old_cps:.4f}", f"${new_cps:.4f}")
            else:
                print(f"  {'Cost per scene':.<45} new: ${new_cps:.4f}")


# -- Section 8: Spot checks --

def report_spot_checks(old_scenes, new_scenes):
    print_header("8. SPOT CHECKS (5 random scenes)")

    # Match by scene number (prefer) or closest start time
    common_nums = sorted(set(old_scenes.keys()) & set(new_scenes.keys()))

    if not common_nums:
        print("  No matching scene numbers to compare.")
        return

    sample_size = min(5, len(common_nums))
    sample_nums = random.sample(common_nums, sample_size)

    for snum in sorted(sample_nums):
        old_s = old_scenes[snum]
        new_s = new_scenes[snum]

        old_kf_count = 0
        kp = old_s.get("keyframe_paths")
        if kp and isinstance(kp, list):
            old_kf_count = len(kp)
        elif old_s.get("keyframe_path"):
            old_kf_count = 1

        new_kf_count = 0
        kp = new_s.get("keyframe_paths")
        if kp and isinstance(kp, list):
            new_kf_count = len(kp)

        dur = new_s.get("duration_seconds", old_s.get("duration_seconds", 0))
        shot_count = new_s.get("shot_count", old_s.get("shot_count", 0))

        print(f"\n  Scene {snum:03d}  ({dur:.1f}s, {shot_count} shots, VLM images: {old_kf_count} -> {new_kf_count})")
        print(f"  {'-'*60}")

        # Narrative comparison
        old_narr = old_s.get("scene_narrative", "(none)")
        new_narr = new_s.get("scene_narrative", "(none)")
        if old_narr != new_narr:
            print(f"    OLD narrative: {old_narr[:120]}{'...' if len(old_narr) > 120 else ''}")
            print(f"    NEW narrative: {new_narr[:120]}{'...' if len(new_narr) > 120 else ''}")
        else:
            print(f"    Narrative: unchanged")

        # Characters comparison
        old_chars = sorted(c.get("character_id", c.get("name", "?")) for c in old_s.get("characters_present", []))
        new_chars = sorted(c.get("character_id", c.get("name", "?")) for c in new_s.get("characters_present", []))
        if old_chars != new_chars:
            print(f"    OLD chars: {', '.join(old_chars) or '(none)'}")
            print(f"    NEW chars: {', '.join(new_chars) or '(none)'}")
        else:
            print(f"    Characters: unchanged ({len(old_chars)})")

        # Shot types comparison
        old_sts = old_s.get("shot_type_sequence", [])
        new_sts = new_s.get("shot_type_sequence", [])
        if old_sts != new_sts:
            old_dist = Counter(old_sts)
            new_dist = Counter(new_sts)
            print(f"    OLD shot types: {dict(old_dist)}")
            print(f"    NEW shot types: {dict(new_dist)}")
        else:
            print(f"    Shot types: unchanged")


def main():
    parser = argparse.ArgumentParser(
        description="Compare single-keyframe baseline vs multi-keyframe anime metadata output"
    )
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

    print(f"{'='*70}")
    print(f"  Multi-Keyframe Validation Report")
    print(f"  Baseline (single-keyframe): {old_dir}")
    print(f"  New (multi-keyframe):       {new_dir}")
    print(f"{'='*70}")

    # Load data
    old_scenes = load_scenes(old_dir)
    new_scenes = load_scenes(new_dir)
    old_shots = get_shot_index(old_dir)
    new_shots = get_shot_index(new_dir)

    if not old_shots:
        print("WARNING: No shot_index.json in baseline -- keyframe/shot-level comparisons will be limited")
    if not new_shots:
        print("WARNING: No shot_index.json in new output -- keyframe/shot-level comparisons will be limited")

    # Run all sections
    report_keyframe_stats(old_shots, new_shots)
    report_vlm_coverage(old_scenes, new_scenes)
    report_shot_types(old_scenes, new_scenes, old_shots, new_shots)
    report_content_origin(old_shots, new_shots)
    report_speaker_attribution(old_dir, new_dir)
    report_characters(old_scenes, new_scenes)
    report_cost(old_dir, new_dir)
    report_spot_checks(old_scenes, new_scenes)

    # Verdict
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    issues = []
    improvements = []

    # Check for regressions
    new_total_kf = sum(len(get_keyframe_paths(s)) for s in new_shots) if new_shots else 0
    old_total_kf = sum(len(get_keyframe_paths(s)) for s in old_shots) if old_shots else 0
    if new_total_kf <= old_total_kf and new_shots:
        issues.append(f"Total keyframes did not increase (old={old_total_kf}, new={new_total_kf})")

    old_non_medium = sum(1 for s in old_shots if s.get("shot_type", "medium") != "medium")
    new_non_medium = sum(1 for s in new_shots if s.get("shot_type", "medium") != "medium")
    if new_non_medium > old_non_medium:
        improvements.append(f"Shot type variety improved ({old_non_medium} -> {new_non_medium} non-medium)")

    old_di = get_dialogue_index(old_dir)
    new_di = get_dialogue_index(new_dir)
    old_hc = sum(1 for d in old_di if d.get("speaker_confidence", 0) >= 0.7)
    new_hc = sum(1 for d in new_di if d.get("speaker_confidence", 0) >= 0.7)
    if new_hc > old_hc:
        improvements.append(f"High-confidence speaker attributions improved ({old_hc} -> {new_hc})")
    elif new_hc < old_hc * 0.9:
        issues.append(f"High-confidence speaker attributions decreased ({old_hc} -> {new_hc})")

    old_unk = sum(1 for d in old_di if "unknown" in (d.get("speaker_id") or "").lower())
    new_unk = sum(1 for d in new_di if "unknown" in (d.get("speaker_id") or "").lower())
    if new_unk < old_unk:
        improvements.append(f"Unknown speakers decreased ({old_unk} -> {new_unk})")

    if issues:
        print("  ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    if improvements:
        print("  IMPROVEMENTS:")
        for imp in improvements:
            print(f"    + {imp}")
    if not issues and not improvements:
        print("  No significant changes detected. Verify pipeline ran with multi-keyframe config.")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
