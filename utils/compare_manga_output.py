"""Compare two manga metadata output directories (baseline vs new).

Usage:
    python compare_manga_output.py output/manga_metadata/0001_baseline output/manga_metadata/0001
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
from collections import Counter
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_iou(box1, box2):
    """IOU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def greedy_iou_match(bboxes_a, bboxes_b, threshold=0.1):
    """Greedy IOU matching between two lists of [x1,y1,x2,y2] bboxes.

    Returns list of (idx_a, idx_b, iou) triples.
    """
    pairs = []
    for ai, a in enumerate(bboxes_a):
        for bi, b in enumerate(bboxes_b):
            iou = compute_iou(a, b)
            pairs.append((iou, ai, bi))
    pairs.sort(reverse=True)

    matched = []
    used_a, used_b = set(), set()
    for iou, ai, bi in pairs:
        if iou < threshold:
            break
        if ai not in used_a and bi not in used_b:
            matched.append((ai, bi, iou))
            used_a.add(ai)
            used_b.add(bi)
    return matched


def bbox_area(box):
    """Area of a normalized [x1,y1,x2,y2] box (0-1 range)."""
    w = max(0, box[2] - box[0])
    h = max(0, box[3] - box[1])
    return w * h


def page_stats(page_json):
    """Extract structural metrics from a per-page JSON."""
    panels = page_json.get("panels", [])
    text_elems = sum(len(p.get("text_elements", [])) for p in panels)
    sfx = sum(len(p.get("sound_effects", [])) for p in panels)
    chars = set()
    for p in panels:
        for c in p.get("characters_present", []):
            cid = c.get("character_id")
            if cid:
                chars.add(cid)
    speaker_ids = []
    for p in panels:
        for t in p.get("text_elements", []):
            if t.get("type") in ("dialog", "internal_monologue"):
                speaker_ids.append(t.get("speaker_id"))
    has_bbox = any("bbox" in p for p in panels)
    has_panel_image_path = any(
        p.get("panel_image_path") is not None for p in panels
    )

    # Bbox source breakdown
    bbox_sources = Counter()
    bboxes = []
    bbox_areas = []
    panels_with_bbox = 0
    panels_with_crop = 0
    for p in panels:
        src = p.get("bbox_source", "unset" if "bbox" not in p else "vlm_legacy")
        bbox_sources[src] += 1
        box = p.get("bbox")
        if box and len(box) == 4:
            try:
                box = [float(v) for v in box]
                bboxes.append(box)
                bbox_areas.append(bbox_area(box))
                panels_with_bbox += 1
            except (TypeError, ValueError):
                pass
        if p.get("panel_image_path") is not None:
            panels_with_crop += 1

    return {
        "page_type": page_json.get("page_type"),
        "panels": len(panels),
        "text_elements": text_elems,
        "sfx": sfx,
        "characters": chars,
        "speaker_ids": speaker_ids,
        "has_bbox": has_bbox,
        "has_panel_image_path": has_panel_image_path,
        "bbox_sources": bbox_sources,
        "bboxes": bboxes,
        "bbox_areas": bbox_areas,
        "panels_with_bbox": panels_with_bbox,
        "panels_with_crop": panels_with_crop,
    }


def fmt_bbox_src(sources):
    """Format bbox source counter as compact string like 'K:3 V:1'."""
    parts = []
    k = sources.get("kumiko", 0)
    v = sources.get("vlm", 0) + sources.get("vlm_legacy", 0)
    u = sources.get("unset", 0)
    if k:
        parts.append(f"K:{k}")
    if v:
        parts.append(f"V:{v}")
    if u:
        parts.append(f"-:{u}")
    return " ".join(parts) if parts else "-"


def compare(baseline_dir, new_dir):
    baseline_dir = Path(baseline_dir)
    new_dir = Path(new_dir)

    print("=" * 90)
    print("Manga Metadata Structural Comparison")
    print(f"  Baseline: {baseline_dir}")
    print(f"  New:      {new_dir}")
    print("=" * 90)

    # ---- Discover page files ----
    baseline_pages = sorted(baseline_dir.glob("page_*.json"))
    new_pages = sorted(new_dir.glob("page_*.json"))

    baseline_nums = {int(p.stem.split("_")[1]) for p in baseline_pages}
    new_nums = {int(p.stem.split("_")[1]) for p in new_pages}

    print(f"\nPages: baseline={len(baseline_pages)}, new={len(new_pages)}")
    if baseline_nums - new_nums:
        print(f"  WARNING: pages in baseline but NOT in new: {sorted(baseline_nums - new_nums)}")
    if new_nums - baseline_nums:
        print(f"  New pages not in baseline: {sorted(new_nums - baseline_nums)}")

    common = sorted(baseline_nums & new_nums)

    # ---- Per-page comparison ----
    print(f"\n{'Page':>6} | {'Panels':>14} | {'Text Elems':>14} | {'SFX':>10} | {'Chars':>10} | {'BBox':>6} | {'BBox Src':>10}")
    print("-" * 90)

    total_b = {"panels": 0, "text": 0, "sfx": 0, "chars": set(), "speaker_total": 0, "speaker_filled": 0,
               "has_bbox": 0, "bbox_sources": Counter(), "bbox_areas": [], "panels_with_bbox": 0, "panels_with_crop": 0}
    total_n = {"panels": 0, "text": 0, "sfx": 0, "chars": set(), "speaker_total": 0, "speaker_filled": 0,
               "has_bbox": 0, "bbox_sources": Counter(), "bbox_areas": [], "panels_with_bbox": 0, "panels_with_crop": 0}

    page_flags = []
    page_ious = []  # (page_num, mean_iou, num_matched, num_unmatched)

    for pnum in common:
        b = page_stats(load_json(baseline_dir / f"page_{pnum:03d}.json"))
        n = page_stats(load_json(new_dir / f"page_{pnum:03d}.json"))

        panel_diff = n["panels"] - b["panels"]
        text_diff = n["text_elements"] - b["text_elements"]
        bbox_str = "Y" if n["has_bbox"] else "-"
        bbox_src_str = fmt_bbox_src(n["bbox_sources"])

        def fmt_diff(old, new):
            d = new - old
            if d == 0:
                return f"{new:>5}  (=)"
            sign = "+" if d > 0 else ""
            return f"{new:>5} ({sign}{d})"

        flags = []
        if abs(panel_diff) > 2:
            flags.append(f"panels {panel_diff:+d}")
        if abs(text_diff) > 3:
            flags.append(f"text {text_diff:+d}")
        missing_chars = b["characters"] - n["characters"]
        if missing_chars:
            flags.append(f"lost chars: {missing_chars}")

        # Bbox IOU between baseline and new
        mean_iou = None
        if b["bboxes"] and n["bboxes"]:
            matched = greedy_iou_match(b["bboxes"], n["bboxes"])
            if matched:
                ious = [m[2] for m in matched]
                mean_iou = sum(ious) / len(ious)
                num_unmatched = max(len(b["bboxes"]), len(n["bboxes"])) - len(matched)
                page_ious.append((pnum, mean_iou, len(matched), num_unmatched))
                if mean_iou < 0.3:
                    flags.append(f"bbox IOU={mean_iou:.2f}")

        # Flag bbox_source transitions
        b_has_kumiko = b["bbox_sources"].get("kumiko", 0) > 0
        n_has_kumiko = n["bbox_sources"].get("kumiko", 0) > 0
        if n_has_kumiko and not b_has_kumiko:
            flags.append("new: kumiko bboxes")

        flag_str = " *** " + ", ".join(flags) if flags else ""

        print(f"{pnum:>6} | {fmt_diff(b['panels'], n['panels']):>14} | "
              f"{fmt_diff(b['text_elements'], n['text_elements']):>14} | "
              f"{fmt_diff(b['sfx'], n['sfx']):>10} | "
              f"{fmt_diff(len(b['characters']), len(n['characters'])):>10} | "
              f"{bbox_str:>6} | {bbox_src_str:>10}{flag_str}")

        if flags:
            page_flags.append((pnum, flags))

        # Accumulate totals
        total_b["panels"] += b["panels"]
        total_b["text"] += b["text_elements"]
        total_b["sfx"] += b["sfx"]
        total_b["chars"] |= b["characters"]
        total_b["speaker_total"] += len(b["speaker_ids"])
        total_b["speaker_filled"] += sum(1 for s in b["speaker_ids"] if s)
        if b["has_bbox"]:
            total_b["has_bbox"] += 1
        total_b["bbox_sources"] += b["bbox_sources"]
        total_b["bbox_areas"].extend(b["bbox_areas"])
        total_b["panels_with_bbox"] += b["panels_with_bbox"]
        total_b["panels_with_crop"] += b["panels_with_crop"]

        total_n["panels"] += n["panels"]
        total_n["text"] += n["text_elements"]
        total_n["sfx"] += n["sfx"]
        total_n["chars"] |= n["characters"]
        total_n["speaker_total"] += len(n["speaker_ids"])
        total_n["speaker_filled"] += sum(1 for s in n["speaker_ids"] if s)
        if n["has_bbox"]:
            total_n["has_bbox"] += 1
        total_n["bbox_sources"] += n["bbox_sources"]
        total_n["bbox_areas"].extend(n["bbox_areas"])
        total_n["panels_with_bbox"] += n["panels_with_bbox"]
        total_n["panels_with_crop"] += n["panels_with_crop"]

    # ---- Totals ----
    print("\n" + "=" * 90)
    print("TOTALS")
    print("=" * 90)

    def pct_change(old, new):
        if old == 0:
            return "N/A"
        return f"{((new - old) / old) * 100:+.1f}%"

    print(f"  Panels:        baseline={total_b['panels']:>5}, new={total_n['panels']:>5}  ({pct_change(total_b['panels'], total_n['panels'])})")
    print(f"  Text elements: baseline={total_b['text']:>5}, new={total_n['text']:>5}  ({pct_change(total_b['text'], total_n['text'])})")
    print(f"  SFX:           baseline={total_b['sfx']:>5}, new={total_n['sfx']:>5}  ({pct_change(total_b['sfx'], total_n['sfx'])})")
    print(f"  Unique chars:  baseline={len(total_b['chars']):>5}, new={len(total_n['chars']):>5}")

    # Speaker attribution rate
    b_rate = (total_b["speaker_filled"] / total_b["speaker_total"] * 100) if total_b["speaker_total"] else 0
    n_rate = (total_n["speaker_filled"] / total_n["speaker_total"] * 100) if total_n["speaker_total"] else 0
    print(f"  Speaker attr:  baseline={b_rate:.1f}%, new={n_rate:.1f}%  (delta={n_rate - b_rate:+.1f}pp)")

    # Bbox coverage
    print(f"  Pages w/bbox:  baseline={total_b['has_bbox']}/{len(common)}, new={total_n['has_bbox']}/{len(common)}")

    # ---- Panel Bbox Analysis ----
    print("\n" + "=" * 90)
    print("PANEL BBOX ANALYSIS")
    print("=" * 90)

    # Bbox source breakdown
    print("\n  Bbox source breakdown:")
    for label, totals in [("Baseline", total_b), ("New", total_n)]:
        src = totals["bbox_sources"]
        parts = []
        for key in ("kumiko", "vlm", "vlm_legacy", "unset"):
            count = src.get(key, 0)
            if count:
                parts.append(f"{key}={count}")
        if parts:
            print(f"    {label:>8}: {', '.join(parts)}")
        else:
            print(f"    {label:>8}: (no panels)")

    # Bbox area stats
    for label, totals in [("Baseline", total_b), ("New", total_n)]:
        areas = totals["bbox_areas"]
        if areas:
            mean_a = sum(areas) / len(areas)
            min_a = min(areas)
            max_a = max(areas)
            degenerate = sum(1 for a in areas if a < 0.01)
            oversized = sum(1 for a in areas if a > 0.95)
            print(f"\n  {label} bbox areas ({len(areas)} panels):")
            print(f"    Mean: {mean_a:.3f}  Min: {min_a:.3f}  Max: {max_a:.3f}")
            if degenerate:
                print(f"    Degenerate (<1% page): {degenerate}")
            if oversized:
                print(f"    Oversized (>95% page): {oversized}")

    # Crop success rate
    for label, totals in [("Baseline", total_b), ("New", total_n)]:
        with_bbox = totals["panels_with_bbox"]
        with_crop = totals["panels_with_crop"]
        if with_bbox > 0:
            rate = with_crop / with_bbox * 100
            print(f"\n  {label} crop success: {with_crop}/{with_bbox} ({rate:.1f}%)")

    # Bbox IOU comparison
    if page_ious:
        all_ious = [m for _, m, _, _ in page_ious]
        overall_mean = sum(all_ious) / len(all_ious)
        print(f"\n  Bbox IOU (baseline vs new):")
        print(f"    Pages compared:  {len(page_ious)}/{len(common)}")
        print(f"    Mean page IOU:   {overall_mean:.3f}")

        # Worst pages
        worst = sorted(page_ious, key=lambda x: x[1])[:5]
        if worst and worst[0][1] < 0.5:
            print(f"    Worst pages (biggest bbox shifts):")
            for pnum, miou, matched, unmatched in worst:
                if miou >= 0.5:
                    break
                extra = f" +{unmatched} unmatched" if unmatched else ""
                print(f"      Page {pnum:03d}: IOU={miou:.3f} ({matched} matched{extra})")

    # ---- Character comparison ----
    print("\n" + "=" * 90)
    print("CHARACTER IDS")
    print("=" * 90)
    missing = total_b["chars"] - total_n["chars"]
    added = total_n["chars"] - total_b["chars"]
    if missing:
        print(f"  MISSING from new (REGRESSION): {sorted(missing)}")
    else:
        print(f"  All {len(total_b['chars'])} baseline characters found in new")
    if added:
        print(f"  New characters not in baseline: {sorted(added)}")

    # ---- Manifest comparison ----
    b_manifest = baseline_dir / "chapter_manifest.json"
    n_manifest = new_dir / "chapter_manifest.json"

    if b_manifest.exists() and n_manifest.exists():
        bm = load_json(b_manifest)
        nm = load_json(n_manifest)

        print("\n" + "=" * 90)
        print("MANIFEST COMPARISON")
        print("=" * 90)

        b_di = len(bm.get("dialogue_index", []))
        n_di = len(nm.get("dialogue_index", []))
        print(f"  Dialogue index: baseline={b_di}, new={n_di}  ({pct_change(b_di, n_di)})")

        b_stats = bm.get("statistics", {})
        n_stats = nm.get("statistics", {})
        for key in ("total_panels", "total_dialog_lines", "total_narration_lines", "total_sfx", "unique_characters"):
            bv = b_stats.get(key, "?")
            nv = n_stats.get(key, "?")
            if isinstance(bv, (int, float)) and isinstance(nv, (int, float)):
                print(f"  {key}: baseline={bv}, new={nv}  ({pct_change(bv, nv)})")
            else:
                print(f"  {key}: baseline={bv}, new={nv}")

        # New fields check
        print("\n  New fields present:")
        print(f"    arc_id:       {'YES' if nm.get('arc_id') else 'MISSING'} ({nm.get('arc_id', '-')})")
        print(f"    locations:    {'YES' if nm.get('locations') else 'MISSING'} ({len(nm.get('locations', []))} items)")
        print(f"    saga:         {'YES' if nm.get('saga') else 'MISSING'} ({nm.get('saga', '-')})")
        print(f"    wiki_grounded: {'YES' if nm.get('processing_metadata', {}).get('wiki_grounded') is not None else 'MISSING'}")

        # Locations overlap
        b_locs = set(bm.get("locations", []))
        n_locs = set(nm.get("locations", []))
        if b_locs and n_locs:
            jaccard = len(b_locs & n_locs) / len(b_locs | n_locs) if (b_locs | n_locs) else 0
            print(f"\n  Locations Jaccard overlap: {jaccard:.1%}")
            print(f"    Baseline only: {sorted(b_locs - n_locs)[:5]}")
            print(f"    New only:      {sorted(n_locs - b_locs)[:5]}")

    # ---- Character registry comparison ----
    b_reg = baseline_dir / "character_registry.json"
    n_reg = new_dir / "character_registry.json"
    if b_reg.exists() and n_reg.exists():
        br = load_json(b_reg)
        nr = load_json(n_reg)
        b_chars = set(br.get("characters", {}).keys())
        n_chars = set(nr.get("characters", {}).keys())
        missing_reg = b_chars - n_chars
        if missing_reg:
            print(f"\n  REGISTRY: Characters lost: {sorted(missing_reg)}")

    # ---- Flagged pages ----
    if page_flags:
        print("\n" + "=" * 90)
        print(f"FLAGGED PAGES ({len(page_flags)} pages with notable differences)")
        print("=" * 90)
        for pnum, flags in page_flags:
            print(f"  Page {pnum:03d}: {', '.join(flags)}")
    else:
        print("\n  No pages flagged (all within thresholds)")

    # ---- Verdict ----
    print("\n" + "=" * 90)
    issues = []
    if missing:
        issues.append(f"LOST {len(missing)} characters")
    if total_b["text"] > 0 and abs(total_n["text"] - total_b["text"]) / total_b["text"] > 0.10:
        issues.append(f"text element count changed by {pct_change(total_b['text'], total_n['text'])}")
    if n_rate - b_rate < -5:
        issues.append(f"speaker attribution dropped {n_rate - b_rate:.1f}pp")
    if total_n["has_bbox"] == 0 and len(common) > 0:
        issues.append("no bounding boxes found (Phase 4 may not be working)")

    # Bbox-specific verdict checks
    if total_b["panels_with_crop"] > 0 and total_n["panels_with_crop"] > 0:
        b_crop_rate = total_b["panels_with_crop"] / max(total_b["panels_with_bbox"], 1)
        n_crop_rate = total_n["panels_with_crop"] / max(total_n["panels_with_bbox"], 1)
        if n_crop_rate < b_crop_rate - 0.10:
            issues.append(f"crop success rate dropped ({n_crop_rate:.0%} vs baseline {b_crop_rate:.0%})")
    if page_ious:
        overall_mean = sum(m for _, m, _, _ in page_ious) / len(page_ious)
        if overall_mean < 0.3:
            issues.append(f"mean bbox IOU={overall_mean:.2f} -- major bbox regression")

    n_degenerate = sum(1 for a in total_n["bbox_areas"] if a < 0.01)
    if n_degenerate > 3:
        issues.append(f"{n_degenerate} degenerate bboxes (<1% page area)")

    if issues:
        print("VERDICT: REVIEW NEEDED")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("VERDICT: LOOKS GOOD")
        print("  Structural metrics within acceptable thresholds.")
        print("  Recommend manual spot-check of 3 pages before accepting.")
    print("=" * 90)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
