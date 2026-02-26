"""
Pass 2 — Bounded Gap Filling and Delta Extraction

For each gap between consecutive dialogue anchors, sends a bounded sub-problem
to GPT-4o. The LLM produces panel-to-shot mappings, transformation deltas,
transformation analysis, and imperative delta prompts.

Also builds alignment records for dialogue-anchored pairs (Pass 1 anchors).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import logging
import re
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

API_COSTS = {
    "gpt4o_input_tokens": 0,
    "gpt4o_output_tokens": 0,
    "llm_calls": 0,
}

PRICING = {
    "gpt4o_input_per_1m": 2.50,
    "gpt4o_output_per_1m": 10.00,
}


def get_llm_cost_summary():
    """Return a cost summary string for LLM API calls."""
    input_cost = (API_COSTS["gpt4o_input_tokens"] / 1_000_000) * PRICING["gpt4o_input_per_1m"]
    output_cost = (API_COSTS["gpt4o_output_tokens"] / 1_000_000) * PRICING["gpt4o_output_per_1m"]
    total = input_cost + output_cost
    return (
        f"LLM API: {API_COSTS['llm_calls']} calls, "
        f"{API_COSTS['gpt4o_input_tokens']:,} in / "
        f"{API_COSTS['gpt4o_output_tokens']:,} out, "
        f"${total:.4f}"
    )


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "openai_model": "gpt-4o",
    "openai_max_tokens": 4096,
    "openai_temperature": 0.2,
    "vlm_max_retries": 3,
    "vlm_retry_delay": 5,
    "vlm_calls_per_minute": 100,
    "context_window_size": 3,
    "resume": False,
}


# ---------------------------------------------------------------------------
# Panel data helpers
# ---------------------------------------------------------------------------

def _get_panel_data(panel_id, manga_manifest):
    """Look up panel data from manga manifest page JSONs."""
    pages_dir = os.path.dirname(manga_manifest.get("_manifest_path", ""))

    for page_info in manga_manifest.get("pages", []):
        metadata_file = page_info.get("metadata_file")
        if not metadata_file or not pages_dir:
            continue
        page_path = os.path.join(pages_dir, metadata_file)
        if not os.path.exists(page_path):
            continue
        try:
            with open(page_path, "r", encoding="utf-8") as f:
                page_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        for panel in page_data.get("panels", []):
            if panel.get("panel_id") == panel_id:
                return panel

    return None


def _build_manga_context_window(panel_id, manga_manifest, window_size=3):
    """
    Get the N panels before and after the given panel_id for context.

    Returns:
        {"before": [...], "after": [...]} with panel summaries.
    """
    # Build flat panel list from dialogue_index (ordered by seq)
    all_panels = []
    seen = set()
    for d in manga_manifest.get("dialogue_index", []):
        pid = d.get("panel_id")
        if pid and pid not in seen:
            seen.add(pid)
            all_panels.append({
                "panel_id": pid,
                "panel_image_path": d.get("panel_image_path"),
                "panel_narrative": None,  # filled from page data if available
                "characters_present": [],
                "page": d.get("page"),
                "location": d.get("location"),
            })

    # Find target index
    target_idx = None
    for i, p in enumerate(all_panels):
        if p["panel_id"] == panel_id:
            target_idx = i
            break

    if target_idx is None:
        return {"before": [], "after": []}

    before = all_panels[max(0, target_idx - window_size):target_idx]
    after = all_panels[target_idx + 1:target_idx + 1 + window_size]

    return {"before": before, "after": after}


def _get_shots_in_time_range(shot_index, start_time, end_time):
    """Get all shots that overlap with the given time range."""
    shots = []
    for shot in shot_index:
        shot_start = shot.get("start_time_seconds", 0)
        shot_end = shot.get("end_time_seconds", 0)
        if shot_end > start_time and shot_start < end_time:
            shots.append(shot)
    return shots


def _get_scene_data(scene_number, scene_data):
    """Look up scene data by scene number."""
    for scene in scene_data:
        if scene.get("scene_number") == scene_number:
            return scene
    return {}


# ---------------------------------------------------------------------------
# LLM prompt building
# ---------------------------------------------------------------------------

def _build_gap_prompt(gap, anchors, manga_manifest, anime_manifest,
                      shot_index, scene_data, config):
    """Build the LLM prompt for a gap between two anchors."""
    episode_id = anime_manifest.get("episode_id", "?")
    chapter_id = manga_manifest.get("chapter_id", "?")
    arc_id = manga_manifest.get("arc_id", "")

    # Find bounding anchors
    start_anchor = None
    end_anchor = None
    for a in anchors:
        if a["anchor_id"] == gap.get("start_anchor_id"):
            start_anchor = a
        if a["anchor_id"] == gap.get("end_anchor_id"):
            end_anchor = a

    lines = []
    lines.append("You are analyzing how a manga chapter was adapted into an anime episode.")
    lines.append(f"Episode: {episode_id}, Chapter: {chapter_id}, Arc: {arc_id}")
    lines.append("")

    # Bounding anchors
    lines.append("=== BOUNDING ANCHORS ===")
    if start_anchor:
        lines.append(f"Prior anchor: manga panel {start_anchor['manga_panel_id']} "
                      f"<-> anime shot {start_anchor['anime_shot_id']} "
                      f"at {start_anchor.get('anime_start_time', '?')}s")
        lines.append(f"  Manga: \"{start_anchor['manga_content']}\"")
        lines.append(f"  Anime: \"{start_anchor['anime_content']}\"")
    else:
        lines.append("Prior anchor: START OF CHAPTER/EPISODE")

    if end_anchor:
        lines.append(f"Next anchor: manga panel {end_anchor['manga_panel_id']} "
                      f"<-> anime shot {end_anchor['anime_shot_id']} "
                      f"at {end_anchor.get('anime_start_time', '?')}s")
        lines.append(f"  Manga: \"{end_anchor['manga_content']}\"")
        lines.append(f"  Anime: \"{end_anchor['anime_content']}\"")
    else:
        lines.append("Next anchor: END OF CHAPTER/EPISODE")
    lines.append("")

    # Manga panels in gap
    manga_by_seq = {d["seq"]: d for d in manga_manifest.get("dialogue_index", [])}
    lines.append("=== MANGA PANELS IN GAP (unanchored) ===")
    seen_panels = set()
    for seq in gap.get("manga_seqs_unanchored", []):
        d = manga_by_seq.get(seq, {})
        pid = d.get("panel_id", "?")
        if pid in seen_panels:
            # Just add dialogue line
            lines.append(f"    Dialogue: \"{d.get('content', '')}\" "
                          f"-- {d.get('speaker', '?')}")
            continue
        seen_panels.add(pid)
        lines.append(f"  Panel {pid} (page {d.get('page', '?')}):")
        lines.append(f"    Location: {d.get('location', '?')}, "
                      f"Time: {d.get('time_context', '?')}")
        lines.append(f"    Dialogue: \"{d.get('content', '')}\" "
                      f"-- {d.get('speaker', '?')}")

    if not gap.get("manga_seqs_unanchored"):
        lines.append("  (no unanchored manga panels in this gap)")
    lines.append("")

    # Anime shots in gap
    anime_by_seq = {d["seq"]: d for d in anime_manifest.get("dialogue_index", [])}
    shot_by_id = {s["shot_id"]: s for s in shot_index}

    lines.append("=== ANIME SHOTS IN GAP ===")
    seen_shots = set()
    for seq in gap.get("anime_seqs_unanchored", []):
        d = anime_by_seq.get(seq, {})
        shot_id = d.get("shot_id")
        if shot_id is not None and shot_id not in seen_shots:
            seen_shots.add(shot_id)
            shot = shot_by_id.get(shot_id, {})
            lines.append(
                f"  Shot {shot_id} (scene {shot.get('scene_number', '?')}, "
                f"{shot.get('start_time_seconds', '?')}s-"
                f"{shot.get('end_time_seconds', '?')}s, "
                f"{shot.get('duration_seconds', '?'):.1f}s):"
                if isinstance(shot.get('duration_seconds'), (int, float))
                else f"  Shot {shot_id} (scene {shot.get('scene_number', '?')}):"
            )
            lines.append(f"    Type: {shot.get('shot_type', '?')}, "
                          f"Content origin: {shot.get('content_origin', '?')}")
            lines.append(f"    Motion: {shot.get('motion_intensity', '?')}")

        # Add dialogue for this shot
        lines.append(f"    Dialogue: \"{d.get('content', '')}\" "
                      f"-- {d.get('speaker', '?')}")

    if not gap.get("anime_seqs_unanchored"):
        lines.append("  (no unanchored anime content in this gap)")
    lines.append("")

    # Task instructions
    lines.append("=== TASK ===")
    lines.append("For each manga panel in the gap, determine which anime shots "
                 "correspond to it.")
    lines.append("")
    lines.append("Return a JSON object with these fields:")
    lines.append("1. panel_to_shot_mapping: array of objects, each with:")
    lines.append("   - panel_id: the manga panel ID")
    lines.append("   - shot_ids: array of anime shot IDs matched to this panel")
    lines.append("   - confidence: 0.0-1.0 how confident this mapping is")
    lines.append("2. content_origin_corrections: array of objects with:")
    lines.append("   - shot_id: anime shot ID")
    lines.append("   - corrected_origin: one of 'manga_derived', 'reaction_shot', "
                 "'action_extension', 'establishing', 'transition', 'anime_original'")
    lines.append("3. transformation_delta: object with:")
    lines.append("   - duration_expansion_ratio: anime seconds / manga panel count")
    lines.append("   - shot_count: number of anime shots")
    lines.append("   - shot_type_sequence: ordered list of shot types")
    lines.append("   - emotional_tone_shift: 'preserved' | 'heightened' | "
                 "'softened' | 'changed'")
    lines.append("   - added_content: array of {shot_id, content_origin, description}")
    lines.append("   - skipped_panels: array of panel_ids with no anime counterpart")
    lines.append("4. transformation_analysis: a paragraph describing what the studio "
                 "did when adapting these manga panels into anime. Write in analytical "
                 "past tense. Describe creative decisions: what was expanded, compressed, "
                 "added, reinterpreted, and why.")
    lines.append("5. imperative_delta_prompt: a shooting script in imperative voice. "
                 "Reference the manga panels as source material. Describe what to do, "
                 "not what was done. Include specific timing, camera directions, "
                 "and tone.")
    lines.append("6. alignment_explanation: one sentence explaining your matching "
                 "rationale.")
    lines.append("")
    lines.append("Return ONLY valid JSON, no markdown fences.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(client, prompt, config):
    """Call GPT-4o with a text prompt and return parsed JSON."""
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(config.get("vlm_max_retries", 3)):
        try:
            response = client.chat.completions.create(
                model=config.get("openai_model", "gpt-4o"),
                messages=messages,
                max_tokens=config.get("openai_max_tokens", 4096),
                temperature=config.get("openai_temperature", 0.2),
            )
            # Track costs
            usage = response.usage
            if usage:
                API_COSTS["gpt4o_input_tokens"] += usage.prompt_tokens
                API_COSTS["gpt4o_output_tokens"] += usage.completion_tokens
            API_COSTS["llm_calls"] += 1

            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < config.get("vlm_max_retries", 3) - 1:
                time.sleep(config.get("vlm_retry_delay", 5))
        except Exception as e:
            logger.warning(f"LLM API error (attempt {attempt + 1}): {e}")
            if attempt < config.get("vlm_max_retries", 3) - 1:
                time.sleep(config.get("vlm_retry_delay", 5))

    return None


# ---------------------------------------------------------------------------
# Build alignment records from anchors (Pass 1 results)
# ---------------------------------------------------------------------------

def _build_anchor_record(anchor, manga_manifest, anime_manifest, shot_index,
                         record_idx, config):
    """Build an alignment record for a dialogue-anchored pair."""
    chapter_id = manga_manifest.get("chapter_id", "")
    episode_id = anime_manifest.get("episode_id", "")
    arc_id = manga_manifest.get("arc_id", "")

    panel_id = anchor.get("manga_panel_id")
    shot_id = anchor.get("anime_shot_id")

    # Look up shot data
    shot_data = None
    for s in shot_index:
        if s.get("shot_id") == shot_id:
            shot_data = s
            break

    # Panel image path from dialogue_index
    panel_image_path = None
    for d in manga_manifest.get("dialogue_index", []):
        if d.get("panel_id") == panel_id:
            panel_image_path = d.get("panel_image_path")
            break

    # Context window
    context_window = _build_manga_context_window(
        panel_id, manga_manifest,
        config.get("context_window_size", 3)
    )

    # Time range from shot
    time_range = [0.0, 0.0]
    if shot_data:
        time_range = [
            shot_data.get("start_time_seconds", 0.0),
            shot_data.get("end_time_seconds", 0.0),
        ]

    record = {
        "alignment_id": f"align_{record_idx:04d}",
        "arc_id": arc_id,
        "manga_side": {
            "chapter_id": chapter_id,
            "panel_ids": [panel_id] if panel_id else [],
            "page_range": [anchor.get("manga_page"), anchor.get("manga_page")],
            "panel_image_paths": [panel_image_path] if panel_image_path else [],
            "dialogue_line_ids": [anchor["manga_seq"]],
            "panel_descriptions": [],
            "manga_context_window": context_window,
        },
        "anime_side": {
            "episode_id": episode_id,
            "scene_numbers": [anchor.get("anime_scene_number")],
            "shot_ids": [shot_id] if shot_id else [],
            "time_range": time_range,
            "dialogue_line_ids": [anchor["anime_seq"]],
        },
        "anchor_type": "dialogue_anchored",
        "confidence": max(
            anchor.get("score_openai") or 0,
            anchor.get("score_bow") or 0,
        ),
        "confidence_tier": 1,
        "score_openai": anchor.get("score_openai"),
        "score_bow": anchor.get("score_bow"),
        "speaker_match": anchor.get("speaker_match", False),
        "location_match": anchor.get("location_match", False),
        "time_context_match": anchor.get("time_context_match", False),
        "transformation_delta": None,  # Filled by gap-fill LLM
        "anime_original_segments": [],
        "transformation_analysis": None,  # Filled by gap-fill LLM
        "imperative_delta_prompt": None,  # Filled by gap-fill LLM
        "alignment_explanation": "Dialogue-anchored via semantic embedding match.",
    }

    return record


# ---------------------------------------------------------------------------
# Build alignment records from gap-fill LLM responses
# ---------------------------------------------------------------------------

def _build_gap_records(gap, llm_result, manga_manifest, anime_manifest,
                       shot_index, start_record_idx):
    """Build alignment records from a gap-fill LLM response."""
    if not llm_result:
        return []

    chapter_id = manga_manifest.get("chapter_id", "")
    episode_id = anime_manifest.get("episode_id", "")
    arc_id = manga_manifest.get("arc_id", "")

    mappings = llm_result.get("panel_to_shot_mapping", [])
    delta = llm_result.get("transformation_delta", {})
    analysis = llm_result.get("transformation_analysis", "")
    imperative = llm_result.get("imperative_delta_prompt", "")
    explanation = llm_result.get("alignment_explanation", "")
    corrections = llm_result.get("content_origin_corrections", [])

    records = []
    shot_by_id = {s["shot_id"]: s for s in shot_index}

    for map_idx, mapping in enumerate(mappings):
        panel_id = mapping.get("panel_id", "")
        shot_ids = mapping.get("shot_ids", [])
        confidence = mapping.get("confidence", 0.5)

        # Determine page and image path
        page = None
        panel_image_path = None
        for d in manga_manifest.get("dialogue_index", []):
            if d.get("panel_id") == panel_id:
                page = d.get("page")
                panel_image_path = d.get("panel_image_path")
                break

        # Collect dialogue line IDs for this panel
        manga_dlg_ids = [
            d["seq"] for d in manga_manifest.get("dialogue_index", [])
            if d.get("panel_id") == panel_id
        ]

        # Collect anime dialogue for matched shots
        anime_dlg_ids = [
            d["seq"] for d in anime_manifest.get("dialogue_index", [])
            if d.get("shot_id") in shot_ids
        ]

        # Scene numbers
        scene_numbers = list(set(
            shot_by_id[sid].get("scene_number")
            for sid in shot_ids if sid in shot_by_id
        ))

        # Time range
        shot_times = [
            (shot_by_id[sid].get("start_time_seconds", 0),
             shot_by_id[sid].get("end_time_seconds", 0))
            for sid in shot_ids if sid in shot_by_id
        ]
        if shot_times:
            time_range = [
                min(t[0] for t in shot_times),
                max(t[1] for t in shot_times),
            ]
        else:
            time_range = [0.0, 0.0]

        # Confidence tier
        manga_count = 1  # single panel per mapping entry
        anime_count = len(shot_ids)
        if manga_count <= 3 and anime_count <= 5:
            tier = 2
        else:
            tier = 3

        record = {
            "alignment_id": f"align_{start_record_idx + map_idx:04d}",
            "arc_id": arc_id,
            "manga_side": {
                "chapter_id": chapter_id,
                "panel_ids": [panel_id],
                "page_range": [page, page],
                "panel_image_paths": [panel_image_path] if panel_image_path else [],
                "dialogue_line_ids": manga_dlg_ids,
                "panel_descriptions": [],
                "manga_context_window": {"before": [], "after": []},
            },
            "anime_side": {
                "episode_id": episode_id,
                "scene_numbers": scene_numbers,
                "shot_ids": shot_ids,
                "time_range": time_range,
                "dialogue_line_ids": anime_dlg_ids,
            },
            "anchor_type": "interpolated",
            "confidence": confidence,
            "confidence_tier": tier,
            "score_openai": None,
            "score_bow": None,
            "speaker_match": None,
            "location_match": None,
            "time_context_match": None,
            "transformation_delta": delta,
            "anime_original_segments": [],
            "transformation_analysis": analysis,
            "imperative_delta_prompt": imperative,
            "alignment_explanation": explanation,
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pass2(anchors, gaps, manga_manifest, anime_manifest,
              shot_index, scene_data, config=None, output_dir=None):
    """
    For each gap between anchors, call GPT-4o to produce panel-to-shot
    mappings, transformation deltas, and imperative delta prompts.

    Also builds alignment records for dialogue-anchored pairs.

    Args:
        anchors: list of anchor dicts from Pass 1
        gaps: list of gap dicts from Pass 1
        manga_manifest: chapter manifest dict
        anime_manifest: episode manifest dict
        shot_index: list of shot dicts from shot_index.json
        scene_data: list of scene dicts
        config: optional config overrides
        output_dir: directory for caching LLM responses

    Returns:
        alignment_records: list of alignment record dicts
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    print(f"[Pass 2] Building alignment records...")
    print(f"[Pass 2] {len(anchors)} anchors + {len(gaps)} gaps to process")

    # Initialize OpenAI client
    api_key = cfg.get("openai_api_key", "")
    client = None
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("openai package not installed -- Pass 2 gap filling disabled")

    # LLM cache directory
    cache_dir = None
    if output_dir:
        cache_dir = os.path.join(output_dir, "llm_cache")
        os.makedirs(cache_dir, exist_ok=True)

    records = []

    # Build records for dialogue-anchored pairs
    print(f"[Pass 2] Building {len(anchors)} anchor records...")
    for idx, anchor in enumerate(anchors):
        record = _build_anchor_record(
            anchor, manga_manifest, anime_manifest, shot_index,
            len(records), cfg
        )
        records.append(record)

    # Process each gap
    if not gaps:
        print("[Pass 2] No gaps to fill.")
    elif not client:
        print("[Pass 2] No OpenAI client -- skipping gap filling. "
              "Anchor records only.")
    else:
        # Rate limiting
        last_call_time = 0
        min_interval = 60.0 / cfg.get("vlm_calls_per_minute", 100)

        for gap_idx, gap in enumerate(gaps):
            manga_count = len(gap.get("manga_seqs_unanchored", []))
            anime_count = len(gap.get("anime_seqs_unanchored", []))

            # Skip gaps with no manga content (pure anime-original)
            if manga_count == 0:
                continue

            print(f"[Pass 2] Gap {gap_idx + 1}/{len(gaps)}: "
                  f"{manga_count} manga lines, {anime_count} anime lines")

            # Check cache
            cache_path = None
            if cache_dir:
                cache_path = os.path.join(
                    cache_dir, f"gap_{gap['gap_id']:03d}.json"
                )
                if cfg.get("resume") and os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            llm_result = json.load(f)
                        print(f"  (cached)")
                        gap_records = _build_gap_records(
                            gap, llm_result, manga_manifest, anime_manifest,
                            shot_index, len(records)
                        )
                        records.extend(gap_records)
                        continue
                    except (json.JSONDecodeError, IOError):
                        pass  # re-run

            # Build prompt
            prompt = _build_gap_prompt(
                gap, anchors, manga_manifest, anime_manifest,
                shot_index, scene_data, cfg
            )

            # Rate limit
            elapsed = time.time() - last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            # Call LLM
            llm_result = _call_llm(client, prompt, cfg)
            last_call_time = time.time()

            # Cache result
            if llm_result and cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(llm_result, f, indent=2, ensure_ascii=False)

            # Build records from LLM result
            if llm_result:
                gap_records = _build_gap_records(
                    gap, llm_result, manga_manifest, anime_manifest,
                    shot_index, len(records)
                )
                records.extend(gap_records)
                print(f"  -> {len(gap_records)} records from gap")
            else:
                print(f"  -> LLM failed, gap skipped")

        print(f"\n[Pass 2] {get_llm_cost_summary()}")

    print(f"[Pass 2] Total alignment records: {len(records)}")
    return records
