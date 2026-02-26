"""Run Stages 6-9 using cached data from stages 1-5."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import glob
import json
import time

# Set API key from environment
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY environment variable"

from extraction.anime_metadata import (
    CONFIG, classify_content_origins, build_dialogue_index,
    build_scene_jsons, build_shot_index, build_character_registry,
    build_episode_manifest, save_json, load_json,
    _stage_start, _stage_end, get_cost_summary
)

CONFIG["openai_api_key"] = os.environ["OPENAI_API_KEY"]

output_dir = "output/anime_metadata/0001"
episode_id = "0001"

# Load cached data
with open(f"{output_dir}/subtitles_raw.json") as f:
    subs = json.load(f)
print(f"Loaded {len(subs)} subtitle segments")

with open(f"{output_dir}/shots_features.json") as f:
    shots = json.load(f)
print(f"Loaded {len(shots)} shots")

with open(f"{output_dir}/scenes_vlm.json") as f:
    scenes = json.load(f)
print(f"Loaded {len(scenes)} scenes (with VLM data)")

processing_start = time.time()
stage_times = {}

# --- Stage 6: Content origin classification ---
t = _stage_start(6)
shots = classify_content_origins(
    shots, scenes, subs,
    chapter_manifest=None,
    output_dir=output_dir
)
stage_times[6] = _stage_end(6, t)

# --- Stage 7: Dialogue index ---
t = _stage_start(7)
dialogue_index = build_dialogue_index(shots, scenes, subs, output_dir)
stage_times[7] = _stage_end(7, t)

# Save dialogue index
save_json(dialogue_index, f"{output_dir}/dialogue_index.json")

# --- Stage 8: Output files ---
t = _stage_start(8)
build_scene_jsons(shots, scenes, dialogue_index, output_dir)
build_shot_index(shots, scenes, output_dir)
character_registry = build_character_registry(
    scenes, shots, dialogue_index, episode_id, output_dir
)
stage_times[8] = _stage_end(8, t)

# --- Stage 9: Episode manifest ---
t = _stage_start(9)
manifest = build_episode_manifest(
    episode_id,
    "I'm Luffy! The Man Who's Gonna Be King of the Pirates!",
    scenes, shots, dialogue_index,
    processing_start, output_dir
)
stage_times[9] = _stage_end(9, t)

# --- Summary ---
total_elapsed = time.time() - processing_start
print(f"\n{'=' * 70}")
print(f"  STAGES 6-9 COMPLETE")
print(f"{'=' * 70}")
for stage_num in sorted(stage_times):
    from extraction.anime_metadata import STAGE_NAMES
    name = STAGE_NAMES.get(stage_num, "")
    elapsed = stage_times[stage_num]
    print(f"  [{stage_num}/9] {name:<42} {elapsed:.1f}s")
print(f"  Total: {total_elapsed:.1f}s")
print(f"{'=' * 70}")
