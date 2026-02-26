"""Run Stage 5 (VLM scene analysis with GPT-4o) using cached data from stages 1-4."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import glob
import json
import time

# Set API key from environment
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY environment variable"

from extraction.anime_metadata import (
    CONFIG, run_vlm_scene_analysis, detect_recaps,
    save_json, load_json, _stage_start, _stage_end, get_cost_summary
)

CONFIG["openai_api_key"] = os.environ["OPENAI_API_KEY"]

video = glob.glob("anime/*0001*.mkv")[0]
output_dir = "output/anime_metadata/0001"

# Load cached data from stages 1-4
with open(f"{output_dir}/subtitles_raw.json") as f:
    subs = json.load(f)
print(f"Loaded {len(subs)} subtitle segments")

with open(f"{output_dir}/shots_features.json") as f:
    shots = json.load(f)
print(f"Loaded {len(shots)} shots with features")

with open(f"{output_dir}/scenes_raw.json") as f:
    scenes = json.load(f)
print(f"Loaded {len(scenes)} scenes")

# Tag flashback scenes (same as pipeline does before stage 5)
for scene in scenes:
    if scene.get("has_flashback_candidate") and scene.get("scene_type", "story") == "story":
        scene["scene_type"] = "flashback"

# Recap detection (no previous episode manifest)
scenes = detect_recaps(scenes, subs)

print(f"\nScene types before VLM:")
from collections import Counter
types = Counter(s.get("scene_type", "story") for s in scenes)
for t, c in types.most_common():
    print(f"  {t}: {c}")

# Run Stage 5: VLM scene analysis
t0 = time.time()
scenes = run_vlm_scene_analysis(
    video, shots, scenes, subs,
    chapter_manifest=None,  # No manga manifest for now
    output_dir=output_dir
)
elapsed = time.time() - t0

print(f"\n=== VLM ANALYSIS COMPLETE ===")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Scenes analyzed: {len(scenes)}")

# Stats
locations = Counter(s.get("location", "unknown") for s in scenes)
emotions = Counter(s.get("emotional_tone", "?") for s in scenes)
scene_types = Counter(s.get("scene_type", "story") for s in scenes)

print(f"\nScene types:")
for t, c in scene_types.most_common():
    print(f"  {t}: {c}")

print(f"\nTop locations:")
for loc, c in locations.most_common(10):
    print(f"  {loc}: {c}")

print(f"\nEmotional tones:")
for tone, c in emotions.most_common():
    print(f"  {tone}: {c}")

# Characters found
all_chars = set()
for s in scenes:
    for ch in s.get("characters_present", []):
        cid = ch.get("character_id") or ch.get("name", "?")
        all_chars.add(cid)
print(f"\nCharacters found: {sorted(all_chars)}")

# Save updated scenes
save_json(scenes, f"{output_dir}/scenes_vlm.json")
print(f"\nSaved to {output_dir}/scenes_vlm.json")
print(get_cost_summary())
