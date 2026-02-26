"""Run Stage 4 (scene detection + eyecatch) and inspect results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
from collections import Counter

from extraction.scene_detector import detect_scenes, detect_eyecatch

# Load featured shots from stage 3
with open("output/anime_metadata/0001/shots_features.json") as f:
    shots = json.load(f)
print(f"Loaded {len(shots)} shots with features")

# Eyecatch detection
t0 = time.time()
eyecatch_ids = detect_eyecatch(shots)
print(f"Eyecatch shot IDs: {eyecatch_ids}")

# Mark eyecatch
for shot in shots:
    shot["_is_eyecatch"] = shot["shot_id"] in eyecatch_ids

# Scene detection
scenes = detect_scenes(shots)
elapsed = time.time() - t0

# Tag eyecatch scenes
for scene in scenes:
    eyecatch_in = [sid for sid in scene["shot_ids"] if sid in eyecatch_ids]
    scene["_is_eyecatch"] = len(eyecatch_in) > 0

print(f"\n=== SCENE DETECTION RESULTS ===")
print(f"Time: {elapsed:.1f}s")
print(f"Total scenes: {len(scenes)}")
eyecatch_scenes = sum(1 for s in scenes if s.get("_is_eyecatch"))
flashback_scenes = sum(1 for s in scenes if s.get("has_flashback_candidate"))
print(f"Eyecatch scenes: {eyecatch_scenes}")
print(f"Flashback candidate scenes: {flashback_scenes}")

# Duration stats
durs = [s["duration"] for s in scenes]
print(f"Duration: min={min(durs):.1f}s, max={max(durs):.1f}s, avg={sum(durs)/len(durs):.1f}s")

# Show all scenes
print(f"\n=== ALL SCENES ===")
for s in scenes:
    m1, s1 = divmod(s["start_time"], 60)
    m2, s2 = divmod(s["end_time"], 60)
    tags = []
    if s.get("_is_eyecatch"):
        tags.append("EYECATCH")
    if s.get("has_flashback_candidate"):
        tags.append("FLASHBACK?")
    tag_str = f"  [{', '.join(tags)}]" if tags else ""
    print(f"  Scene {s['scene_number']:2d}: {int(m1)}:{s1:05.2f} - {int(m2)}:{s2:05.2f}  "
          f"({s['duration']:6.1f}s, {s['shot_count']:2d} shots){tag_str}")

# Save
with open("output/anime_metadata/0001/scenes_raw.json", "w") as f:
    json.dump(scenes, f, indent=2)
print(f"\nSaved to output/anime_metadata/0001/scenes_raw.json")
