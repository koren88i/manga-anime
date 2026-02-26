"""Run Stage 3 (per-shot feature extraction) and inspect results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import glob
import json
import time

from extraction.scene_detector import extract_shot_features

video = glob.glob("anime/*0001*.mkv")[0]

# Load shots from stage 2
with open("output/anime_metadata/0001/shots_raw.json") as f:
    shots = json.load(f)
print(f"Loaded {len(shots)} shots")

# Load subtitles from stage 1
with open("output/anime_metadata/0001/subtitles_raw.json") as f:
    subs = json.load(f)
print(f"Loaded {len(subs)} subtitle segments")

t0 = time.time()
shots = extract_shot_features(video, shots, whisper_segments=subs)
elapsed = time.time() - t0

print(f"\n=== FEATURE EXTRACTION RESULTS ===")
print(f"Time: {elapsed:.1f}s")

# Stats
motion_vals = [s.get("motion_intensity", 0) for s in shots]
dialogue_shots = sum(1 for s in shots if s.get("audio_features", {}).get("has_dialogue"))
flashback_shots = sum(1 for s in shots if s.get("_flashback_candidate"))

print(f"Shots with dialogue: {dialogue_shots}/{len(shots)}")
print(f"Flashback candidates: {flashback_shots}")
print(f"Motion intensity: min={min(motion_vals):.3f}, max={max(motion_vals):.3f}, avg={sum(motion_vals)/len(motion_vals):.3f}")

# Show a few examples
print(f"\n=== SAMPLE SHOTS (high motion) ===")
high_motion = sorted(shots, key=lambda s: s.get("motion_intensity", 0), reverse=True)[:5]
for s in high_motion:
    m1, s1 = divmod(s["start_time"], 60)
    print(f"  Shot {s['shot_id']:3d} ({int(m1)}:{s1:05.2f}): motion={s['motion_intensity']:.3f}  "
          f"dialogue={s['audio_features']['has_dialogue']}  flashback={s.get('_flashback_candidate', False)}")

print(f"\n=== SAMPLE SHOTS (with dialogue) ===")
with_dialogue = [s for s in shots if s.get("audio_features", {}).get("has_dialogue")][:5]
for s in with_dialogue:
    m1, s1 = divmod(s["start_time"], 60)
    print(f"  Shot {s['shot_id']:3d} ({int(m1)}:{s1:05.2f}): motion={s['motion_intensity']:.3f}  "
          f"density={s['audio_features']['dialogue_density']:.2f}")

# Save
with open("output/anime_metadata/0001/shots_features.json", "w") as f:
    json.dump(shots, f, indent=2)
print(f"\nSaved to output/anime_metadata/0001/shots_features.json")
