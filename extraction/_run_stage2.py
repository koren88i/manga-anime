"""Run Stage 2 (shot detection) standalone and inspect results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import glob
import json
import time
from collections import Counter

from extraction.scene_detector import detect_shots

video = glob.glob("anime/*0001*.mkv")[0]
print(f"Video: {video}")

t0 = time.time()
shots = detect_shots(video)
elapsed = time.time() - t0

print(f"\n=== SHOT DETECTION RESULTS ===")
print(f"Total shots: {len(shots)}")
print(f"Time: {elapsed:.1f}s")

# Fade stats
fades = [s["fade_type"] for s in shots if s["fade_type"] != "none"]
print(f"Fades: {len(fades)} ({dict(Counter(fades))})")

# Duration stats
durs = [s["duration"] for s in shots]
print(f"Duration: min={min(durs):.2f}s, max={max(durs):.2f}s, avg={sum(durs)/len(durs):.2f}s")

# Show first 15 shots
print(f"\n=== FIRST 15 SHOTS ===")
for s in shots[:15]:
    m1, s1 = divmod(s["start_time"], 60)
    m2, s2 = divmod(s["end_time"], 60)
    print(f"  Shot {s['shot_id']:3d}: {int(m1)}:{s1:05.2f} - {int(m2)}:{s2:05.2f}  ({s['duration']:.2f}s)  fade={s['fade_type']}")

# Show shots around the midpoint (eyecatch area)
total_dur = shots[-1]["end_time"]
mid = total_dur / 2
print(f"\n=== SHOTS NEAR MIDPOINT ({mid:.0f}s) ===")
for s in shots:
    mid_shot = (s["start_time"] + s["end_time"]) / 2
    if abs(mid_shot - mid) < 30:
        m1, s1 = divmod(s["start_time"], 60)
        m2, s2 = divmod(s["end_time"], 60)
        print(f"  Shot {s['shot_id']:3d}: {int(m1)}:{s1:05.2f} - {int(m2)}:{s2:05.2f}  ({s['duration']:.2f}s)  fade={s['fade_type']}")

# Save
with open("output/anime_metadata/0001/shots_raw.json", "w") as f:
    json.dump(shots, f, indent=2)
print(f"\nSaved to output/anime_metadata/0001/shots_raw.json")
