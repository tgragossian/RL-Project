"""Quick debug script to see what's in the timeline data."""

import json
import sys
sys.path.append('src')

from pathlib import Path

# Load the timeline
timeline_file = Path("data/raw/timeline_NA1_5089218526.json")
with open(timeline_file) as f:
    timeline = json.load(f)

frames = timeline['info']['frames']

print(f"Total frames: {len(frames)}")
print()

# Check a few participants to see jungle minions
for participant_id in range(1, 11):
    print(f"\n=== Participant {participant_id} ===")

    total_jg_minions = 0
    for i, frame in enumerate(frames[:10]):  # First 10 frames (10 minutes)
        pframe = frame['participantFrames'].get(str(participant_id))
        if pframe:
            jg_minions = pframe['jungleMinionsKilled']
            total_gold = pframe['totalGold']
            pos = pframe['position']

            if i == 0 or jg_minions > total_jg_minions:
                print(f"  Frame {i} (t={frame['timestamp']/1000:.0f}s): "
                      f"jg_minions={jg_minions}, gold={total_gold}, "
                      f"pos=({pos['x']}, {pos['y']})")
                total_jg_minions = jg_minions

    print(f"  Total jungle minions killed in first 10min: {total_jg_minions}")
