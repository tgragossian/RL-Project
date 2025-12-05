"""Detailed test of camp inference."""

import json
import sys
sys.path.append('src')

from pathlib import Path
from camp_inference import infer_camp_clear, identify_camp_by_position

# Load the timeline
timeline_file = Path("data/raw/timeline_NA1_5089218526.json")
with open(timeline_file) as f:
    timeline = json.load(f)

frames = timeline['info']['frames']

# Let's manually check participant 2 (blue jungler)
participant_id = 2

print(f"=== DETAILED ANALYSIS: Participant {participant_id} ===\n")

for i in range(1, min(10, len(frames))):
    prev_frame = frames[i-1]
    curr_frame = frames[i]

    prev_data = prev_frame['participantFrames'][str(participant_id)]
    curr_data = curr_frame['participantFrames'][str(participant_id)]

    gold_delta = curr_data['totalGold'] - prev_data['totalGold']
    jg_delta = curr_data['jungleMinionsKilled'] - prev_data['jungleMinionsKilled']

    if jg_delta > 0:
        pos = (curr_data['position']['x'], curr_data['position']['y'])
        camp = identify_camp_by_position(pos, max_distance=3000)

        print(f"Frame {i-1} → {i} (t={prev_frame['timestamp']/1000:.0f}s → {curr_frame['timestamp']/1000:.0f}s)")
        print(f"  Jungle minions: {prev_data['jungleMinionsKilled']} → {curr_data['jungleMinionsKilled']} (Δ={jg_delta})")
        print(f"  Gold: {prev_data['totalGold']} → {curr_data['totalGold']} (Δ={gold_delta}g)")
        print(f"  Position: {pos}")
        print(f"  Nearest camp: {camp}")

        # Try inference
        camp_clear = infer_camp_clear(prev_frame, curr_frame, participant_id)
        if camp_clear:
            print(f"  ✓ Inferred: {camp_clear.camp_type} at {camp_clear.camp_name} (conf={camp_clear.confidence:.2f})")
        else:
            print(f"  ✗ Inference failed")
        print()
