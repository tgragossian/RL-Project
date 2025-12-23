"""
Visualize map zones with actual jungle path data.

Shows how zone detection classifies real jungle positions from collected matches.
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent / "src"))
from map_zones import get_zone, zone_to_string, MapZone


def analyze_jungle_positions():
    """Analyze zone distribution of jungle path positions."""

    jungle_data_path = Path(__file__).parent.parent / "data" / "processed" / "challenger_jungle_data.json"

    if not jungle_data_path.exists():
        print(f"❌ Jungle data not found: {jungle_data_path}")
        return

    with open(jungle_data_path, 'r') as f:
        jungle_data = json.load(f)

    print("="*70)
    print("JUNGLE POSITION ZONE ANALYSIS")
    print("="*70)
    print(f"Analyzing positions from {jungle_data['total_matches']} matches\n")

    # Collect all positions with their camp names
    position_data = []
    zone_counter = Counter()
    camp_zone_mapping = {}

    for player in jungle_data['players']:
        for match in player['matches']:
            for action in match['jungle_path']:
                x, y = action['position']
                camp_name = action['camp_name']
                confidence = action['confidence']

                zone = get_zone(x, y)
                zone_counter[zone] += 1

                # Track camp → zone mapping
                if camp_name not in camp_zone_mapping:
                    camp_zone_mapping[camp_name] = Counter()
                camp_zone_mapping[camp_name][zone] += 1

                position_data.append({
                    'position': (x, y),
                    'camp_name': camp_name,
                    'zone': zone,
                    'confidence': confidence
                })

    # Print zone distribution
    print("Zone Distribution:")
    print("-" * 70)
    total_positions = len(position_data)

    for zone, count in zone_counter.most_common():
        if zone != MapZone.UNKNOWN:
            pct = count / total_positions * 100
            print(f"  {zone_to_string(zone):25s}: {count:4d} positions ({pct:5.1f}%)")

    if MapZone.UNKNOWN in zone_counter:
        unknown_count = zone_counter[MapZone.UNKNOWN]
        pct = unknown_count / total_positions * 100
        print(f"  {'Unknown':25s}: {unknown_count:4d} positions ({pct:5.1f}%)")

    # Print camp → zone mapping
    print("\n" + "="*70)
    print("Camp → Zone Mapping:")
    print("="*70 + "\n")

    for camp_name in sorted(camp_zone_mapping.keys()):
        zones = camp_zone_mapping[camp_name]
        total_camp = sum(zones.values())

        print(f"{camp_name}:")
        for zone, count in zones.most_common(3):  # Top 3 zones per camp
            pct = count / total_camp * 100
            print(f"  → {zone_to_string(zone):25s}: {count:3d} ({pct:5.1f}%)")
        print()

    # Sample positions for each action type
    print("="*70)
    print("Sample Positions by Action Type:")
    print("="*70 + "\n")

    action_samples = {}
    for item in position_data:
        camp = item['camp_name']
        if camp not in action_samples:
            action_samples[camp] = []
        if len(action_samples[camp]) < 3:
            action_samples[camp].append(item)

    for camp_name in sorted(action_samples.keys()):
        samples = action_samples[camp_name]
        print(f"{camp_name}:")
        for sample in samples:
            x, y = sample['position']
            zone = sample['zone']
            print(f"  ({x:5.0f}, {y:5.0f}) → {zone_to_string(zone)}")
        print()


if __name__ == "__main__":
    analyze_jungle_positions()
