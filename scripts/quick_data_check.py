"""
Quick data quality check for collected jungle paths.

Analyzes:
1. How many games collected
2. Average actions per game
3. Position sanity checks (teleporting detection)
4. Camp distribution
"""

import json
import numpy as np
from pathlib import Path
import math


def distance(pos1, pos2):
    """Calculate distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def analyze_jungle_data(data_file: Path):
    """Analyze collected jungle data for quality issues."""

    with open(data_file) as f:
        data = json.load(f)

    print("="*70)
    print("JUNGLE DATA QUALITY ANALYSIS")
    print("="*70)

    # Basic stats
    total_players = len(data['players'])
    total_matches = sum(len(p['matches']) for p in data['players'])

    print(f"\nCollection Summary:")
    print(f"  Players: {total_players}")
    print(f"  Matches: {total_matches}")
    print(f"  Collection date: {data.get('collection_date', 'Unknown')}")

    # Analyze each match
    all_paths = []
    suspicious_paths = []
    camp_counts = {}

    for player in data['players']:
        for match in player['matches']:
            path = match['jungle_path']
            all_paths.append(path)

            # Check for teleporting (large distances between consecutive actions)
            is_suspicious = False
            for i in range(len(path) - 1):
                curr_pos = path[i]['position']
                next_pos = path[i + 1]['position']
                dist = distance(curr_pos, next_pos)
                time_delta = path[i + 1]['timestamp'] - path[i]['timestamp']

                # If they moved >6000 units in <30 seconds, likely teleporting
                if dist > 6000 and time_delta < 30:
                    is_suspicious = True
                    break

            if is_suspicious:
                suspicious_paths.append({
                    'match_id': match['match_id'],
                    'path': path
                })

            # Count camp types
            for action in path:
                camp_name = action['camp_name']
                camp_counts[camp_name] = camp_counts.get(camp_name, 0) + 1

    # Path length statistics
    path_lengths = [len(p) for p in all_paths]
    avg_length = np.mean(path_lengths)
    min_length = np.min(path_lengths)
    max_length = np.max(path_lengths)

    print(f"\nPath Statistics:")
    print(f"  Average actions per game: {avg_length:.1f}")
    print(f"  Min actions: {min_length}")
    print(f"  Max actions: {max_length}")

    # Suspicious paths
    print(f"\nQuality Check:")
    print(f"  Suspicious paths (teleporting): {len(suspicious_paths)}/{len(all_paths)} ({len(suspicious_paths)/len(all_paths)*100:.1f}%)")

    if suspicious_paths:
        print(f"\n  Example suspicious path:")
        example = suspicious_paths[0]
        print(f"    Match: {example['match_id']}")
        for i, action in enumerate(example['path'][:5]):
            print(f"      {i+1}. {action['timestamp']:.0f}s - {action['camp_name']} at {action['position']}")

    # Camp distribution
    print(f"\nTop 10 Most Common Actions:")
    sorted_camps = sorted(camp_counts.items(), key=lambda x: -x[1])
    for camp_name, count in sorted_camps[:10]:
        pct = count / sum(camp_counts.values()) * 100
        print(f"  {camp_name:<20}: {count:>4} ({pct:>5.1f}%)")

    print("\n" + "="*70)

    # Return summary
    return {
        'total_matches': total_matches,
        'avg_path_length': avg_length,
        'suspicious_rate': len(suspicious_paths) / len(all_paths),
        'camp_distribution': camp_counts
    }


def main():
    data_file = Path(__file__).parent.parent / "data" / "processed" / "challenger_jungle_data.json"

    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("Run collect_from_leaderboard.py first")
        return

    summary = analyze_jungle_data(data_file)

    # Quality warnings
    print("\nQuality Assessment:")
    if summary['suspicious_rate'] > 0.3:
        print("  ⚠️  WARNING: >30% of paths show teleporting behavior")
        print("     Consider re-collecting with improved detection (v5)")
    elif summary['suspicious_rate'] > 0.1:
        print("  ⚠️  CAUTION: 10-30% of paths may have issues")
        print("     Use visualization tool to manually clean data")
    else:
        print("  ✅ Data quality looks good (<10% suspicious paths)")

    if summary['avg_path_length'] < 10:
        print("  ⚠️  WARNING: Average path length is very short")
        print("     May indicate detection issues")
    elif summary['avg_path_length'] > 50:
        print("  ⚠️  WARNING: Average path length is very long")
        print("     May indicate false positives in detection")
    else:
        print(f"  ✅ Path length looks reasonable (~{summary['avg_path_length']:.0f} actions/game)")

    print("\nNext Steps:")
    print("  1. Run: python scripts/visualize_and_correct_data.py")
    print("     - Visually inspect camp positions")
    print("     - Manually fix obvious errors with lasso tool")
    print("  2. Run: python src/training_data.py")
    print("     - Process data into training format")
    print("  3. Run: python scripts/train_behavior_cloning.py")
    print("     - Train model and check real accuracy")


if __name__ == "__main__":
    main()
