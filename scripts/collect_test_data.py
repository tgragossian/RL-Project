"""
Quick test data collection - collects just 50 matches for testing.

Use this to verify the pipeline works before doing a full 2000+ match collection.
"""

import sys
from pathlib import Path

# Use the same collection logic but with smaller targets
sys.path.append(str(Path(__file__).parent))

# Import from the main collection script
from collect_from_leaderboard import collect_from_region
import json
import time


def main():
    print("="*70)
    print("TEST DATA COLLECTION (50 matches)")
    print("This is a quick test - use collect_from_leaderboard.py for full collection")
    print("="*70)

    # Collect just 50 matches from NA (faster than KR usually)
    print("\nCollecting 50 matches from NA...")
    all_data = []

    all_data.extend(collect_from_region("na1", target_jungle_matches=50, matches_per_player=10))

    # Save data
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "test_jungle_data.json"

    with open(output_file, 'w') as f:
        json.dump({
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_players': len(all_data),
            'total_matches': sum(len(p['matches']) for p in all_data),
            'players': all_data
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ TEST COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Players: {len(all_data)}")
    print(f"Matches: {sum(len(p['matches']) for p in all_data)}")
    print(f"Saved to: {output_file}")
    print(f"\nIf this worked, you can now run:")
    print(f"  python scripts/collect_from_leaderboard.py")
    print(f"to collect the full dataset (2000+ matches)")


if __name__ == "__main__":
    main()
