"""
Minimal test collection with detailed progress output.
"""

import sys
from pathlib import Path
import json
import time

sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent))

from collect_from_leaderboard import (
    get_challenger_league,
    get_match_ids,
    get_match_info,
    get_match_timeline,
    is_valid_patch,
    extract_full_jungle_path,
    REGION_ROUTING
)

def main():
    print("="*70)
    print("MINIMAL TEST COLLECTION - WITH PROGRESS")
    print("="*70)
    print("\nThis will collect just 5 jungle matches for testing\n")

    platform = "na1"
    routing_region = REGION_ROUTING[platform]

    print(f"Step 1: Fetching {platform.upper()} Challenger leaderboard...")
    challenger = get_challenger_league(platform)

    if not challenger:
        print("✗ Failed to fetch leaderboard")
        return

    entries = challenger.get('entries', [])
    print(f"✓ Found {len(entries)} Challenger players\n")

    target_matches = 5
    collected_matches = 0

    for i, entry in enumerate(entries[:20]):  # Check first 20 players
        puuid = entry.get('puuid')
        summoner_name = entry.get('summonerName', 'Unknown')

        if not puuid:
            continue

        print(f"\n[Player {i+1}] {summoner_name}")

        # Get match history
        print(f"  Fetching match history...")
        match_ids = get_match_ids(puuid, routing_region, count=10)

        if not match_ids:
            print(f"  ✗ No matches found")
            continue

        print(f"  ✓ Found {len(match_ids)} matches")

        # Process matches
        for j, match_id in enumerate(match_ids):
            if collected_matches >= target_matches:
                break

            print(f"  [{j+1}/{len(match_ids)}] {match_id[:15]}...")

            # Get match info
            print(f"    Fetching match info...", end=" ", flush=True)
            match_info = get_match_info(match_id, routing_region)

            if not match_info:
                print(f"✗ Failed")
                continue

            print(f"✓")

            # Check patch
            game_version = match_info.get('info', {}).get('gameVersion', '')
            if not is_valid_patch(game_version, min_patch="14.21"):
                print(f"    ✗ Old patch: {game_version}")
                continue

            # Check if jungler
            participants = match_info.get('info', {}).get('participants', [])
            participant_id = None

            for p in participants:
                if p.get('puuid') == puuid and p.get('teamPosition') == 'JUNGLE':
                    participant_id = p.get('participantId')
                    break

            if not participant_id:
                print(f"    ✗ Not a jungle game")
                continue

            print(f"    ✓ Jungle game (participant {participant_id})")

            # Get timeline
            print(f"    Fetching timeline...", end=" ", flush=True)
            timeline = get_match_timeline(match_id, routing_region)

            if not timeline:
                print(f"✗ Failed")
                continue

            print(f"✓")

            # Extract jungle path
            print(f"    Extracting jungle path...", end=" ", flush=True)
            jungler_team = 100 if participant_id <= 5 else 200
            jungle_path = extract_full_jungle_path(
                timeline, participant_id, jungler_team,
                camp_min_confidence=0.3,
                gank_min_confidence=0.6
            )

            if len(jungle_path) < 5:
                print(f"✗ Too few actions ({len(jungle_path)})")
                continue

            print(f"✓ {len(jungle_path)} actions")

            collected_matches += 1
            print(f"    ✅ Match collected! ({collected_matches}/{target_matches})")

        if collected_matches >= target_matches:
            print(f"\n✓ Reached target of {target_matches} matches!")
            break

    print(f"\n{'='*70}")
    print(f"✅ TEST COMPLETE")
    print(f"{'='*70}")
    print(f"Collected {collected_matches} jungle matches")
    print(f"\nThe full collection should work the same way,")
    print(f"just with more matches (2000+) and more players.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
