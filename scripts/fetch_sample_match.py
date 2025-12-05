"""
Fetch a real high-elo match timeline from Riot API.

This script will:
1. Get a high-elo player's recent matches
2. Download a match timeline
3. Analyze what data we can extract for RL training
"""

import requests
import json
import os
import time
from pathlib import Path


# API Configuration
API_KEY = "RGAPI-c6aee0b2-4405-465a-a1fb-507e754538d6"
REGION = "americas"  # Routing value for match data
PLATFORM = "na1"     # Platform for summoner data

# High-elo players to sample from (these are known pro/high-elo accounts)
SAMPLE_SUMMONERS = {
    "Doublelift": "na1",
    "Spica": "na1",
    "Blaber": "na1",
}


def get_summoner_puuid(summoner_name: str, platform: str = PLATFORM) -> str:
    """Get summoner PUUID from summoner name."""
    url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner_name}"
    headers = {"X-Riot-Token": API_KEY}

    print(f"Fetching summoner: {summoner_name}...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        puuid = data['puuid']
        print(f"✓ Found PUUID: {puuid[:20]}...")
        return puuid
    else:
        print(f"✗ Error {response.status_code}: {response.text}")
        return None


def get_recent_matches(puuid: str, count: int = 5) -> list:
    """Get recent match IDs for a player."""
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    headers = {"X-Riot-Token": API_KEY}
    params = {
        "start": 0,
        "count": count,
        "queue": 420,  # 420 = Ranked Solo/Duo
    }

    print(f"Fetching {count} recent ranked matches...")
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        matches = response.json()
        print(f"✓ Found {len(matches)} matches")
        return matches
    else:
        print(f"✗ Error {response.status_code}: {response.text}")
        return []


def get_match_timeline(match_id: str) -> dict:
    """Get detailed timeline for a match."""
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": API_KEY}

    print(f"Fetching timeline for match {match_id}...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print(f"✓ Timeline downloaded successfully")
        return response.json()
    else:
        print(f"✗ Error {response.status_code}: {response.text}")
        return None


def analyze_timeline(timeline: dict):
    """Analyze what data is available in the timeline."""
    print("\n" + "="*60)
    print("TIMELINE DATA ANALYSIS")
    print("="*60)

    info = timeline.get('info', {})
    frames = info.get('frames', [])

    print(f"\nTotal frames: {len(frames)}")
    print(f"Frame interval: {info.get('frameInterval')}ms ({info.get('frameInterval')/1000}s)")
    print(f"Game duration: ~{len(frames) * info.get('frameInterval') / 60000:.1f} minutes")

    # Analyze first frame
    if frames:
        frame = frames[0]
        print(f"\n--- Frame 0 Structure ---")
        print(f"Timestamp: {frame['timestamp']}ms")
        print(f"Keys: {list(frame.keys())}")

        # Events
        events = frame.get('events', [])
        event_types = set(e.get('type') for e in events)
        print(f"\nEvent types in frame 0: {event_types}")

        # Participant frames
        participant_frames = frame.get('participantFrames', {})
        if participant_frames:
            sample_p = list(participant_frames.values())[0]
            print(f"\nParticipant data keys: {list(sample_p.keys())}")
            print(f"Sample participant frame:")
            for key, value in sample_p.items():
                if isinstance(value, dict):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

    # Analyze mid-game frame (around 5 minutes)
    mid_frame_idx = min(5, len(frames) - 1)
    if mid_frame_idx > 0:
        frame = frames[mid_frame_idx]
        print(f"\n--- Frame {mid_frame_idx} (t={frame['timestamp']/1000:.0f}s) ---")

        events = frame.get('events', [])
        event_types = {}
        for e in events:
            etype = e.get('type')
            event_types[etype] = event_types.get(etype, 0) + 1

        print(f"Events by type:")
        for etype, count in sorted(event_types.items()):
            print(f"  {etype}: {count}")

    # Look for jungle-relevant events across all frames
    print(f"\n--- Jungle-Relevant Events (all frames) ---")
    all_events = {}
    for frame in frames:
        for event in frame.get('events', []):
            etype = event.get('type')
            all_events[etype] = all_events.get(etype, 0) + 1

    for etype in sorted(all_events.keys()):
        print(f"  {etype}: {all_events[etype]}")

    print("\n" + "="*60)


def save_timeline(timeline: dict, match_id: str):
    """Save timeline to file for inspection."""
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"timeline_{match_id}.json"
    with open(filepath, 'w') as f:
        json.dump(timeline, f, indent=2)

    print(f"\n✓ Timeline saved to: {filepath}")


def main():
    print("="*60)
    print("RIOT API TIMELINE FETCHER")
    print("="*60)
    print()

    # Try to get a summoner's matches
    # Note: Summoner names can change, so this might fail
    # Alternative: Use a known match ID directly

    # Let's try with a direct match ID first (more reliable)
    # You can get match IDs from op.gg or similar sites

    print("Option 1: Fetch using summoner name")
    print("Option 2: Use a known match ID")
    print()

    # Try option 2 first (more reliable for testing)
    # This is a sample match ID - replace with a recent one if it fails
    test_match_id = "NA1_5089218526"  # Example match ID

    print(f"Testing with match ID: {test_match_id}")
    timeline = get_match_timeline(test_match_id)

    if timeline:
        analyze_timeline(timeline)
        save_timeline(timeline, test_match_id)
        print("\n✅ SUCCESS! We can access Riot API timeline data!")
        print("\nNext steps:")
        print("1. This data shows position every 60s + event data")
        print("2. We can infer camp clears from jungle minions killed delta")
        print("3. We have champion kills, objectives, gold/xp/level")
        print("4. Ready to build data collection pipeline!")
    else:
        print("\n⚠️  Failed to fetch timeline. This could be:")
        print("1. Match ID is too old (timelines expire after 1 year)")
        print("2. API key issue")
        print("3. Rate limiting")
        print()
        print("Let's try getting recent matches from a known player...")

        # Fallback: try to get fresh match IDs
        # Note: This requires the summoner name to be exact
        summoner_name = "Doublelift"
        platform = "na1"

        puuid = get_summoner_puuid(summoner_name, platform)
        if puuid:
            match_ids = get_recent_matches(puuid, count=3)
            if match_ids:
                print(f"\nTrying most recent match: {match_ids[0]}")
                timeline = get_match_timeline(match_ids[0])
                if timeline:
                    analyze_timeline(timeline)
                    save_timeline(timeline, match_ids[0])


if __name__ == "__main__":
    main()
