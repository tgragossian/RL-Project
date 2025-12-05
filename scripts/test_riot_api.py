"""
Test script to explore Riot API timeline data structure.

This will help us understand what data we can actually extract for the RL model.
"""

import requests
import json


def get_sample_timeline():
    """
    Fetch a sample match timeline to see what data is available.

    We'll use a publicly available high-elo match ID.
    Note: You'll need a Riot API key from https://developer.riotgames.com/
    """

    # TODO: Get your API key from https://developer.riotgames.com/
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your key

    # Example: NA1 region, sample match ID (you'd get this from match history)
    # This is just a placeholder - we'll need real match IDs
    REGION = "americas"  # americas, asia, europe, sea
    MATCH_ID = "NA1_1234567890"  # Placeholder

    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{MATCH_ID}/timeline"
    headers = {"X-Riot-Token": API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {response.text}")
        return None


def analyze_timeline_structure(timeline_data):
    """
    Explore the structure of timeline data to see what we can extract.
    """
    if not timeline_data:
        print("No timeline data to analyze")
        return

    print("=== TIMELINE DATA STRUCTURE ===\n")

    # Top level keys
    print("Top-level keys:", list(timeline_data.keys()))
    print()

    # Info section
    if 'info' in timeline_data:
        info = timeline_data['info']
        print("Info keys:", list(info.keys()))
        print(f"Frame interval: {info.get('frameInterval')}ms")
        print(f"Number of frames: {len(info.get('frames', []))}")
        print()

        # First frame structure
        if info.get('frames'):
            first_frame = info['frames'][0]
            print("First frame keys:", list(first_frame.keys()))
            print(f"Timestamp: {first_frame.get('timestamp')}ms")
            print()

            # Events in first frame
            if 'events' in first_frame:
                print(f"Number of events in first frame: {len(first_frame['events'])}")
                if first_frame['events']:
                    print("Sample event types:",
                          list(set(e.get('type') for e in first_frame['events'])))
                print()

            # Participant frames
            if 'participantFrames' in first_frame:
                print("Number of participants:", len(first_frame['participantFrames']))
                sample_participant = list(first_frame['participantFrames'].values())[0]
                print("Participant data keys:", list(sample_participant.keys()))
                print()

    # Save full structure to file for detailed inspection
    with open('/Users/thomas/Desktop/RL Project/RL-Project/timeline_sample.json', 'w') as f:
        json.dump(timeline_data, f, indent=2)
    print("Full timeline saved to timeline_sample.json")


def extract_jungle_events(timeline_data):
    """
    Try to extract jungle-relevant events from timeline.
    """
    if not timeline_data or 'info' not in timeline_data:
        return

    print("\n=== JUNGLE-RELEVANT DATA ===\n")

    frames = timeline_data['info'].get('frames', [])

    jungle_events = []
    position_snapshots = []

    for frame_idx, frame in enumerate(frames):
        timestamp = frame.get('timestamp', 0)

        # Extract events
        for event in frame.get('events', []):
            event_type = event.get('type')

            # Look for jungle-related events
            if event_type in ['ELITE_MONSTER_KILL', 'MONSTER_KILL', 'CHAMPION_KILL']:
                jungle_events.append({
                    'timestamp': timestamp,
                    'type': event_type,
                    'event': event
                })

        # Extract participant positions (every minute)
        if 'participantFrames' in frame:
            for participant_id, pframe in frame['participantFrames'].items():
                position_snapshots.append({
                    'timestamp': timestamp,
                    'participant': participant_id,
                    'position': pframe.get('position'),
                    'jungleMinionsKilled': pframe.get('jungleMinionsKilled', 0),
                    'minionsKilled': pframe.get('minionsKilled', 0),
                    'level': pframe.get('level', 1),
                    'xp': pframe.get('xp', 0),
                    'gold': pframe.get('currentGold', 0),
                })

    print(f"Found {len(jungle_events)} jungle-related events")
    print(f"Event types: {set(e['type'] for e in jungle_events)}")
    print()

    print(f"Position snapshots: {len(position_snapshots)}")
    if position_snapshots:
        sample = position_snapshots[0]
        print(f"Sample snapshot keys: {list(sample.keys())}")

    return jungle_events, position_snapshots


def main():
    """
    Main exploration function.
    """
    print("Riot API Timeline Data Explorer")
    print("=" * 50)
    print()
    print("This script will help us understand what data we can extract")
    print("for the RL jungling project.")
    print()
    print("SETUP REQUIRED:")
    print("1. Get API key from: https://developer.riotgames.com/")
    print("2. Replace 'YOUR_API_KEY_HERE' in this script")
    print("3. Get a real match ID from a high-elo game")
    print()
    print("=" * 50)
    print()

    # For now, let's document what we KNOW is available based on research
    print("KNOWN AVAILABLE DATA (from API docs):")
    print()
    print("✓ Position data (x, y) every minute for all players")
    print("✓ Elite monster kills (Dragon, Baron, Herald)")
    print("✓ Champion kills, deaths, assists")
    print("✓ Gold, XP, level snapshots per minute")
    print("✓ Total jungle minions killed (cumulative)")
    print("✓ Item purchases and sales")
    print("✓ Skill level-ups")
    print()
    print("✗ Individual jungle camp clears (Blue, Red, etc.) - REMOVED by Riot")
    print("✗ Real-time position data (only 1-minute intervals)")
    print("✗ Vision/ward data in replay format")
    print()
    print("=" * 50)
    print()

    # Uncomment when you have API key and match ID:
    # timeline = get_sample_timeline()
    # if timeline:
    #     analyze_timeline_structure(timeline)
    #     extract_jungle_events(timeline)


if __name__ == "__main__":
    main()
