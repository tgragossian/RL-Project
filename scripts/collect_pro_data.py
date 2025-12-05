"""
Collect match data from top pro junglers across multiple regions.

This script:
1. Reads pro jungler list from data/pro_junglers.json
2. Fetches their recent ranked games
3. Downloads match timelines
4. Extracts jungle paths using camp inference
5. Saves processed training data
"""

import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from camp_inference_v3 import extract_jungle_path_cs


# API Configuration
API_KEY = "RGAPI-c6aee0b2-4405-465a-a1fb-507e754538d6"

# Region mapping: platform → routing region
REGION_ROUTING = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",

    "euw1": "europe",
    "eune1": "europe",
    "tr1": "europe",
    "ru": "europe",

    "kr": "asia",
    "jp1": "asia",
}

# Rate limiting
REQUESTS_PER_SECOND = 20  # Dev key limit
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND


def rate_limit():
    """Simple rate limiting."""
    time.sleep(REQUEST_DELAY)


def get_summoner_by_name(summoner_name: str, platform: str) -> Optional[Dict]:
    """Get summoner data by name."""
    url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner_name}"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        print(f"  ✗ Summoner not found: {summoner_name} on {platform}")
        return None
    else:
        print(f"  ✗ Error {response.status_code}: {response.text}")
        return None


def get_match_ids(puuid: str, routing_region: str, count: int = 20) -> List[str]:
    """Get recent ranked match IDs for a player."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    headers = {"X-Riot-Token": API_KEY}
    params = {
        "start": 0,
        "count": count,
        "queue": 420,  # Ranked Solo/Duo
    }

    rate_limit()
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"  ✗ Error fetching matches: {response.status_code}")
        return []


def get_match_timeline(match_id: str, routing_region: str) -> Optional[Dict]:
    """Get match timeline."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return None  # Timeline expired
    else:
        print(f"  ✗ Error fetching timeline: {response.status_code}")
        return None


def get_match_info(match_id: str, routing_region: str) -> Optional[Dict]:
    """Get basic match info to identify jungler participant ID."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def find_jungler_participant(match_info: Dict, puuid: str) -> Optional[int]:
    """Find which participant ID is the jungler with given PUUID."""
    participants = match_info.get('info', {}).get('participants', [])

    for p in participants:
        if p.get('puuid') == puuid and p.get('teamPosition') == 'JUNGLE':
            return p.get('participantId')

    return None


def collect_player_data(summoner_name: str, platform: str, num_matches: int = 20):
    """
    Collect jungle pathing data for a single player.

    Returns:
        Dict with player info and collected matches
    """
    routing_region = REGION_ROUTING.get(platform)
    if not routing_region:
        print(f"✗ Unknown platform: {platform}")
        return None

    print(f"\n{'='*70}")
    print(f"Collecting data for: {summoner_name} ({platform})")
    print(f"{'='*70}")

    # Get summoner
    summoner = get_summoner_by_name(summoner_name, platform)
    if not summoner:
        return None

    puuid = summoner['puuid']
    print(f"✓ Found summoner (PUUID: {puuid[:20]}...)")

    # Get match IDs
    print(f"Fetching {num_matches} recent ranked matches...")
    match_ids = get_match_ids(puuid, routing_region, count=num_matches)
    print(f"✓ Found {len(match_ids)} matches")

    # Collect timelines
    collected_matches = []

    for i, match_id in enumerate(match_ids):
        print(f"\n  [{i+1}/{len(match_ids)}] Processing {match_id}...")

        # Get match info
        match_info = get_match_info(match_id, routing_region)
        if not match_info:
            print(f"    ✗ Could not fetch match info")
            continue

        # Find jungler participant ID
        participant_id = find_jungler_participant(match_info, puuid)
        if not participant_id:
            print(f"    ✗ Player was not jungler in this match")
            continue

        # Get timeline
        timeline = get_match_timeline(match_id, routing_region)
        if not timeline:
            print(f"    ✗ Timeline not available (may be expired)")
            continue

        # Extract jungle path
        jungle_path = extract_jungle_path_cs(timeline, participant_id, min_confidence=0.3)

        if len(jungle_path) < 5:
            print(f"    ✗ Not enough jungle activity ({len(jungle_path)} clears)")
            continue

        print(f"    ✓ Extracted {len(jungle_path)} camp clears")

        # Store data
        collected_matches.append({
            'match_id': match_id,
            'participant_id': participant_id,
            'jungle_path': [
                {
                    'timestamp': clear.timestamp,
                    'camp_name': clear.camp_name,
                    'camps_cleared': clear.camps_cleared,
                    'position': clear.position,
                    'confidence': clear.confidence
                }
                for clear in jungle_path
            ],
            'timeline': timeline  # Save full timeline for later processing
        })

    print(f"\n{'='*70}")
    print(f"✓ Collected {len(collected_matches)} matches for {summoner_name}")
    print(f"{'='*70}")

    return {
        'summoner_name': summoner_name,
        'platform': platform,
        'puuid': puuid,
        'matches': collected_matches
    }


def main():
    """Main data collection pipeline."""
    print("="*70)
    print("PRO JUNGLER DATA COLLECTION PIPELINE")
    print("="*70)

    # Load pro junglers list
    junglers_file = Path(__file__).parent.parent / "data" / "pro_junglers.json"

    if not junglers_file.exists():
        print("✗ No pro_junglers.json found!")
        print("  Create data/pro_junglers.json with list of players")
        return

    with open(junglers_file) as f:
        config = json.load(f)

    junglers = config.get('junglers', [])
    print(f"\nLoaded {len(junglers)} pro junglers from config")
    print()

    # Collect data for each player
    all_player_data = []

    for jungler in junglers:
        name = jungler.get('name')
        region = jungler.get('region')

        if not name or not region:
            print(f"✗ Skipping invalid entry: {jungler}")
            continue

        player_data = collect_player_data(name, region, num_matches=20)

        if player_data:
            all_player_data.append(player_data)

        # Small delay between players
        time.sleep(2)

    # Save collected data
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pro_jungle_data.json"

    with open(output_file, 'w') as f:
        json.dump({
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_players': len(all_player_data),
            'total_matches': sum(len(p['matches']) for p in all_player_data),
            'players': all_player_data
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ DATA COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Players collected: {len(all_player_data)}")
    print(f"Total matches: {sum(len(p['matches']) for p in all_player_data)}")
    print(f"Saved to: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
