"""
Collect jungle data from Challenger/Grandmaster leaderboards.

Since we can't look up pros by name easily, we'll:
1. Get the Challenger leaderboard
2. Get their recent matches
3. Filter for jungle games
4. Extract jungle paths
"""

import json
import time
import requests
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from camp_inference_v3 import extract_jungle_path_cs

# API Configuration
API_KEY = "RGAPI-c6aee0b2-4405-465a-a1fb-507e754538d6"

REGION_ROUTING = {
    "na1": "americas",
    "euw1": "europe",
    "kr": "asia",
}

REQUEST_DELAY = 0.05  # 20 req/sec


def rate_limit():
    time.sleep(REQUEST_DELAY)


def get_challenger_league(platform: str) -> dict:
    """Get Challenger league entries."""
    url = f"https://{platform}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"✗ Error {response.status_code}: {response.text}")
        return {}


def get_summoner_by_id(summoner_id: str, platform: str) -> dict:
    """Get summoner by encrypted summonerID."""
    url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    return {}


def get_match_ids(puuid: str, routing_region: str, count: int = 10) -> list:
    """Get recent match IDs."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    headers = {"X-Riot-Token": API_KEY}
    params = {"start": 0, "count": count, "queue": 420}

    rate_limit()
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    return []


def get_match_info(match_id: str, routing_region: str) -> dict:
    """Get match info."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    return {}


def get_match_timeline(match_id: str, routing_region: str) -> dict:
    """Get match timeline."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    return {}


def collect_from_region(platform: str, num_players: int = 5, matches_per_player: int = 10):
    """Collect jungle data from top players in a region."""
    routing_region = REGION_ROUTING.get(platform)

    print(f"\n{'='*70}")
    print(f"Collecting from {platform.upper()} Challenger")
    print(f"{'='*70}")

    # Get Challenger leaderboard
    print("Fetching Challenger leaderboard...")
    league = get_challenger_league(platform)
    entries = league.get('entries', [])

    if not entries:
        print("✗ Could not fetch leaderboard")
        return []

    print(f"✓ Found {len(entries)} Challenger players")

    collected_data = []

    for i, entry in enumerate(entries[:num_players]):
        puuid = entry.get('puuid')
        if not puuid:
            continue

        print(f"\n[{i+1}/{num_players}] Challenger #{i+1}")
        print(f"  ✓ PUUID: {puuid[:20]}...")

        # Get match history
        match_ids = get_match_ids(puuid, routing_region, count=matches_per_player)
        print(f"  ✓ Found {len(match_ids)} matches")

        # Process matches
        jungle_matches = []

        for match_id in match_ids:
            # Get match info
            match_info = get_match_info(match_id, routing_region)
            if not match_info:
                continue

            # Find if this player was jungle
            participants = match_info.get('info', {}).get('participants', [])
            participant_id = None

            for p in participants:
                if p.get('puuid') == puuid and p.get('teamPosition') == 'JUNGLE':
                    participant_id = p.get('participantId')
                    break

            if not participant_id:
                continue  # Not a jungle game

            # Get timeline
            timeline = get_match_timeline(match_id, routing_region)
            if not timeline:
                continue

            # Extract jungle path
            jungle_path = extract_jungle_path_cs(timeline, participant_id, min_confidence=0.3)

            if len(jungle_path) >= 5:
                jungle_matches.append({
                    'match_id': match_id,
                    'participant_id': participant_id,
                    'jungle_path': [
                        {
                            'timestamp': c.timestamp,
                            'camp_name': c.camp_name,
                            'camps_cleared': c.camps_cleared,
                            'position': c.position,
                            'confidence': c.confidence
                        }
                        for c in jungle_path
                    ]
                })
                print(f"    ✓ {match_id[:15]}: {len(jungle_path)} clears")

        if jungle_matches:
            collected_data.append({
                'player_id': f"Challenger_{platform}_{i+1}",
                'puuid': puuid,
                'platform': platform,
                'matches': jungle_matches
            })

            print(f"  ✓ Collected {len(jungle_matches)} jungle games")

    return collected_data


def main():
    print("="*70)
    print("CHALLENGER LEADERBOARD DATA COLLECTION")
    print("="*70)

    # Collect from multiple regions for diverse playstyles
    all_data = []

    # Target: 400 jungle games total
    # Strategy: ~100 players × 10 matches = 1000 total matches
    # With ~40% jungle rate → ~400 jungle games

    # NA - 40 players, 10 matches each
    all_data.extend(collect_from_region("na1", num_players=40, matches_per_player=10))

    # Korea - 40 players, 10 matches each (best region)
    all_data.extend(collect_from_region("kr", num_players=40, matches_per_player=10))

    # EU West - 20 players, 10 matches each
    all_data.extend(collect_from_region("euw1", num_players=20, matches_per_player=10))

    # Save data
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "challenger_jungle_data.json"

    with open(output_file, 'w') as f:
        json.dump({
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_players': len(all_data),
            'total_matches': sum(len(p['matches']) for p in all_data),
            'players': all_data
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Players: {len(all_data)}")
    print(f"Matches: {sum(len(p['matches']) for p in all_data)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
