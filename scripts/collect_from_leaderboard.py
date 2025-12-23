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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append(str(Path(__file__).parent.parent / "src"))
from jungle_path_unified import extract_full_jungle_path

# API Configuration - Multiple keys for rotation
API_KEYS = [
    "RGAPI-c6aee0b2-4405-465a-a1fb-507e754538d6"
    # Add more API keys here to rotate between accounts:
    # "RGAPI-YOUR-SECOND-KEY-HERE",
    # "RGAPI-YOUR-THIRD-KEY-HERE",
]

# Track which key to use next (round-robin)
_current_key_index = 0

def get_next_api_key():
    """Get next API key in rotation."""
    global _current_key_index
    key = API_KEYS[_current_key_index]
    _current_key_index = (_current_key_index + 1) % len(API_KEYS)
    return key

REGION_ROUTING = {
    "na1": "americas",
    "euw1": "europe",
    "kr": "asia",
}

# Adjust delay based on number of keys (more keys = less delay needed)
REQUEST_DELAY = 0.05 / len(API_KEYS)  # Divide delay by number of keys
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds


def create_session_with_retries():
    """Create a requests session with automatic retry logic."""
    session = requests.Session()

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# Global session object
_session = create_session_with_retries()


def rate_limit():
    time.sleep(REQUEST_DELAY)


def get_challenger_league(platform: str) -> dict:
    """Get Challenger league entries with retry logic."""
    url = f"https://{platform}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
    headers = {"X-Riot-Token": get_next_api_key()}

    rate_limit()
    try:
        response = _session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return {}
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return {}


def get_grandmaster_league(platform: str) -> dict:
    """Get Grandmaster league entries with retry logic."""
    url = f"https://{platform}.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5"
    headers = {"X-Riot-Token": get_next_api_key()}

    rate_limit()
    try:
        response = _session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return {}
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return {}


def get_summoner_by_id(summoner_id: str, platform: str) -> dict:
    """Get summoner by encrypted summonerID with retry logic."""
    url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    headers = {"X-Riot-Token": get_next_api_key()}

    rate_limit()
    try:
        response = _session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"✗ Connection error getting summoner: {e}")
        return {}


def get_match_ids(puuid: str, routing_region: str, count: int = 10) -> list:
    """Get recent match IDs with retry logic."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    headers = {"X-Riot-Token": get_next_api_key()}
    params = {"start": 0, "count": count, "queue": 420}

    rate_limit()
    try:
        response = _session.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"✗ Connection error getting match IDs: {e}")
        return []


def is_likely_pro_player(summoner_name: str) -> bool:
    """Check if summoner name suggests they might be a pro player."""
    # Common patterns in pro player names
    pro_indicators = [
        'T1', 'GEN', 'DK', 'KT', 'HLE', 'DRX', 'BRO',  # LCK teams
        'C9', 'TL', 'FLY', 'NRG', 'DIG', '100',  # LCS teams
        'G2', 'FNC', 'MAD', 'VIT', 'BDS', 'SK', 'XL',  # LEC teams
        'JDG', 'BLG', 'WBG', 'LNG', 'EDG', 'TES', 'IG', 'RNG',  # LPL teams
    ]

    summoner_upper = summoner_name.upper()
    return any(indicator in summoner_upper for indicator in pro_indicators)


def get_match_info(match_id: str, routing_region: str) -> dict:
    """Get match info with retry logic."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": get_next_api_key()}

    rate_limit()
    try:
        response = _session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"✗ Connection error getting match info for {match_id}: {e}")
        time.sleep(2)  # Back off a bit on connection errors
        return {}


def get_match_timeline(match_id: str, routing_region: str) -> dict:
    """Get match timeline with retry logic."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": get_next_api_key()}

    rate_limit()
    try:
        response = _session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"✗ Connection error getting timeline for {match_id}: {e}")
        time.sleep(2)  # Back off a bit on connection errors
        return {}


def is_valid_patch(game_version: str, min_patch: str = "14.21") -> bool:
    """
    Check if game version is valid (patch 14.21 or later).

    As of Dec 2025, this covers approximately the last 5 patches:
    - 14.21, 14.22, 14.23, 14.24, 15.1+

    Args:
        game_version: Full version string like "14.23.123.4567"
        min_patch: Minimum patch version to accept (e.g., "14.21")

    Returns:
        True if patch is >= min_patch
    """
    try:
        # Extract major.minor from version (e.g., "14.23.123.4567" -> "14.23")
        version_parts = game_version.split('.')
        if len(version_parts) < 2:
            return False

        major = int(version_parts[0])
        minor = int(version_parts[1])

        min_parts = min_patch.split('.')
        min_major = int(min_parts[0])
        min_minor = int(min_parts[1])

        # Compare versions
        if major > min_major:
            return True
        elif major == min_major and minor >= min_minor:
            return True
        else:
            return False
    except (ValueError, IndexError):
        return False


def collect_from_region(platform: str, target_jungle_matches: int = 350, matches_per_player: int = 20):
    """Collect jungle data from top players in a region until target is reached."""
    routing_region = REGION_ROUTING.get(platform)

    print(f"\n{'='*70}")
    print(f"Collecting from {platform.upper()}")
    print(f"Target: {target_jungle_matches} jungle matches")
    print(f"{'='*70}")

    # Get both Challenger and Grandmaster leaderboards
    print("Fetching Challenger leaderboard...")
    challenger = get_challenger_league(platform)
    challenger_entries = challenger.get('entries', [])

    print("Fetching Grandmaster leaderboard...")
    grandmaster = get_grandmaster_league(platform)
    grandmaster_entries = grandmaster.get('entries', [])

    # Combine and sort by LP (highest first)
    all_entries = []
    for entry in challenger_entries:
        entry['tier'] = 'CHALLENGER'
        all_entries.append(entry)
    for entry in grandmaster_entries:
        entry['tier'] = 'GRANDMASTER'
        all_entries.append(entry)

    # Sort by: pro player status > LP
    all_entries.sort(
        key=lambda e: (
            is_likely_pro_player(e.get('summonerName', '')),  # Pro players first
            e.get('leaguePoints', 0)  # Then by LP
        ),
        reverse=True
    )

    if not all_entries:
        print("✗ Could not fetch leaderboards")
        return []

    print(f"✓ Found {len(challenger_entries)} Challenger + {len(grandmaster_entries)} Grandmaster players")

    collected_data = []
    total_jungle_matches = 0
    players_processed = 0

    for i, entry in enumerate(all_entries):
        # Stop if we've reached our target
        if total_jungle_matches >= target_jungle_matches:
            print(f"\n✓ Reached target of {target_jungle_matches} jungle matches!")
            break

        puuid = entry.get('puuid')
        summoner_name = entry.get('summonerName', 'Unknown')
        tier = entry.get('tier', 'UNKNOWN')
        lp = entry.get('leaguePoints', 0)

        if not puuid:
            continue

        players_processed += 1
        is_pro = is_likely_pro_player(summoner_name)
        pro_tag = " [PRO?]" if is_pro else ""

        print(f"\n[Player {players_processed}] {tier} {summoner_name}{pro_tag} ({lp} LP)")
        print(f"  Progress: {total_jungle_matches}/{target_jungle_matches} jungle matches")
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

            # Check patch version
            game_version = match_info.get('info', {}).get('gameVersion', '')
            if not is_valid_patch(game_version, min_patch="14.21"):
                continue  # Skip old patches

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

            # Extract full jungle path (camps + ganks)
            jungler_team = 100 if participant_id <= 5 else 200
            jungle_path = extract_full_jungle_path(
                timeline, participant_id, jungler_team,
                camp_min_confidence=0.3,
                gank_min_confidence=0.6
            )

            if len(jungle_path) >= 5:
                camp_count = sum(1 for a in jungle_path if a.action_type == "camp")
                gank_count = sum(1 for a in jungle_path if a.action_type == "gank")

                # Extract patch version for storage
                patch_version = '.'.join(game_version.split('.')[:2])  # e.g., "14.23"

                jungle_matches.append({
                    'match_id': match_id,
                    'participant_id': participant_id,
                    'patch': patch_version,
                    'jungle_path': [
                        {
                            'timestamp': a.timestamp,
                            'camp_name': a.action_name,  # Can be camp or "gank_top/mid/bot"
                            'position': a.position,
                            'confidence': a.confidence
                        }
                        for a in jungle_path
                    ]
                })
                print(f"    ✓ {match_id[:15]} (Patch {patch_version}): {len(jungle_path)} actions ({camp_count} camps, {gank_count} ganks)")

        if jungle_matches:
            collected_data.append({
                'player_id': f"{tier}_{platform}_{players_processed}",
                'summoner_name': summoner_name,
                'puuid': puuid,
                'platform': platform,
                'tier': tier,
                'lp': lp,
                'is_likely_pro': is_pro,
                'matches': jungle_matches
            })

            total_jungle_matches += len(jungle_matches)
            print(f"  ✓ Collected {len(jungle_matches)} jungle games (Total: {total_jungle_matches})")

    print(f"\n✓ Region complete: {total_jungle_matches} jungle matches from {len(collected_data)} players")
    return collected_data


def main():
    print("="*70)
    print("HIGH ELO JUNGLE DATA COLLECTION")
    print("Collecting matches from Patch 14.21+ (Last 5 patches)")
    print(f"Using {len(API_KEYS)} API key(s) with rotation")
    print(f"Effective request delay: {REQUEST_DELAY*1000:.1f}ms")
    print("="*70)

    # Collect from multiple regions for diverse playstyles
    all_data = []

    # Target: 50 jungle games for testing data quality
    # Strategy: Quick test collection to validate camp detection
    # After visualizing, can scale up to 2000+ games

    # Korea - 50 jungle matches (test run)
    all_data.extend(collect_from_region("kr", target_jungle_matches=50, matches_per_player=20))

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
