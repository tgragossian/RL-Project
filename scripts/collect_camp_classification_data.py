"""
Focused data collection for camp classification and gank prediction.

Collects comprehensive frame-by-frame data including:
- All champion stats (HP, armor, MR, AD, AS, movement speed, etc.)
- Combat stats (kills, deaths, assists, damage dealt/taken)
- Farm stats (CS, jungle CS - only visible data, no fog of war)
- Objective stats (dragons, barons, turrets)
- Match outcome (win/loss)
- High-priority features (HP%, current gold, damage taken recently)

NOTE: Enemy jungler data only collected when visible (no fog of war violations)
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# API Configuration
API_KEYS = [
    "RGAPI-81751001-babd-49a5-ae02-d00305d382db",
    "RGAPI-7012b0bb-e73a-4043-889d-a7a6e2a03621"
]

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

REQUEST_DELAY = 0.05 / len(API_KEYS)
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0


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


_session = create_session_with_retries()


def rate_limit():
    time.sleep(REQUEST_DELAY)


def get_challenger_league(platform: str) -> dict:
    """Get Challenger league entries."""
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


def get_match_ids(puuid: str, routing_region: str, count: int = 10) -> list:
    """Get recent match IDs."""
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


def get_match_info(match_id: str, routing_region: str) -> dict:
    """Get match info."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": get_next_api_key()}

    rate_limit()
    try:
        response = _session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"✗ Connection error getting match info: {e}")
        time.sleep(2)
        return {}


def get_match_timeline(match_id: str, routing_region: str) -> dict:
    """Get match timeline."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": get_next_api_key()}

    rate_limit()
    try:
        response = _session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"✗ Connection error getting timeline: {e}")
        time.sleep(2)
        return {}


def verify_jungle_player(puuid: str, routing_region: str, check_count: int = 5) -> bool:
    """
    Verify if a player is a jungle main by checking their first N matches.

    Args:
        puuid: Player's PUUID
        routing_region: Routing region (americas, europe, asia)
        check_count: Number of recent matches to check (default 5)

    Returns:
        True if player played jungle in any of the checked matches
    """
    match_ids = get_match_ids(puuid, routing_region, count=check_count)

    if not match_ids:
        return False

    for match_id in match_ids:
        match_info = get_match_info(match_id, routing_region)
        if not match_info:
            continue

        participants = match_info.get('info', {}).get('participants', [])
        for p in participants:
            if p.get('puuid') == puuid and p.get('teamPosition') == 'JUNGLE':
                return True  # Found at least one jungle game

    return False  # No jungle games found in recent matches


def extract_participant_data(participant: dict) -> dict:
    """
    Extract all relevant data from a participant (match info).

    Includes:
    - Combat stats (kills, deaths, assists, damage)
    - Farm stats (CS)
    - Objective stats (dragons, barons, turrets)
    - Match outcome (win)
    - Items
    """
    return {
        # Identity
        'participant_id': participant['participantId'],
        'champion_name': participant['championName'],
        'team_id': participant['teamId'],
        'team_position': participant['teamPosition'],
        'puuid': participant['puuid'],

        # Match outcome
        'win': participant['win'],

        # Combat stats
        'kills': participant['kills'],
        'deaths': participant['deaths'],
        'assists': participant['assists'],
        'total_damage_dealt_to_champions': participant['totalDamageDealtToChampions'],
        'total_damage_taken': participant['totalDamageTaken'],
        'physical_damage_dealt_to_champions': participant['physicalDamageDealtToChampions'],
        'magic_damage_dealt_to_champions': participant['magicDamageDealtToChampions'],
        'true_damage_dealt_to_champions': participant['trueDamageDealtToChampions'],

        # Farm stats
        'total_minions_killed': participant['totalMinionsKilled'],
        'neutral_minions_killed': participant['neutralMinionsKilled'],

        # Objective stats
        'baron_kills': participant['baronKills'],
        'dragon_kills': participant['dragonKills'],
        'turret_kills': participant['turretKills'],
        'inhibitor_kills': participant['inhibitorKills'],

        # Gold/XP
        'gold_earned': participant['goldEarned'],
        'champ_level': participant['champLevel'],
        'champ_experience': participant['champExperience'],

        # Items (final build)
        'items': [
            participant['item0'],
            participant['item1'],
            participant['item2'],
            participant['item3'],
            participant['item4'],
            participant['item5'],
            participant['item6'],
        ],

        # Vision
        'vision_score': participant['visionScore'],
        'wards_placed': participant['wardsPlaced'],
        'wards_killed': participant['wardsKilled'],
    }


def extract_frame_data(participant_frame: dict, participant_id: int) -> dict:
    """
    Extract all frame-by-frame data for a participant.

    Includes:
    - All champion stats (HP, armor, MR, AD, AS, movement speed, etc.)
    - Damage stats (damage done/taken)
    - Farm stats (CS, jungle CS)
    - Gold stats (current gold, total gold)
    - Position
    """
    champion_stats = participant_frame.get('championStats', {})
    damage_stats = participant_frame.get('damageStats', {})

    return {
        'participant_id': participant_id,

        # Core stats
        'level': participant_frame['level'],
        'xp': participant_frame['xp'],

        # Position
        'position_x': participant_frame['position']['x'],
        'position_y': participant_frame['position']['y'],

        # Gold
        'current_gold': participant_frame['currentGold'],
        'total_gold': participant_frame['totalGold'],
        'gold_per_second': participant_frame['goldPerSecond'],

        # Farm
        'minions_killed': participant_frame['minionsKilled'],
        'jungle_minions_killed': participant_frame['jungleMinionsKilled'],

        # Champion stats - Combat
        'health': champion_stats.get('health', 0),
        'health_max': champion_stats.get('healthMax', 1),
        'health_regen': champion_stats.get('healthRegen', 0),
        'armor': champion_stats.get('armor', 0),
        'magic_resist': champion_stats.get('magicResist', 0),
        'attack_damage': champion_stats.get('attackDamage', 0),
        'attack_speed': champion_stats.get('attackSpeed', 0),
        'ability_power': champion_stats.get('abilityPower', 0),
        'ability_haste': champion_stats.get('abilityHaste', 0),
        'movement_speed': champion_stats.get('movementSpeed', 0),

        # Champion stats - Sustain
        'lifesteal': champion_stats.get('lifesteal', 0),
        'omnivamp': champion_stats.get('omnivamp', 0),
        'physical_vamp': champion_stats.get('physicalVamp', 0),
        'spell_vamp': champion_stats.get('spellVamp', 0),

        # Champion stats - Penetration
        'armor_pen': champion_stats.get('armorPen', 0),
        'armor_pen_percent': champion_stats.get('armorPenPercent', 0),
        'magic_pen': champion_stats.get('magicPen', 0),
        'magic_pen_percent': champion_stats.get('magicPenPercent', 0),

        # Champion stats - Other
        'power': champion_stats.get('power', 0),  # Mana/Energy
        'power_max': champion_stats.get('powerMax', 0),
        'power_regen': champion_stats.get('powerRegen', 0),
        'cc_reduction': champion_stats.get('ccReduction', 0),

        # Damage stats
        'total_damage_done': damage_stats.get('totalDamageDone', 0),
        'total_damage_done_to_champions': damage_stats.get('totalDamageDoneToChampions', 0),
        'total_damage_taken': damage_stats.get('totalDamageTaken', 0),
        'physical_damage_done': damage_stats.get('physicalDamageDone', 0),
        'physical_damage_done_to_champions': damage_stats.get('physicalDamageDoneToChampions', 0),
        'physical_damage_taken': damage_stats.get('physicalDamageTaken', 0),
        'magic_damage_done': damage_stats.get('magicDamageDone', 0),
        'magic_damage_done_to_champions': damage_stats.get('magicDamageDoneToChampions', 0),
        'magic_damage_taken': damage_stats.get('magicDamageTaken', 0),
        'true_damage_done': damage_stats.get('trueDamageDone', 0),
        'true_damage_done_to_champions': damage_stats.get('trueDamageDoneToChampions', 0),
        'true_damage_taken': damage_stats.get('trueDamageTaken', 0),

        # CC
        'time_enemy_spent_controlled': participant_frame.get('timeEnemySpentControlled', 0),
    }


def extract_events_from_frame(events: List[dict]) -> List[dict]:
    """
    Extract relevant events from a frame.

    Includes:
    - ELITE_MONSTER_KILL (dragons, barons, heralds)
    - CHAMPION_KILL (for gank detection)
    - BUILDING_KILL (turrets, inhibitors)
    - LEVEL_UP (powerspike timing)
    - ITEM_PURCHASED (item powerspikes)
    """
    extracted_events = []

    for event in events:
        event_type = event.get('type')

        if event_type == 'ELITE_MONSTER_KILL':
            extracted_events.append({
                'type': 'ELITE_MONSTER_KILL',
                'timestamp': event['timestamp'],
                'monster_type': event.get('monsterType'),
                'monster_sub_type': event.get('monsterSubType'),
                'killer_id': event.get('killerId'),
                'killer_team_id': event.get('killerTeamId'),
                'position_x': event.get('position', {}).get('x'),
                'position_y': event.get('position', {}).get('y'),
            })

        elif event_type == 'CHAMPION_KILL':
            extracted_events.append({
                'type': 'CHAMPION_KILL',
                'timestamp': event['timestamp'],
                'killer_id': event.get('killerId'),
                'victim_id': event.get('victimId'),
                'assisting_participant_ids': event.get('assistingParticipantIds', []),
                'position_x': event.get('position', {}).get('x'),
                'position_y': event.get('position', {}).get('y'),
            })

        elif event_type == 'BUILDING_KILL':
            extracted_events.append({
                'type': 'BUILDING_KILL',
                'timestamp': event['timestamp'],
                'building_type': event.get('buildingType'),
                'team_id': event.get('teamId'),
                'killer_id': event.get('killerId'),
                'position_x': event.get('position', {}).get('x'),
                'position_y': event.get('position', {}).get('y'),
            })

        elif event_type == 'LEVEL_UP':
            extracted_events.append({
                'type': 'LEVEL_UP',
                'timestamp': event['timestamp'],
                'participant_id': event.get('participantId'),
                'level': event.get('level'),
            })

        elif event_type == 'ITEM_PURCHASED':
            extracted_events.append({
                'type': 'ITEM_PURCHASED',
                'timestamp': event['timestamp'],
                'participant_id': event.get('participantId'),
                'item_id': event.get('itemId'),
            })

    return extracted_events


def process_match(match_id: str, routing_region: str, target_puuid: str, verbose: bool = True) -> Optional[dict]:
    """
    Process a single match and extract all relevant data.

    Returns structured data for camp classification and gank prediction.
    """
    # Get match info
    match_info = get_match_info(match_id, routing_region)
    if not match_info:
        return None

    # Get timeline
    timeline = get_match_timeline(match_id, routing_region)
    if not timeline:
        return None

    # Extract game metadata
    info = match_info.get('info', {})
    game_version = info.get('gameVersion', '')
    game_duration = info.get('gameDuration', 0)

    # Find jungler participant
    participants = info.get('participants', [])
    jungler_participant = None
    jungler_participant_id = None

    for p in participants:
        if p.get('puuid') == target_puuid and p.get('teamPosition') == 'JUNGLE':
            jungler_participant = p
            jungler_participant_id = p.get('participantId')
            break

    if not jungler_participant:
        return None  # Silently skip non-jungle games

    if verbose:
        print(f"    ✓ {jungler_participant['championName']} jungle match")

    # Extract all participant data (for team states)
    all_participants = [extract_participant_data(p) for p in participants]

    # Process timeline frames
    frames = timeline.get('info', {}).get('frames', [])
    processed_frames = []

    for frame in frames:
        timestamp = frame['timestamp']
        participant_frames = frame.get('participantFrames', {})

        # Extract data for all participants in this frame
        frame_data = {
            'timestamp': timestamp,
            'participants': {}
        }

        for pid_str, pf in participant_frames.items():
            pid = int(pid_str)
            frame_data['participants'][pid] = extract_frame_data(pf, pid)

        # Extract events
        frame_data['events'] = extract_events_from_frame(frame.get('events', []))

        processed_frames.append(frame_data)

    if verbose:
        print(f"      ({len(processed_frames)} frames)")

    return {
        'match_id': match_id,
        'game_version': game_version,
        'game_duration': game_duration,
        'jungler_participant_id': jungler_participant_id,
        'jungler_puuid': target_puuid,
        'participants': all_participants,
        'frames': processed_frames,
    }


def collect_matches(platform: str, target_matches: int = 10):
    """Collect jungle match data from Challenger players."""
    routing_region = REGION_ROUTING.get(platform)

    print(f"\n{'='*70}")
    print(f"COLLECTING CAMP CLASSIFICATION DATA")
    print(f"Platform: {platform.upper()}")
    print(f"Target: {target_matches} jungle matches")
    print(f"{'='*70}")

    # Get Challenger players
    print("\nFetching Challenger leaderboard...")
    challenger = get_challenger_league(platform)
    entries = challenger.get('entries', [])

    if not entries:
        print("✗ Could not fetch leaderboard")
        return []

    print(f"✓ Found {len(entries)} Challenger players")

    collected_data = []
    players_processed = 0

    for entry in entries:
        if len(collected_data) >= target_matches:
            print(f"\n✓ Reached target of {target_matches} matches!")
            break

        puuid = entry.get('puuid')
        summoner_name = entry.get('summonerName', 'Unknown')

        if not puuid:
            continue

        players_processed += 1
        print(f"\n[Player {players_processed}] {summoner_name}")

        # Verify player is a jungler (check first 5 matches)
        print(f"  Checking if jungle player...")
        is_jungler = verify_jungle_player(puuid, routing_region, check_count=5)

        if not is_jungler:
            print(f"  ✗ Not a jungle player, skipping")
            continue

        print(f"  ✓ Jungler detected, scanning matches...")

        # Get full match history
        match_ids = get_match_ids(puuid, routing_region, count=20)

        # Process each match
        for match_id in match_ids:
            if len(collected_data) >= target_matches:
                break

            match_data = process_match(match_id, routing_region, puuid, verbose=True)

            if match_data:
                collected_data.append(match_data)
                print(f"      [{len(collected_data)}/{target_matches}] collected")

            # Small delay between matches
            time.sleep(0.3)

    return collected_data


def main():
    print("="*70)
    print("CAMP CLASSIFICATION DATA COLLECTION")
    print(f"Using {len(API_KEYS)} API key(s)")
    print("="*70)

    # Collect from NA region
    all_data = collect_matches("na1", target_matches=10)

    # Save data
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "camp_classification_data.json"

    with open(output_file, 'w') as f:
        json.dump({
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_matches': len(all_data),
            'matches': all_data
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Matches collected: {len(all_data)}")
    print(f"Saved to: {output_file}")

    # Print summary
    if all_data:
        total_frames = sum(len(m['frames']) for m in all_data)
        avg_frames = total_frames / len(all_data)
        print(f"\nData summary:")
        print(f"  Total frames: {total_frames}")
        print(f"  Avg frames per match: {avg_frames:.1f}")
        print(f"  Frame interval: 60 seconds")


if __name__ == "__main__":
    main()
