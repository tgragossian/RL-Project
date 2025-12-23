"""
Data collection script to connect powerspike CSV with actual match data.

Fetches match data from Riot API and extracts:
- Champion levels at different time points
- Items completed at different time points
- Gold earned at different time points
- Powerspike scores calculated from the CSV

This creates the training dataset for learning when champions are strong/weak.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from powerspike_system import PowerspikeSystem

# API Configuration
API_KEY = "RGAPI-c6aee0b2-4405-465a-a1fb-507e754538d6"

REGION_ROUTING = {
    "na1": "americas",
    "euw1": "europe",
    "kr": "asia",
}

REQUEST_DELAY = 0.05  # 20 req/sec


@dataclass
class ChampionSnapshot:
    """Champion state at a specific timestamp."""
    timestamp: int  # Game time in milliseconds
    participant_id: int
    champion: str
    level: int
    gold_earned: int
    current_gold: int
    items: List[str]  # List of item names
    position: Tuple[int, int]  # (x, y) position
    # Powerspike scores
    level_spike_score: float
    item_spike_score: float
    overall_spike_score: float


@dataclass
class MatchPowerspikeData:
    """Complete powerspike data for a match."""
    match_id: str
    game_duration: int
    snapshots: List[ChampionSnapshot]


def rate_limit():
    """Rate limiting for API requests."""
    time.sleep(REQUEST_DELAY)


def get_match_info(match_id: str, routing_region: str) -> dict:
    """Get match info from Riot API."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"✗ Error fetching match {match_id}: {response.status_code}")
        return {}


def get_match_timeline(match_id: str, routing_region: str) -> dict:
    """Get match timeline from Riot API."""
    url = f"https://{routing_region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": API_KEY}

    rate_limit()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"✗ Error fetching timeline {match_id}: {response.status_code}")
        return {}


# Item ID to name mapping (common items)
ITEM_MAP = {
    # Jungle items
    3862: "Scorched Earth",
    3863: "Lightning Strike",
    3864: "Gustwalker",

    # Mythic/Legendary items - Assassin
    6692: "Eclipse",
    6693: "Prowler's Claw",
    3071: "Black Cleaver",
    3156: "Maw of Malmortius",
    3053: "Sterak's Gage",
    3179: "Umbral Glaive",
    3814: "Edge of Night",

    # Mythic/Legendary - Mage
    6653: "Liandry's Torment",
    4646: "Stormsurge",
    3152: "Hextech Rocketbelt",
    3165: "Morellonomicon",
    3135: "Void Staff",
    3089: "Rabadon's Deathcap",
    3116: "Rylai's Crystal Scepter",
    3003: "Archangel's Staff",
    3040: "Seraph's Embrace",
    6655: "Luden's Companion",
    3118: "Malignance",
    4005: "Imperial Mandate",
    6657: "Rod of Ages",

    # Mythic/Legendary - Fighter
    3078: "Trinity Force",
    6630: "Goredrinker",
    6631: "Stridebreaker",
    3161: "Spear of Shojin",
    3074: "Ravenous Hydra",
    3748: "Titanic Hydra",
    3153: "Blade of the Ruined King",
    6632: "Divine Sunderer",
    3142: "Youmuu's Ghostblade",
    7000: "Sandshrike's Claw",
    7001: "Syzygy",
    7002: "Draktharr's Shadowcarver",
    6676: "The Collector",
    3036: "Lord Dominik's Regards",
    3033: "Mortal Reminder",

    # Mythic/Legendary - Tank
    3068: "Sunfire Aegis",
    6664: "Turbo Chemtank",
    3190: "Locket of the Iron Solari",
    3109: "Knight's Vow",
    3001: "Abyssal Mask",
    3065: "Spirit Visage",
    3075: "Thornmail",
    3742: "Dead Man's Plate",
    3143: "Randuin's Omen",
    3110: "Frozen Heart",
    4401: "Force of Nature",
    3222: "Mikael's Blessing",
    6667: "Radiant Virtue",
    3119: "Winter's Approach",
    3121: "Fimbulwinter",

    # Mythic/Legendary - ADC
    6672: "Kraken Slayer",
    6671: "Galeforce",
    3031: "Infinity Edge",
    3094: "Rapid Firecannon",
    3087: "Statikk Shiv",
    3046: "Phantom Dancer",
    3508: "Essence Reaver",
    3091: "Wit's End",
    6675: "Navori Quickblades",
    3085: "Runaan's Hurricane",
    6673: "Immortal Shieldbow",

    # Boots
    3006: "Berserker's Greaves",
    3020: "Sorcerer's Shoes",
    3047: "Plated Steelcaps",
    3111: "Mercury's Treads",
    3009: "Swiftness Boots",
    3158: "Ionian Boots of Lucidity",
    3117: "Mobility Boots",

    # Component items
    3134: "Serrated Dirk",
    3133: "Caulfield's Warhammer",
    1037: "Pickaxe",
    1038: "B.F. Sword",
    1053: "Vampiric Scepter",
    1018: "Cloak of Agility",
    3086: "Zeal",
    3051: "Hearthbound Axe",
    3044: "Phage",
    3057: "Sheen",
    1026: "Blasting Wand",
    1058: "Needlessly Large Rod",
    3145: "Hextech Alternator",
    3108: "Fiendish Codex",
    3802: "Lost Chapter",
    1052: "Amplifying Tome",
    1028: "Ruby Crystal",
    1011: "Giant's Belt",
    3066: "Winged Moonplate",
    1031: "Chain Vest",
    1029: "Cloth Armor",
    1033: "Null-Magic Mantle",
    1057: "Negatron Cloak",
    3067: "Kindlegem",
    3024: "Glacial Buckler",
    3082: "Warden's Mail",
}


def get_item_name(item_id: int) -> str:
    """Convert item ID to readable name."""
    return ITEM_MAP.get(item_id, f"Item_{item_id}")


def extract_champion_snapshots(
    match_info: dict,
    timeline: dict,
    powerspike_system: PowerspikeSystem,
    snapshot_interval: int = 60000  # Every 60 seconds
) -> List[ChampionSnapshot]:
    """
    Extract champion snapshots at regular intervals throughout the game.

    Args:
        match_info: Match info from Riot API
        timeline: Match timeline from Riot API
        powerspike_system: PowerspikeSystem instance for calculating scores
        snapshot_interval: How often to take snapshots (milliseconds)

    Returns:
        List of ChampionSnapshot objects
    """
    snapshots = []

    # Get participant info (for champion names)
    participants_info = match_info.get('info', {}).get('participants', [])
    participant_map = {
        p.get('participantId'): p.get('championName')
        for p in participants_info
    }

    game_duration = match_info.get('info', {}).get('gameDuration', 0) * 1000  # Convert to ms

    # Process timeline frames
    frames = timeline.get('info', {}).get('frames', [])

    for frame in frames:
        timestamp = frame.get('timestamp', 0)

        # Only take snapshots at our intervals
        if timestamp % snapshot_interval != 0 and timestamp != 0:
            continue

        participant_frames = frame.get('participantFrames', {})

        for participant_id_str, pframe in participant_frames.items():
            participant_id = int(participant_id_str)
            champion = participant_map.get(participant_id, "Unknown")

            level = pframe.get('level', 1)
            gold_earned = pframe.get('totalGold', 0)
            current_gold = pframe.get('currentGold', 0)

            # Get position
            position_data = pframe.get('position', {})
            position = (position_data.get('x', 0), position_data.get('y', 0))

            # Get items - this is a list of item IDs
            item_ids = [
                pframe.get(f'item{i}', 0)
                for i in range(7)  # 0-6 item slots
            ]
            items = [get_item_name(item_id) for item_id in item_ids if item_id != 0]

            # Calculate powerspike scores
            level_score = powerspike_system.calculate_level_spike_score(champion, level)
            item_score = powerspike_system.calculate_item_spike_score(champion, items)
            overall_score = powerspike_system.calculate_overall_spike_score(champion, level, items)

            snapshot = ChampionSnapshot(
                timestamp=timestamp,
                participant_id=participant_id,
                champion=champion,
                level=level,
                gold_earned=gold_earned,
                current_gold=current_gold,
                items=items,
                position=position,
                level_spike_score=level_score,
                item_spike_score=item_score,
                overall_spike_score=overall_score
            )

            snapshots.append(snapshot)

    return snapshots


def collect_powerspike_data_from_match(
    match_id: str,
    routing_region: str,
    powerspike_system: PowerspikeSystem
) -> Optional[MatchPowerspikeData]:
    """
    Collect complete powerspike data for a single match.

    Args:
        match_id: Match ID to collect
        routing_region: Routing region (americas, europe, asia)
        powerspike_system: PowerspikeSystem instance

    Returns:
        MatchPowerspikeData or None if failed
    """
    # Get match info and timeline
    match_info = get_match_info(match_id, routing_region)
    if not match_info:
        return None

    timeline = get_match_timeline(match_id, routing_region)
    if not timeline:
        return None

    game_duration = match_info.get('info', {}).get('gameDuration', 0)

    # Extract snapshots
    snapshots = extract_champion_snapshots(match_info, timeline, powerspike_system)

    if not snapshots:
        return None

    return MatchPowerspikeData(
        match_id=match_id,
        game_duration=game_duration,
        snapshots=snapshots
    )


def collect_from_match_list(
    match_ids: List[str],
    routing_region: str,
    powerspike_csv_path: Path,
    output_path: Path
):
    """
    Collect powerspike data from a list of match IDs.

    Args:
        match_ids: List of match IDs to process
        routing_region: Routing region for API calls
        powerspike_csv_path: Path to powerspike CSV
        output_path: Where to save the collected data
    """
    # Load powerspike system
    print("Loading powerspike system...")
    powerspike_system = PowerspikeSystem(powerspike_csv_path)
    print(f"✓ Loaded powerspike data for {len(powerspike_system.champion_spikes)} champions\n")

    collected_matches = []

    print(f"{'='*70}")
    print(f"Collecting powerspike data from {len(match_ids)} matches")
    print(f"{'='*70}\n")

    for i, match_id in enumerate(match_ids):
        print(f"[{i+1}/{len(match_ids)}] Processing {match_id}...")

        match_data = collect_powerspike_data_from_match(
            match_id,
            routing_region,
            powerspike_system
        )

        if match_data:
            collected_matches.append(match_data)
            snapshot_count = len(match_data.snapshots)
            duration_min = match_data.game_duration // 60
            print(f"  ✓ Collected {snapshot_count} snapshots ({duration_min}m game)\n")
        else:
            print(f"  ✗ Failed to collect data\n")

    # Save data
    print(f"{'='*70}")
    print(f"Saving collected data...")
    print(f"{'='*70}\n")

    # Convert to JSON-serializable format
    output_data = {
        'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_matches': len(collected_matches),
        'total_snapshots': sum(len(m.snapshots) for m in collected_matches),
        'matches': [
            {
                'match_id': m.match_id,
                'game_duration': m.game_duration,
                'snapshots': [asdict(s) for s in m.snapshots]
            }
            for m in collected_matches
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Collection complete!")
    print(f"Matches: {len(collected_matches)}")
    print(f"Total snapshots: {output_data['total_snapshots']}")
    print(f"Saved to: {output_path}")


def main():
    """Main entry point - collect data from existing challenger jungle data."""

    # Load existing jungle match data
    jungle_data_path = Path(__file__).parent.parent / "data" / "processed" / "challenger_jungle_data.json"

    if not jungle_data_path.exists():
        print(f"❌ Jungle data not found at {jungle_data_path}")
        print("Run collect_from_leaderboard.py first to collect jungle matches!")
        return

    with open(jungle_data_path, 'r') as f:
        jungle_data = json.load(f)

    # Extract match IDs grouped by region
    matches_by_region = {
        'americas': [],
        'europe': [],
        'asia': []
    }

    for player in jungle_data.get('players', []):
        platform = player.get('platform', 'na1')
        routing_region = REGION_ROUTING.get(platform, 'americas')

        for match in player.get('matches', []):
            match_id = match.get('match_id')
            if match_id and match_id not in matches_by_region[routing_region]:
                matches_by_region[routing_region].append(match_id)

    total_matches = sum(len(matches) for matches in matches_by_region.values())
    print(f"Found {total_matches} unique jungle matches:")
    for region, matches in matches_by_region.items():
        print(f"  {region}: {len(matches)} matches")
    print()

    # Paths
    powerspike_csv = Path(__file__).parent.parent / "data" / "raw" / "Champ Powerspikes _ Updated Patch 14.21 - Powerspikes.csv"
    powerspike_system = PowerspikeSystem(powerspike_csv)

    # Collect from all regions
    all_collected = []

    for region, match_ids in matches_by_region.items():
        if not match_ids:
            continue

        print(f"\n{'='*70}")
        print(f"Processing {region.upper()} region ({len(match_ids)} matches)")
        print(f"{'='*70}\n")

        # Process matches for this region
        for i, match_id in enumerate(match_ids[:20]):  # Limit to 20 per region for now
            print(f"[{i+1}/{min(20, len(match_ids))}] Processing {match_id}...")

            match_data = collect_powerspike_data_from_match(
                match_id,
                region,
                powerspike_system
            )

            if match_data:
                all_collected.append(match_data)
                snapshot_count = len(match_data.snapshots)
                duration_min = match_data.game_duration // 60
                print(f"  ✓ Collected {snapshot_count} snapshots ({duration_min}m game)\n")
            else:
                print(f"  ✗ Failed to collect data\n")

    # Save combined data
    output_path = Path(__file__).parent.parent / "data" / "processed" / "powerspike_match_data.json"
    output_data = {
        'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_matches': len(all_collected),
        'total_snapshots': sum(len(m.snapshots) for m in all_collected),
        'matches': [
            {
                'match_id': m.match_id,
                'game_duration': m.game_duration,
                'snapshots': [asdict(s) for s in m.snapshots]
            }
            for m in all_collected
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Matches: {len(all_collected)}")
    print(f"Total snapshots: {output_data['total_snapshots']}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
