"""
Infer jungle camp clears from Riot API timeline data.

Uses gold change + position to determine which camps were cleared.
Each camp has a unique gold reward, making identification robust.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


# Gold rewards for each camp (from monster_scaling.py)
CAMP_GOLD_REWARDS = {
    "Blue Sentinel": 90,
    "Red Brambleback": 90,
    "Gromp": 80,
    "Murk Wolves": 55,      # Greater Wolf gives 55
    "Crimson Raptor": 35,   # Large Raptor gives 35 (+ 5 small ones)
    "Ancient Krug": 15,     # Per Krug (multiple split)
    "Rift Scuttler": 55,    # Can scale to 121
}

# Approximate camp positions on Summoner's Rift
# Blue side jungle (bottom-left)
BLUE_SIDE_CAMPS = {
    "blue_blue": (3850, 7900),      # Blue Buff
    "blue_gromp": (2150, 8500),     # Gromp
    "blue_wolves": (3500, 6500),    # Wolves
    "blue_raptors": (7000, 5500),   # Raptors
    "blue_red": (7900, 4150),       # Red Buff
    "blue_krugs": (8500, 2850),     # Krugs
}

# Red side jungle (top-right)
RED_SIDE_CAMPS = {
    "red_red": (6850, 10750),       # Red Buff
    "red_krugs": (6300, 12050),     # Krugs
    "red_raptors": (7750, 9350),    # Raptors
    "red_wolves": (11150, 8400),    # Wolves
    "red_gromp": (12700, 6400),     # Gromp
    "red_blue": (10950, 6950),      # Blue Buff
}

# Neutral
NEUTRAL_CAMPS = {
    "dragon": (9850, 4350),         # Dragon pit
    "baron": (5150, 10450),         # Baron pit
    "herald": (5150, 10450),        # Same as Baron
    "scuttle_bot": (10500, 5000),   # Bot scuttle
    "scuttle_top": (4500, 9750),    # Top scuttle
}

ALL_CAMPS = {**BLUE_SIDE_CAMPS, **RED_SIDE_CAMPS, **NEUTRAL_CAMPS}


@dataclass
class CampClear:
    """Represents an inferred jungle camp clear."""
    timestamp: float        # Game time in seconds
    participant_id: int     # Which player (1-10)
    camp_name: str          # Camp identifier
    camp_type: str          # "Blue Sentinel", "Gromp", etc.
    gold_gained: int        # Gold from this clear
    position: Tuple[int, int]  # (x, y) position
    confidence: float       # 0-1 confidence score


def distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def identify_camp_by_position(position: Tuple[int, int],
                                max_distance: float = 2000) -> Optional[str]:
    """
    Identify which camp the player is near based on position.

    Args:
        position: (x, y) coordinates
        max_distance: Maximum distance to consider "near" a camp

    Returns:
        Camp name or None if not near any camp
    """
    closest_camp = None
    closest_dist = float('inf')

    for camp_name, camp_pos in ALL_CAMPS.items():
        dist = distance(position, camp_pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_camp = camp_name

    if closest_dist <= max_distance:
        return closest_camp
    return None


def identify_camp_by_gold(gold_gained: int,
                          tolerance: int = 5) -> List[str]:
    """
    Identify possible camps based on gold gained.

    Args:
        gold_gained: Amount of gold earned
        tolerance: Allow +/- this much variation

    Returns:
        List of possible camp types
    """
    candidates = []

    for camp_type, camp_gold in CAMP_GOLD_REWARDS.items():
        if abs(gold_gained - camp_gold) <= tolerance:
            candidates.append(camp_type)

    # Special cases
    # Raptors: Large (35g) + 5 small (~5g each) = ~60g total
    if 55 <= gold_gained <= 65:
        candidates.append("Crimson Raptor (full camp)")

    # Krugs: Multiple splits, varies widely
    if 30 <= gold_gained <= 100:
        candidates.append("Ancient Krug (partial)")

    # Scuttle can scale
    if 55 <= gold_gained <= 125:
        if "Rift Scuttler" not in candidates:
            candidates.append("Rift Scuttler")

    return candidates


def infer_camp_clear(prev_frame: dict,
                     curr_frame: dict,
                     participant_id: int) -> Optional[CampClear]:
    """
    Infer if a camp was cleared between two frames.

    Args:
        prev_frame: Previous timeline frame
        curr_frame: Current timeline frame
        participant_id: Which player to analyze (1-10)

    Returns:
        CampClear object if a camp was cleared, else None
    """
    # Get participant data
    prev_data = prev_frame['participantFrames'].get(str(participant_id))
    curr_data = curr_frame['participantFrames'].get(str(participant_id))

    if not prev_data or not curr_data:
        return None

    # Calculate changes
    gold_delta = curr_data['totalGold'] - prev_data['totalGold']
    jg_minions_delta = curr_data['jungleMinionsKilled'] - prev_data['jungleMinionsKilled']

    # Must have killed jungle minions
    if jg_minions_delta <= 0:
        return None

    # Get position (use current position where clear happened)
    position = (curr_data['position']['x'], curr_data['position']['y'])

    # Identify camp by position
    camp_name = identify_camp_by_position(position)

    # Identify possible camp types by gold
    camp_type_candidates = identify_camp_by_gold(gold_delta)

    if not camp_name or not camp_type_candidates:
        return None

    # Determine most likely camp type
    # For now, take first candidate (could improve with better heuristics)
    camp_type = camp_type_candidates[0]

    # Calculate confidence based on:
    # 1. How close to a camp position
    # 2. How well gold matches expected
    # 3. Reasonable jungle minion count for that camp
    camp_pos = ALL_CAMPS.get(camp_name)
    if camp_pos:
        pos_dist = distance(position, camp_pos)
        pos_confidence = max(0, 1 - (pos_dist / 2000))  # 1.0 at camp, 0.0 at 2000 units
    else:
        pos_confidence = 0.5

    # Gold confidence
    expected_gold = CAMP_GOLD_REWARDS.get(camp_type.split('(')[0].strip(), 0)
    gold_diff = abs(gold_delta - expected_gold)
    gold_confidence = max(0, 1 - (gold_diff / 50))

    # Overall confidence (weighted average)
    confidence = 0.6 * pos_confidence + 0.4 * gold_confidence

    return CampClear(
        timestamp=curr_frame['timestamp'] / 1000,  # Convert ms to seconds
        participant_id=participant_id,
        camp_name=camp_name,
        camp_type=camp_type,
        gold_gained=gold_delta,
        position=position,
        confidence=confidence
    )


def extract_jungle_path(timeline: dict,
                        participant_id: int,
                        min_confidence: float = 0.5) -> List[CampClear]:
    """
    Extract the full jungle pathing for a player from timeline.

    Args:
        timeline: Full match timeline from Riot API
        participant_id: Which player (1-10, typically 1-5 are blue side)
        min_confidence: Minimum confidence to include a clear

    Returns:
        List of CampClear objects in chronological order
    """
    frames = timeline.get('info', {}).get('frames', [])
    jungle_path = []

    for i in range(1, len(frames)):
        prev_frame = frames[i-1]
        curr_frame = frames[i]

        camp_clear = infer_camp_clear(prev_frame, curr_frame, participant_id)

        if camp_clear and camp_clear.confidence >= min_confidence:
            jungle_path.append(camp_clear)

    return jungle_path


def print_jungle_path(jungle_path: List[CampClear], participant_id: int):
    """Pretty print a jungler's pathing."""
    print(f"\n{'='*70}")
    print(f"JUNGLE PATH - Participant {participant_id}")
    print(f"{'='*70}")
    print(f"{'Time':<8} {'Camp':<20} {'Type':<25} {'Gold':<6} {'Conf':<5}")
    print(f"{'-'*70}")

    for clear in jungle_path:
        time_str = f"{int(clear.timestamp//60)}:{int(clear.timestamp%60):02d}"
        print(f"{time_str:<8} {clear.camp_name:<20} {clear.camp_type:<25} "
              f"{clear.gold_gained:<6} {clear.confidence:.2f}")

    print(f"{'='*70}")
    print(f"Total camps cleared: {len(jungle_path)}")
    print()


if __name__ == "__main__":
    # Test with the sample match we downloaded
    import json
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data" / "raw"
    timeline_file = data_dir / "timeline_NA1_5089218526.json"

    if timeline_file.exists():
        print("Loading sample match timeline...")
        with open(timeline_file) as f:
            timeline = json.load(f)

        # Analyze both junglers
        # Typically participants 1-5 are blue side, 6-10 are red side
        # Junglers are usually role 3 or 4, but let's try both teams

        print("Extracting jungle paths...")

        for participant_id in range(1, 11):
            jungle_path = extract_jungle_path(timeline, participant_id, min_confidence=0.3)

            # Only print if they cleared camps (i.e., they're the jungler)
            if len(jungle_path) > 5:  # Arbitrary threshold
                print_jungle_path(jungle_path, participant_id)
    else:
        print(f"No sample data found at {timeline_file}")
        print("Run: python scripts/fetch_sample_match.py first")
