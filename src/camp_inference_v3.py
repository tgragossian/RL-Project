"""
Final camp inference using CS (creep score) + position.

KEY INSIGHT: Each camp gives exactly 4 CS, regardless of how many monsters.
This is much more robust than counting individual jungle minions!

From League tracking guides:
- Each jungle camp = 4 CS
- Scuttle = 4 CS
- Epic monsters (Dragon/Baron/Herald) = 4 CS

This means partial clears don't matter - only completed camps show up in CS.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import json
from pathlib import Path


# Camp positions on Summoner's Rift
CAMP_POSITIONS = {
    # Blue side jungle
    "blue_blue": (3850, 7900),       # Blue Buff
    "blue_gromp": (2150, 8500),      # Gromp
    "blue_wolves": (3500, 6500),     # Wolves
    "blue_raptors": (7000, 5500),    # Raptors
    "blue_red": (7900, 4150),        # Red Buff
    "blue_krugs": (8500, 2850),      # Krugs

    # Red side jungle
    "red_red": (6850, 10750),        # Red Buff
    "red_krugs": (6300, 12050),      # Krugs
    "red_raptors": (7750, 9350),     # Raptors
    "red_wolves": (11150, 8400),     # Wolves
    "red_gromp": (12700, 6400),      # Gromp
    "red_blue": (10950, 6950),       # Blue Buff

    # Neutral objectives
    "dragon": (9850, 4350),          # Dragon pit
    "baron": (5150, 10450),          # Baron pit
    "herald": (5150, 10450),         # Herald (same as Baron)
    "scuttle_bot": (10500, 5000),    # Bot scuttle
    "scuttle_top": (4500, 9750),     # Top scuttle
}


@dataclass
class CampClear:
    """Represents an inferred jungle camp clear."""
    timestamp: float              # Game time in seconds
    participant_id: int           # Which player (1-10)
    camp_name: str               # Camp identifier (e.g., "blue_blue")
    camps_cleared: int           # Number of camps (CS_delta / 4)
    position: Tuple[int, int]    # (x, y) position
    confidence: float            # 0-1 confidence score


def distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def find_nearest_camp(position: Tuple[int, int],
                      max_distance: float = 3000) -> Tuple[Optional[str], float]:
    """
    Find the nearest camp to a given position.

    Args:
        position: (x, y) coordinates
        max_distance: Maximum distance to consider valid

    Returns:
        (camp_name, confidence) or (None, 0.0) if no camp nearby
    """
    closest_camp = None
    closest_dist = float('inf')

    for camp_name, camp_pos in CAMP_POSITIONS.items():
        dist = distance(position, camp_pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_camp = camp_name

    if closest_dist <= max_distance:
        # Confidence decreases linearly with distance
        confidence = max(0, 1 - (closest_dist / max_distance))
        return closest_camp, confidence

    return None, 0.0


def infer_camp_clear_cs(prev_frame: dict,
                        curr_frame: dict,
                        participant_id: int,
                        min_confidence: float = 0.3) -> Optional[CampClear]:
    """
    Infer camp clear using CS delta + position.

    Each camp = 4 CS, so:
    - CS delta of 4 = 1 camp cleared
    - CS delta of 8 = 2 camps cleared
    - CS delta of 12 = 3 camps cleared (rare but possible)

    Args:
        prev_frame: Previous timeline frame
        curr_frame: Current timeline frame
        participant_id: Which player (1-10)
        min_confidence: Minimum confidence threshold

    Returns:
        CampClear object or None
    """
    prev_data = prev_frame['participantFrames'].get(str(participant_id))
    curr_data = curr_frame['participantFrames'].get(str(participant_id))

    if not prev_data or not curr_data:
        return None

    # Calculate CS change (total minions + neutral minions)
    # Note: In Riot API, jungle CS is part of minionsKilled
    prev_cs = prev_data['minionsKilled'] + prev_data['jungleMinionsKilled']
    curr_cs = curr_data['minionsKilled'] + curr_data['jungleMinionsKilled']
    cs_delta = curr_cs - prev_cs

    # Check if they cleared jungle camps (each camp = 4 CS)
    # We look for CS increases that are multiples of 4 (or close to it)
    # Some tolerance for edge cases
    if cs_delta < 3:  # Less than 1 camp worth
        return None

    # Estimate number of camps cleared
    camps_cleared = round(cs_delta / 4)

    # If they cleared camps but we don't detect it as a multiple of 4,
    # they might have also killed lane minions. Let's check jungle minions.
    jg_delta = curr_data['jungleMinionsKilled'] - prev_data['jungleMinionsKilled']

    # If no jungle activity, skip
    if jg_delta <= 0:
        return None

    # Get position where clear happened
    position = (curr_data['position']['x'], curr_data['position']['y'])

    # Find nearest camp
    camp_name, pos_confidence = find_nearest_camp(position)

    if not camp_name or pos_confidence < min_confidence:
        return None

    return CampClear(
        timestamp=curr_frame['timestamp'] / 1000,
        participant_id=participant_id,
        camp_name=camp_name,
        camps_cleared=camps_cleared,
        position=position,
        confidence=pos_confidence
    )


def extract_jungle_path_cs(timeline: dict,
                           participant_id: int,
                           min_confidence: float = 0.3) -> List[CampClear]:
    """
    Extract full jungle pathing from timeline using CS-based inference.

    Args:
        timeline: Full match timeline from Riot API
        participant_id: Which player (1-10)
        min_confidence: Minimum confidence threshold

    Returns:
        List of CampClear objects in chronological order
    """
    frames = timeline.get('info', {}).get('frames', [])
    jungle_path = []

    for i in range(1, len(frames)):
        camp_clear = infer_camp_clear_cs(frames[i-1], frames[i],
                                         participant_id, min_confidence)
        if camp_clear:
            jungle_path.append(camp_clear)

    return jungle_path


def print_jungle_path(jungle_path: List[CampClear], participant_id: int):
    """Pretty print jungle pathing."""
    print(f"\n{'='*75}")
    print(f"JUNGLE PATH - Participant {participant_id}")
    print(f"{'='*75}")
    print(f"{'Time':<8} {'Camp':<20} {'#Camps':<8} {'Position':<20} {'Conf':<6}")
    print(f"{'-'*75}")

    for clear in jungle_path:
        time_str = f"{int(clear.timestamp//60)}:{int(clear.timestamp%60):02d}"
        pos_str = f"({clear.position[0]}, {clear.position[1]})"
        print(f"{time_str:<8} {clear.camp_name:<20} {clear.camps_cleared:<8} "
              f"{pos_str:<20} {clear.confidence:.2f}")

    print(f"{'='*75}")
    print(f"Total clears: {len(jungle_path)}")
    print(f"Total camps: {sum(c.camps_cleared for c in jungle_path)}")
    print()


def analyze_jungle_efficiency(jungle_path: List[CampClear]):
    """Analyze jungling efficiency metrics."""
    if not jungle_path:
        return

    total_time = jungle_path[-1].timestamp - jungle_path[0].timestamp
    total_camps = sum(c.camps_cleared for c in jungle_path)

    camps_per_minute = (total_camps / total_time) * 60 if total_time > 0 else 0

    print(f"{'='*75}")
    print(f"JUNGLE EFFICIENCY ANALYSIS")
    print(f"{'='*75}")
    print(f"Time span: {jungle_path[0].timestamp:.0f}s to {jungle_path[-1].timestamp:.0f}s")
    print(f"Duration: {total_time/60:.1f} minutes")
    print(f"Total camps cleared: {total_camps}")
    print(f"Camps per minute: {camps_per_minute:.2f}")
    print(f"Average confidence: {sum(c.confidence for c in jungle_path)/len(jungle_path):.2f}")
    print()


if __name__ == "__main__":
    # Test on sample match
    timeline_file = Path(__file__).parent.parent / "data" / "raw" / "timeline_NA1_5089218526.json"

    if not timeline_file.exists():
        print("No sample data. Run: python scripts/fetch_sample_match.py")
        exit(1)

    with open(timeline_file) as f:
        timeline = json.load(f)

    print("Analyzing jungle paths using CS-based inference...\n")

    # Analyze all participants
    for participant_id in range(1, 11):
        jungle_path = extract_jungle_path_cs(timeline, participant_id, min_confidence=0.3)

        # Only print if they have significant jungle activity
        if len(jungle_path) >= 5:
            print_jungle_path(jungle_path, participant_id)
            analyze_jungle_efficiency(jungle_path)
