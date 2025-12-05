"""
Improved camp inference using jungle minion count + position.

Strategy:
- jungle_minions_delta tells us HOW MANY monsters were killed
- Position tells us WHERE (which camp)
- Gold is too noisy (includes passive income, CS, assists)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import json
from pathlib import Path


# Jungle minion counts per camp
CAMP_MINION_COUNTS = {
    # Blue side
    "blue_blue": 2,      # Blue Sentinel + 1 small
    "blue_gromp": 1,     # Gromp
    "blue_wolves": 3,    # Greater + 2 small
    "blue_raptors": 6,   # Large + 5 small
    "blue_red": 2,       # Red Brambleback + 1 small
    "blue_krugs": 6,     # Ancient + splits (approximate)

    # Red side
    "red_red": 2,
    "red_krugs": 6,
    "red_raptors": 6,
    "red_wolves": 3,
    "red_gromp": 1,
    "red_blue": 2,

    # Neutral
    "dragon": 1,
    "baron": 1,
    "herald": 1,
    "scuttle_bot": 1,
    "scuttle_top": 1,
}

# Camp positions (from your simulation)
CAMP_POSITIONS = {
    # Blue side
    "blue_blue": (3850, 7900),
    "blue_gromp": (2150, 8500),
    "blue_wolves": (3500, 6500),
    "blue_raptors": (7000, 5500),
    "blue_red": (7900, 4150),
    "blue_krugs": (8500, 2850),

    # Red side
    "red_red": (6850, 10750),
    "red_krugs": (6300, 12050),
    "red_raptors": (7750, 9350),
    "red_wolves": (11150, 8400),
    "red_gromp": (12700, 6400),
    "red_blue": (10950, 6950),

    # Neutral
    "dragon": (9850, 4350),
    "baron": (5150, 10450),
    "herald": (5150, 10450),
    "scuttle_bot": (10500, 5000),
    "scuttle_top": (4500, 9750),
}


@dataclass
class CampClear:
    """Represents an inferred jungle camp clear."""
    timestamp: float
    participant_id: int
    camp_name: str
    minions_killed: int
    position: Tuple[int, int]
    confidence: float


def distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def find_nearest_camps(position: Tuple[int, int],
                       minions_killed: int,
                       max_distance: float = 3000) -> List[Tuple[str, float]]:
    """
    Find camps that match:
    1. Position (within max_distance)
    2. Minion count (exact or close)

    Returns list of (camp_name, confidence_score) sorted by confidence.
    """
    candidates = []

    for camp_name, camp_pos in CAMP_POSITIONS.items():
        dist = distance(position, camp_pos)

        if dist > max_distance:
            continue

        # Check minion count match
        expected_minions = CAMP_MINION_COUNTS.get(camp_name, 0)
        minion_diff = abs(minions_killed - expected_minions)

        # Position confidence (1.0 at camp center, 0.0 at max_distance)
        pos_conf = max(0, 1 - (dist / max_distance))

        # Minion count confidence
        if minion_diff == 0:
            minion_conf = 1.0
        elif minion_diff <= 2:  # Allow some tolerance
            minion_conf = 0.7
        else:
            minion_conf = 0.3

        # Overall confidence
        confidence = 0.7 * pos_conf + 0.3 * minion_conf

        candidates.append((camp_name, confidence))

    # Sort by confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def infer_camp_clear_v2(prev_frame: dict,
                        curr_frame: dict,
                        participant_id: int,
                        min_confidence: float = 0.4) -> Optional[CampClear]:
    """
    Infer camp clear using minion count + position.
    """
    prev_data = prev_frame['participantFrames'].get(str(participant_id))
    curr_data = curr_frame['participantFrames'].get(str(participant_id))

    if not prev_data or not curr_data:
        return None

    # Check if jungle minions increased
    jg_delta = curr_data['jungleMinionsKilled'] - prev_data['jungleMinionsKilled']

    if jg_delta <= 0:
        return None

    # Get position
    position = (curr_data['position']['x'], curr_data['position']['y'])

    # Find matching camps
    candidates = find_nearest_camps(position, jg_delta)

    if not candidates or candidates[0][1] < min_confidence:
        return None

    camp_name, confidence = candidates[0]

    return CampClear(
        timestamp=curr_frame['timestamp'] / 1000,
        participant_id=participant_id,
        camp_name=camp_name,
        minions_killed=jg_delta,
        position=position,
        confidence=confidence
    )


def extract_jungle_path_v2(timeline: dict,
                           participant_id: int,
                           min_confidence: float = 0.4) -> List[CampClear]:
    """Extract full jungle path."""
    frames = timeline.get('info', {}).get('frames', [])
    jungle_path = []

    for i in range(1, len(frames)):
        camp_clear = infer_camp_clear_v2(frames[i-1], frames[i],
                                         participant_id, min_confidence)
        if camp_clear:
            jungle_path.append(camp_clear)

    return jungle_path


def print_jungle_path(jungle_path: List[CampClear], participant_id: int):
    """Pretty print jungle path."""
    print(f"\n{'='*70}")
    print(f"JUNGLE PATH - Participant {participant_id}")
    print(f"{'='*70}")
    print(f"{'Time':<8} {'Camp':<20} {'Minions':<8} {'Position':<20} {'Conf':<6}")
    print(f"{'-'*70}")

    for clear in jungle_path:
        time_str = f"{int(clear.timestamp//60)}:{int(clear.timestamp%60):02d}"
        pos_str = f"({clear.position[0]}, {clear.position[1]})"
        print(f"{time_str:<8} {clear.camp_name:<20} {clear.minions_killed:<8} "
              f"{pos_str:<20} {clear.confidence:.2f}")

    print(f"{'='*70}")
    print(f"Total camps cleared: {len(jungle_path)}\n")


if __name__ == "__main__":
    # Test on sample match
    timeline_file = Path(__file__).parent.parent / "data" / "raw" / "timeline_NA1_5089218526.json"

    if not timeline_file.exists():
        print("No sample data. Run: python scripts/fetch_sample_match.py")
        exit(1)

    with open(timeline_file) as f:
        timeline = json.load(f)

    print("Analyzing jungle paths from match...\n")

    # Try all participants, print those with significant jungle activity
    for participant_id in range(1, 11):
        jungle_path = extract_jungle_path_v2(timeline, participant_id, min_confidence=0.3)

        if len(jungle_path) >= 5:  # Likely a jungler
            print_jungle_path(jungle_path, participant_id)
