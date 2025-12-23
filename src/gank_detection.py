"""
Gank detection from timeline positional data.

Detects when jungler is attempting to gank based on position in lane gank zones.
These zones are defined based on map_outlined.png - the blue areas between outer turrets
and river where junglers typically gank from.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


# Gank zones on Summoner's Rift (based on map_outlined.png)
# These are polygon regions where junglers are when ganking
# Coordinates are approximate based on standard 14820x14820 map

# Top lane gank zone (blue side - upper left)
TOP_GANK_ZONE = [
    # Polygon points defining the top gank region
    (1000, 9000), (5000, 11000), (5000, 13000), (1000, 13000)
]

# Mid lane gank zone (river area in center)
MID_GANK_ZONE = [
    # Polygon for mid gank region
    (4000, 6000), (9000, 11000), (11000, 9000), (6000, 4000)
]

# Bot lane gank zone (red side - lower right)
BOT_GANK_ZONE = [
    # Polygon for bot gank region
    (9000, 1000), (13000, 1000), (13000, 5000), (11000, 5000)
]


@dataclass
class GankAttempt:
    """Represents a detected gank attempt."""
    timestamp: float              # Game time in seconds
    participant_id: int           # Jungler ID
    lane: str                     # Which lane (top/mid/bot)
    position: Tuple[int, int]     # Where the gank occurred
    confidence: float             # 0-1 confidence score


def distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        point: (x, y) coordinate
        polygon: List of (x, y) vertices defining the polygon

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def get_gank_lane(position: Tuple[int, int]) -> Optional[str]:
    """
    Determine which lane the jungler is ganking based on position.

    Args:
        position: (x, y) coordinate

    Returns:
        "gank_top", "gank_mid", or "gank_bot" if in gank zone, None otherwise
    """
    if point_in_polygon(position, TOP_GANK_ZONE):
        return "gank_top"
    elif point_in_polygon(position, MID_GANK_ZONE):
        return "gank_mid"
    elif point_in_polygon(position, BOT_GANK_ZONE):
        return "gank_bot"
    return None


def detect_gank_from_frames(prev_frame: dict,
                            curr_frame: dict,
                            jungler_id: int,
                            min_confidence: float = 0.6) -> Optional[GankAttempt]:
    """
    Detect if jungler is ganking based on position in gank zones.

    A gank is detected when the jungler enters a gank zone (blue areas on map).

    Args:
        prev_frame: Previous timeline frame
        curr_frame: Current timeline frame
        jungler_id: Participant ID of jungler
        min_confidence: Minimum confidence threshold

    Returns:
        GankAttempt if detected, None otherwise
    """
    # Get participant frames
    curr_participants = {
        p['participantId']: p
        for p in curr_frame.get('participantFrames', {}).values()
    }

    prev_participants = {
        p['participantId']: p
        for p in prev_frame.get('participantFrames', {}).values()
    }

    if jungler_id not in curr_participants or jungler_id not in prev_participants:
        return None

    jungler_curr = curr_participants[jungler_id]
    jungler_prev = prev_participants[jungler_id]

    # Get positions
    curr_pos = (jungler_curr['position']['x'], jungler_curr['position']['y'])
    prev_pos = (jungler_prev['position']['x'], jungler_prev['position']['y'])

    # Check if currently in a gank zone
    curr_lane = get_gank_lane(curr_pos)
    if not curr_lane:
        return None  # Not in a gank zone

    # Check if was NOT in this gank zone before (entering gank zone)
    prev_lane = get_gank_lane(prev_pos)
    if prev_lane == curr_lane:
        return None  # Already in this gank zone, not a new gank

    # Calculate confidence (can be enhanced with more factors)
    confidence = 0.7  # Base confidence for being in gank zone

    if confidence < min_confidence:
        return None

    timestamp = curr_frame.get('timestamp', 0) / 1000.0  # Convert ms to seconds

    return GankAttempt(
        timestamp=timestamp,
        participant_id=jungler_id,
        lane=curr_lane,
        position=curr_pos,
        confidence=confidence
    )


def extract_ganks_from_timeline(timeline: dict,
                                jungler_id: int,
                                min_confidence: float = 0.6,
                                cooldown_seconds: float = 30) -> List[GankAttempt]:
    """
    Extract all gank attempts from a match timeline.

    Args:
        timeline: Full match timeline from Riot API
        jungler_id: Participant ID of jungler
        min_confidence: Minimum confidence threshold
        cooldown_seconds: Minimum time between ganks in same lane (prevents duplicates)

    Returns:
        List of GankAttempt objects
    """
    frames = timeline.get('info', {}).get('frames', [])
    ganks = []
    last_gank_time = {}  # Track last gank time per lane

    for i in range(1, len(frames)):
        gank = detect_gank_from_frames(
            frames[i-1], frames[i], jungler_id, min_confidence
        )

        if gank:
            # Check cooldown (prevent detecting same gank multiple times)
            last_time = last_gank_time.get(gank.lane, 0)
            if gank.timestamp - last_time >= cooldown_seconds:
                ganks.append(gank)
                last_gank_time[gank.lane] = gank.timestamp

    return ganks


def print_ganks(ganks: List[GankAttempt], participant_id: int):
    """Pretty print detected ganks."""
    print(f"\n{'='*70}")
    print(f"DETECTED GANKS - Participant {participant_id}")
    print(f"{'='*70}")
    print(f"{'Time':<8} {'Lane':<12} {'Position':<20} {'Conf':<6}")
    print(f"{'-'*70}")

    for gank in ganks:
        time_str = f"{int(gank.timestamp//60)}:{int(gank.timestamp%60):02d}"
        pos_str = f"({gank.position[0]}, {gank.position[1]})"
        print(f"{time_str:<8} {gank.lane:<12} {pos_str:<20} {gank.confidence:.2f}")

    print(f"{'='*70}")
    print(f"Total ganks: {len(ganks)}")
    print()


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Test on sample match
    timeline_file = Path(__file__).parent.parent / "data" / "raw" / "timeline_NA1_5089218526.json"

    if not timeline_file.exists():
        print("No sample data. Run: python scripts/fetch_sample_match.py")
        exit(1)

    with open(timeline_file) as f:
        timeline = json.load(f)

    print("Detecting ganks from positional data...\n")

    # Test on all participants
    for participant_id in range(1, 11):
        ganks = extract_ganks_from_timeline(
            timeline, participant_id, min_confidence=0.6
        )

        if len(ganks) >= 1:  # Only print if any gank activity
            print_ganks(ganks, participant_id)
