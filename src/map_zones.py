"""
Map zone definitions for Summoner's Rift.

Discretizes the map into macro-regions for jungle decision-making:
- Lanes: Top, Mid, Bot
- Gank regions: Top Gank, Mid Gank, Bot Gank
- River: Top River, Mid River, Bot River
- Objectives: Baron Pit, Dragon Pit
- Jungle quadrants: Blue Side (Top/Bot), Red Side (Top/Bot)

Coordinates are based on Riot's map coordinate system (0-14820 for both x and y).
Blue team base is bottom-left (low x, low y), Red team base is top-right (high x, high y).

Based on annotated_map.png visual reference.
"""

from typing import Tuple, Optional
from enum import Enum


class MapZone(Enum):
    """Enumeration of all map zones."""
    # Lanes
    TOP_LANE = "top_lane"
    MID_LANE = "mid_lane"
    BOT_LANE = "bot_lane"

    # Gank regions
    TOP_GANK = "top_gank"
    MID_GANK = "mid_gank"
    BOT_GANK = "bot_gank"

    # River
    TOP_RIVER = "top_river"
    MID_RIVER = "mid_river"
    BOT_RIVER = "bot_river"

    # Objectives
    BARON_PIT = "baron_pit"
    DRAGON_PIT = "dragon_pit"

    # Blue side jungle (bottom-left team)
    BLUE_TOP_JUNGLE = "blue_top_jungle"
    BLUE_BOT_JUNGLE = "blue_bot_jungle"

    # Red side jungle (top-right team)
    RED_TOP_JUNGLE = "red_top_jungle"
    RED_BOT_JUNGLE = "red_bot_jungle"

    # Bases
    BLUE_BASE = "blue_base"
    RED_BASE = "red_base"

    # Unknown
    UNKNOWN = "unknown"


# Zone boundaries (x_min, x_max, y_min, y_max)
# Based on annotated_map.png and standard Summoner's Rift coordinates
ZONE_BOUNDARIES = {
    # Lanes (diagonal corridors)
    MapZone.TOP_LANE: (1000, 13820, 10000, 13820),
    MapZone.MID_LANE: (5000, 9820, 5000, 9820),
    MapZone.BOT_LANE: (1000, 4820, 1000, 4820),

    # Gank regions (between jungle and lanes)
    MapZone.TOP_GANK: (4000, 10000, 10000, 13000),
    MapZone.MID_GANK: (5500, 9300, 5500, 9300),
    MapZone.BOT_GANK: (4820, 10820, 1820, 4820),

    # River regions
    MapZone.TOP_RIVER: (3500, 11000, 9500, 12000),
    MapZone.MID_RIVER: (5000, 10000, 5000, 10000),
    MapZone.BOT_RIVER: (3000, 11000, 3000, 6000),

    # Objectives
    MapZone.BARON_PIT: (4500, 5500, 10000, 11000),
    MapZone.DRAGON_PIT: (9300, 10300, 4000, 5000),

    # Blue side jungle (bottom-left quadrants)
    MapZone.BLUE_BOT_JUNGLE: (1500, 6500, 1500, 6500),
    MapZone.BLUE_TOP_JUNGLE: (1500, 6500, 6500, 10500),

    # Red side jungle (top-right quadrants)
    MapZone.RED_TOP_JUNGLE: (8300, 13300, 8500, 13300),
    MapZone.RED_BOT_JUNGLE: (8300, 13300, 4000, 8500),

    # Bases
    MapZone.BLUE_BASE: (0, 2000, 0, 2000),
    MapZone.RED_BASE: (12820, 14820, 12820, 14820),
}


def point_in_rect(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
    """Check if point is inside rectangular boundary."""
    return x_min <= x <= x_max and y_min <= y <= y_max


def point_in_diagonal_lane(x: float, y: float, zone: MapZone, threshold: float = 2000) -> bool:
    """
    Check if point is in a diagonal lane region.

    Lanes run diagonally across the map:
    - Top lane: runs from top-left to top-right (high y region)
    - Bot lane: runs from bot-left to bot-right (low y region)
    - Mid lane: diagonal from bot-left base to top-right base

    Args:
        x, y: Position coordinates
        zone: Lane zone to check
        threshold: Distance threshold from lane center line
    """
    if zone == MapZone.TOP_LANE:
        # Top lane runs along top edge (high y)
        # L-shaped: vertical left side + horizontal top side
        in_top_region = y > 10000
        near_left_edge_top = x < 4000 and y > 8000
        return in_top_region or near_left_edge_top

    elif zone == MapZone.BOT_LANE:
        # Bot lane runs along bottom edge (low y)
        # L-shaped: horizontal bottom side + vertical right side
        in_bot_region = y < 4820
        near_right_edge_bot = x > 10820 and y < 7000
        return in_bot_region or near_right_edge_bot

    elif zone == MapZone.MID_LANE:
        # Mid lane is diagonal from (0,0) to (14820, 14820)
        # Distance from line y = x
        distance_from_diagonal = abs(x - y)
        on_diagonal = distance_from_diagonal < threshold
        in_center = 3000 < x < 11820 and 3000 < y < 11820
        # Exclude top/bot lane regions
        not_in_top = y < 10000 or x > 4000
        not_in_bot = y > 4820 or x < 10820
        return on_diagonal and in_center and not_in_top and not_in_bot

    return False


def get_zone(x: float, y: float) -> MapZone:
    """
    Determine which map zone a position belongs to.

    Priority order:
    1. Objectives (Baron, Dragon)
    2. Bases
    3. Lanes
    4. Gank regions
    5. River
    6. Jungle quadrants

    Args:
        x, y: Map coordinates (0-14820 range)

    Returns:
        MapZone enum value
    """
    # Check objectives first (highest priority)
    for zone in [MapZone.BARON_PIT, MapZone.DRAGON_PIT]:
        bounds = ZONE_BOUNDARIES[zone]
        if point_in_rect(x, y, *bounds):
            return zone

    # Check bases
    for zone in [MapZone.BLUE_BASE, MapZone.RED_BASE]:
        bounds = ZONE_BOUNDARIES[zone]
        if point_in_rect(x, y, *bounds):
            return zone

    # Check lanes (diagonal)
    for zone in [MapZone.TOP_LANE, MapZone.MID_LANE, MapZone.BOT_LANE]:
        if point_in_diagonal_lane(x, y, zone):
            return zone

    # Check gank regions
    for zone in [MapZone.TOP_GANK, MapZone.MID_GANK, MapZone.BOT_GANK]:
        bounds = ZONE_BOUNDARIES[zone]
        if point_in_rect(x, y, *bounds):
            return zone

    # Check river
    for zone in [MapZone.TOP_RIVER, MapZone.MID_RIVER, MapZone.BOT_RIVER]:
        bounds = ZONE_BOUNDARIES[zone]
        if point_in_rect(x, y, *bounds):
            return zone

    # Check jungle quadrants (lowest priority, catch-all)
    for zone in [MapZone.BLUE_BOT_JUNGLE, MapZone.BLUE_TOP_JUNGLE,
                 MapZone.RED_TOP_JUNGLE, MapZone.RED_BOT_JUNGLE]:
        bounds = ZONE_BOUNDARIES[zone]
        if point_in_rect(x, y, *bounds):
            return zone

    return MapZone.UNKNOWN


def get_zone_center(zone: MapZone) -> Tuple[float, float]:
    """
    Get approximate center coordinates of a zone.

    Args:
        zone: Map zone

    Returns:
        (x, y) tuple of center coordinates
    """
    if zone not in ZONE_BOUNDARIES:
        return (7410, 7410)  # Map center

    bounds = ZONE_BOUNDARIES[zone]
    x_min, x_max, y_min, y_max = bounds

    return ((x_min + x_max) / 2, (y_min + y_max) / 2)


def is_gank_zone(zone: MapZone) -> bool:
    """Check if zone is a gank region."""
    return zone in [MapZone.TOP_GANK, MapZone.MID_GANK, MapZone.BOT_GANK]


def is_objective_zone(zone: MapZone) -> bool:
    """Check if zone is an objective (Baron/Dragon)."""
    return zone in [MapZone.BARON_PIT, MapZone.DRAGON_PIT]


def is_jungle_zone(zone: MapZone) -> bool:
    """Check if zone is a jungle quadrant."""
    return zone in [
        MapZone.BLUE_BOT_JUNGLE, MapZone.BLUE_TOP_JUNGLE,
        MapZone.RED_TOP_JUNGLE, MapZone.RED_BOT_JUNGLE
    ]


def zone_to_string(zone: MapZone) -> str:
    """Convert zone to human-readable string."""
    return zone.value.replace('_', ' ').title()


if __name__ == "__main__":
    # Test zone detection with known camp positions
    print("="*70)
    print("MAP ZONE DETECTION TEST")
    print("="*70)

    # Test cases: (x, y, expected_zone_type)
    test_positions = [
        (2711, 8231, "Blue Top Jungle"),  # Blue Gromp area
        (8047, 2346, "Red Bot Jungle"),   # Red Krugs area
        (5000, 10500, "Baron Pit area"),  # Near Baron
        (9800, 4500, "Dragon Pit area"),  # Near Dragon
        (7410, 7410, "Mid Lane/River"),   # Map center
        (1000, 13000, "Top Lane"),        # Top lane
        (13000, 1000, "Bot Lane"),        # Bot lane
    ]

    print("\nTesting position → zone mapping:\n")
    for x, y, description in test_positions:
        zone = get_zone(x, y)
        print(f"Position ({x:5.0f}, {y:5.0f}) [{description:20s}] → {zone_to_string(zone)}")

    print("\n" + "="*70)
    print("Zone centers:")
    print("="*70 + "\n")

    for zone in MapZone:
        if zone != MapZone.UNKNOWN:
            center_x, center_y = get_zone_center(zone)
            print(f"{zone_to_string(zone):25s}: ({center_x:7.1f}, {center_y:7.1f})")
