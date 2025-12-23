"""
Unified jungle path extraction combining camp clears and gank attempts.

This merges:
1. Camp clears from camp_inference_v3 (CS-based detection)
2. Gank attempts from gank_detection (position-based detection)

Into a single chronological sequence of jungle actions.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
from camp_inference_v3 import extract_jungle_path_cs, CampClear
from gank_detection import extract_ganks_from_timeline, GankAttempt


@dataclass
class JungleAction:
    """Unified representation of any jungle action (camp clear or gank)."""
    timestamp: float
    participant_id: int
    action_name: str  # Either camp name or "gank_top"/"gank_mid"/"gank_bot"
    position: Tuple[int, int]
    confidence: float
    action_type: str  # "camp" or "gank"


def merge_jungle_actions(
    camp_clears: List[CampClear],
    ganks: List[GankAttempt]
) -> List[JungleAction]:
    """
    Merge camp clears and ganks into a single chronological list.

    Args:
        camp_clears: List of detected camp clears
        ganks: List of detected gank attempts

    Returns:
        Sorted list of JungleAction objects
    """
    actions = []

    # Add camp clears
    for camp in camp_clears:
        actions.append(JungleAction(
            timestamp=camp.timestamp,
            participant_id=camp.participant_id,
            action_name=camp.camp_name,
            position=camp.position,
            confidence=camp.confidence,
            action_type="camp"
        ))

    # Add ganks
    for gank in ganks:
        actions.append(JungleAction(
            timestamp=gank.timestamp,
            participant_id=gank.participant_id,
            action_name=gank.lane,  # "gank_top", "gank_mid", or "gank_bot"
            position=gank.position,
            confidence=gank.confidence,
            action_type="gank"
        ))

    # Sort by timestamp
    actions.sort(key=lambda a: a.timestamp)

    return actions


def extract_full_jungle_path(timeline: dict,
                             participant_id: int,
                             jungler_team: int = None,
                             camp_min_confidence: float = 0.3,
                             gank_min_confidence: float = 0.6) -> List[JungleAction]:
    """
    Extract complete jungle path including camps and ganks.

    Args:
        timeline: Full match timeline from Riot API
        participant_id: Which player (1-10)
        jungler_team: 100 or 200 (auto-detected if None)
        camp_min_confidence: Minimum confidence for camp clears
        gank_min_confidence: Minimum confidence for ganks

    Returns:
        Chronologically sorted list of all jungle actions
    """
    # Auto-detect team if not provided
    if jungler_team is None:
        jungler_team = 100 if participant_id <= 5 else 200

    # Extract camp clears
    camp_clears = extract_jungle_path_cs(
        timeline, participant_id, min_confidence=camp_min_confidence
    )

    # Extract ganks
    ganks = extract_ganks_from_timeline(
        timeline, participant_id, min_confidence=gank_min_confidence
    )

    # Merge into single path
    full_path = merge_jungle_actions(camp_clears, ganks)

    return full_path


def print_jungle_path(jungle_path: List[JungleAction], participant_id: int):
    """Pretty print full jungle path."""
    print(f"\n{'='*80}")
    print(f"FULL JUNGLE PATH - Participant {participant_id}")
    print(f"{'='*80}")
    print(f"{'Time':<8} {'Type':<6} {'Action':<20} {'Position':<20} {'Conf':<6}")
    print(f"{'-'*80}")

    for action in jungle_path:
        time_str = f"{int(action.timestamp//60)}:{int(action.timestamp%60):02d}"
        pos_str = f"({action.position[0]}, {action.position[1]})"
        action_type_str = action.action_type.upper()

        print(f"{time_str:<8} {action_type_str:<6} {action.action_name:<20} "
              f"{pos_str:<20} {action.confidence:.2f}")

    print(f"{'='*80}")
    camp_count = sum(1 for a in jungle_path if a.action_type == "camp")
    gank_count = sum(1 for a in jungle_path if a.action_type == "gank")
    print(f"Total actions: {len(jungle_path)} ({camp_count} camps, {gank_count} ganks)")
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

    print("Extracting full jungle paths (camps + ganks)...\n")

    # Analyze all participants
    for participant_id in range(1, 11):
        jungle_path = extract_full_jungle_path(
            timeline, participant_id,
            camp_min_confidence=0.3,
            gank_min_confidence=0.6
        )

        # Only print if significant jungle activity
        if len(jungle_path) >= 5:
            print_jungle_path(jungle_path, participant_id)
