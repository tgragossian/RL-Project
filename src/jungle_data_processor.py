"""
Process raw jungle match data into RL training format.

Converts match snapshots + jungle paths into state-action-reward sequences
that can be used for behavior cloning and RL training.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class JungleAction:
    """Represents a jungle action."""
    action_type: str  # "camp" or "gank"
    action_name: str  # e.g., "blue_buff", "gank_top"
    timestamp: int
    position: Tuple[int, int]


@dataclass
class GameState:
    """Game state at a specific timestamp."""
    timestamp: int
    # Jungler state
    jungler_level: int
    jungler_gold: int
    jungler_items: List[str]
    jungler_position: Tuple[int, int]
    jungler_spike_score: float

    # Team states (simplified - just spike scores for now)
    ally_top_spike: float
    ally_mid_spike: float
    ally_bot_spike: float
    enemy_top_spike: float
    enemy_mid_spike: float
    enemy_bot_spike: float

    # Next action taken by the pro player
    next_action: Optional[JungleAction] = None

    # Match metadata
    match_id: str = ""
    patch: str = ""


class JungleDataProcessor:
    """Process raw match data into RL training format."""

    # Define action space
    CAMP_ACTIONS = [
        "blue_buff",
        "red_buff",
        "gromp",
        "wolves",
        "raptors",
        "krugs",
    ]

    GANK_ACTIONS = [
        "gank_top",
        "gank_mid",
        "gank_bot",
    ]

    ALL_ACTIONS = CAMP_ACTIONS + GANK_ACTIONS

    def __init__(self, jungle_data_path: Path, powerspike_data_path: Path):
        """
        Initialize processor with paths to data files.

        Args:
            jungle_data_path: Path to challenger_jungle_data.json
            powerspike_data_path: Path to powerspike_match_data.json
        """
        self.jungle_data_path = jungle_data_path
        self.powerspike_data_path = powerspike_data_path

        # Load data
        with open(jungle_data_path, 'r') as f:
            self.jungle_data = json.load(f)

        with open(powerspike_data_path, 'r') as f:
            self.powerspike_data = json.load(f)

        # Create match ID to data mappings
        self._build_match_mappings()

    def _build_match_mappings(self):
        """Build quick lookup for match data."""
        # Map match_id -> jungle path data
        self.jungle_paths = {}
        for player in self.jungle_data.get('players', []):
            for match in player.get('matches', []):
                match_id = match.get('match_id')
                self.jungle_paths[match_id] = match.get('jungle_path', [])

        # Map match_id -> powerspike snapshots
        self.powerspike_snapshots = {}
        for match in self.powerspike_data.get('matches', []):
            match_id = match.get('match_id')
            self.powerspike_snapshots[match_id] = match.get('snapshots', [])

    def _normalize_camp_name(self, camp_name: str) -> str:
        """Normalize camp names to standard format."""
        camp_name = camp_name.lower()

        # Map variations to standard names
        if 'blue' in camp_name:
            return 'blue_buff'
        elif 'red' in camp_name:
            return 'red_buff'
        elif 'gromp' in camp_name:
            return 'gromp'
        elif 'wolves' in camp_name or 'wolf' in camp_name:
            return 'wolves'
        elif 'raptors' in camp_name or 'raptor' in camp_name:
            return 'raptors'
        elif 'krugs' in camp_name or 'krug' in camp_name:
            return 'krugs'
        elif 'gank' in camp_name:
            # Already in format like "gank_top"
            return camp_name
        else:
            return camp_name

    def _get_snapshot_at_time(self, snapshots: List[Dict], timestamp: int, participant_id: int) -> Optional[Dict]:
        """Get the snapshot closest to the given timestamp for a participant."""
        # Find snapshots for this participant
        participant_snapshots = [
            s for s in snapshots
            if s.get('participant_id') == participant_id
        ]

        if not participant_snapshots:
            return None

        # Find closest timestamp (before or at the action time)
        closest = None
        min_diff = float('inf')

        for snapshot in participant_snapshots:
            snap_time = snapshot.get('timestamp', 0)
            diff = abs(timestamp - snap_time)

            # Prefer snapshots before the action
            if snap_time <= timestamp and diff < min_diff:
                min_diff = diff
                closest = snapshot

        return closest

    def _get_lane_snapshots(self, snapshots: List[Dict], timestamp: int) -> Dict[str, Dict]:
        """Get snapshots for each lane at a given timestamp."""
        # Simplified: Get all participants and approximate lanes by participant ID
        # In real implementation, would use position data to determine lanes

        lane_snapshots = {
            'ally_top': None,
            'ally_mid': None,
            'ally_bot': None,
            'enemy_top': None,
            'enemy_mid': None,
            'enemy_bot': None,
        }

        # Get snapshots near this timestamp
        time_snapshots = [s for s in snapshots if abs(s.get('timestamp', 0) - timestamp) < 60000]

        if not time_snapshots:
            return lane_snapshots

        # Approximate lane assignments (participant IDs 1-5 are team 1, 6-10 are team 2)
        # Top: 1, 6 | Mid: 2, 7 | Bot: 4, 9 | Jungle: 3, 8 | Support: 5, 10
        lane_map = {
            'ally_top': 1,
            'ally_mid': 2,
            'ally_bot': 4,
            'enemy_top': 6,
            'enemy_mid': 7,
            'enemy_bot': 9,
        }

        for lane, pid in lane_map.items():
            snapshot = self._get_snapshot_at_time(time_snapshots, timestamp, pid)
            if snapshot:
                lane_snapshots[lane] = snapshot

        return lane_snapshots

    def process_match(self, match_id: str) -> List[GameState]:
        """
        Process a single match into state-action sequences.

        Args:
            match_id: Match ID to process

        Returns:
            List of GameState objects with actions
        """
        # Get jungle path and snapshots for this match
        jungle_path = self.jungle_paths.get(match_id, [])
        snapshots = self.powerspike_snapshots.get(match_id, [])

        if not jungle_path or not snapshots:
            return []

        # Find the jungler's participant ID and patch from jungle path data
        jungler_participant_id = None
        match_patch = ""
        for player in self.jungle_data.get('players', []):
            for match in player.get('matches', []):
                if match.get('match_id') == match_id:
                    jungler_participant_id = match.get('participant_id')
                    match_patch = match.get('patch', '')
                    break
            if jungler_participant_id:
                break

        if not jungler_participant_id:
            return []

        game_states = []

        # Process each action in the jungle path
        for i, action_data in enumerate(jungle_path):
            timestamp = action_data.get('timestamp', 0)
            camp_name = action_data.get('camp_name', '')
            position = tuple(action_data.get('position', [0, 0]))

            # Normalize action name
            normalized_action = self._normalize_camp_name(camp_name)

            # Get jungler snapshot at this time
            jungler_snapshot = self._get_snapshot_at_time(snapshots, timestamp, jungler_participant_id)
            if not jungler_snapshot:
                continue

            # Get lane snapshots
            lane_snapshots = self._get_lane_snapshots(snapshots, timestamp)

            # Create action
            action_type = "gank" if "gank" in normalized_action else "camp"
            action = JungleAction(
                action_type=action_type,
                action_name=normalized_action,
                timestamp=timestamp,
                position=position
            )

            # Build game state
            state = GameState(
                timestamp=timestamp,
                jungler_level=jungler_snapshot.get('level', 1),
                jungler_gold=jungler_snapshot.get('gold_earned', 0),
                jungler_items=jungler_snapshot.get('items', []),
                jungler_position=tuple(jungler_snapshot.get('position', [0, 0])),
                jungler_spike_score=jungler_snapshot.get('overall_spike_score', 0.5),
                ally_top_spike=lane_snapshots['ally_top'].get('overall_spike_score', 0.5) if lane_snapshots['ally_top'] else 0.5,
                ally_mid_spike=lane_snapshots['ally_mid'].get('overall_spike_score', 0.5) if lane_snapshots['ally_mid'] else 0.5,
                ally_bot_spike=lane_snapshots['ally_bot'].get('overall_spike_score', 0.5) if lane_snapshots['ally_bot'] else 0.5,
                enemy_top_spike=lane_snapshots['enemy_top'].get('overall_spike_score', 0.5) if lane_snapshots['enemy_top'] else 0.5,
                enemy_mid_spike=lane_snapshots['enemy_mid'].get('overall_spike_score', 0.5) if lane_snapshots['enemy_mid'] else 0.5,
                enemy_bot_spike=lane_snapshots['enemy_bot'].get('overall_spike_score', 0.5) if lane_snapshots['enemy_bot'] else 0.5,
                next_action=action,
                match_id=match_id,
                patch=match_patch
            )

            game_states.append(state)

        return game_states

    def process_all_matches(self) -> List[GameState]:
        """Process all matches and return combined training data."""
        all_states = []

        match_ids = list(self.jungle_paths.keys())
        print(f"Processing {len(match_ids)} matches...")

        for i, match_id in enumerate(match_ids):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(match_ids)} matches...")

            states = self.process_match(match_id)
            all_states.extend(states)

        print(f"✓ Total training examples: {len(all_states)}")
        return all_states

    def states_to_arrays(self, states: List[GameState]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert GameState objects to numpy arrays for training.

        Returns:
            (observations, actions) tuple
            - observations: (N, state_dim) array
            - actions: (N,) array of action indices
        """
        observations = []
        actions = []

        for state in states:
            # Skip if no action
            if not state.next_action:
                continue

            # Convert action to index
            action_name = state.next_action.action_name
            if action_name not in self.ALL_ACTIONS:
                # Unknown action, skip this state entirely
                continue

            # Build observation vector (only for valid actions)
            obs = [
                state.jungler_level / 18.0,  # Normalize to 0-1
                state.jungler_gold / 20000.0,  # Normalize (assume max ~20k gold)
                len(state.jungler_items) / 6.0,  # Number of items (0-6)
                state.jungler_spike_score,
                state.ally_top_spike,
                state.ally_mid_spike,
                state.ally_bot_spike,
                state.enemy_top_spike,
                state.enemy_mid_spike,
                state.enemy_bot_spike,
                # Could add position, time, etc.
            ]

            observations.append(obs)
            action_idx = self.ALL_ACTIONS.index(action_name)
            actions.append(action_idx)

        return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


if __name__ == "__main__":
    # Test the processor
    jungle_data = Path(__file__).parent.parent / "data" / "processed" / "challenger_jungle_data.json"
    powerspike_data = Path(__file__).parent.parent / "data" / "processed" / "powerspike_match_data.json"

    if not jungle_data.exists():
        print(f"❌ Jungle data not found: {jungle_data}")
        exit(1)

    if not powerspike_data.exists():
        print(f"❌ Powerspike data not found: {powerspike_data}")
        exit(1)

    processor = JungleDataProcessor(jungle_data, powerspike_data)

    print(f"Loaded {len(processor.jungle_paths)} jungle matches")
    print(f"Loaded {len(processor.powerspike_snapshots)} powerspike matches")

    # Process all matches
    states = processor.process_all_matches()

    # Convert to arrays
    obs, actions = processor.states_to_arrays(states)

    print(f"\n{'='*70}")
    print("TRAINING DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Observations shape: {obs.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"State dimension: {obs.shape[1]}")
    print(f"Number of actions: {len(processor.ALL_ACTIONS)}")
    print(f"\nAction distribution:")
    for i, action_name in enumerate(processor.ALL_ACTIONS):
        count = np.sum(actions == i)
        pct = count / len(actions) * 100
        print(f"  {action_name}: {count} ({pct:.1f}%)")
