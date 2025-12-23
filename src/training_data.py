"""
Convert collected jungle paths into training data for behavior cloning.

State representation: What the jungler knows at each decision point
Action: Which camp they chose to clear next

ENHANCED: Now includes powerspike scores and gank priority predictions.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

try:
    from gank_priority import GankPriorityPredictor, create_lane_priority_features
    GANK_PRIORITY_AVAILABLE = True
except ImportError:
    GANK_PRIORITY_AVAILABLE = False
    print("Warning: gank_priority module not available, using fallback features")


# All possible camps + gank actions
ALL_CAMPS = [
    # Blue side jungle camps
    "blue_blue", "blue_gromp", "blue_wolves", "blue_raptors", "blue_red", "blue_krugs",
    # Red side jungle camps
    "red_red", "red_krugs", "red_raptors", "red_wolves", "red_gromp", "red_blue",
    # Epic objectives
    "dragon", "baron", "herald", "scuttle_bot", "scuttle_top",
    # Gank actions
    "gank_top", "gank_mid", "gank_bot"
]

CAMP_TO_IDX = {camp: i for i, camp in enumerate(ALL_CAMPS)}


@dataclass
class TrainingExample:
    """A single state-action pair for training."""
    state: np.ndarray    # Game state features
    action: int          # Camp index (0-16)
    timestamp: float     # For debugging


def create_state_vector(
    current_time: float,
    current_position: Tuple[int, int],
    last_camp: str,
    recent_camps: List[str],
    powerspike_features: Optional[Dict] = None,
    gank_priority_predictor: Optional['GankPriorityPredictor'] = None
) -> np.ndarray:
    """
    Create a state representation from game information.

    ENHANCED State features (~83-dim):
    - Game time (1 dim)
    - Current position (x, y) (2 dims)
    - Last camp one-hot (20 dims)
    - Recent camp history (last 3 camps, one-hot) (60 dims)
    - Powerspike features (if available):
      - Jungler spike score (1 dim)
      - Lane spike scores (6 dims: ally top/mid/bot, enemy top/mid/bot)
      - Gank lane priorities (3 dims: top/mid/bot predicted probabilities)

    Total: 3 + 80 base = 83 dimensions with powerspikes
           (or 73 without powerspikes - backward compatible)

    Args:
        current_time: Game time in seconds
        current_position: (x, y) position tuple
        last_camp: Name of last camp cleared
        recent_camps: List of recent camp names
        powerspike_features: Optional dict with:
            - jungler_level, jungler_gold, jungler_spike
            - ally_top_spike, ally_mid_spike, ally_bot_spike
            - enemy_top_spike, enemy_mid_spike, enemy_bot_spike
        gank_priority_predictor: Optional trained model for gank priorities
    """
    features = []

    # 1. Game time (1 feature) - normalize to 0-1 for 0-30 minutes
    time_normalized = min(current_time / 1800.0, 1.0)  # 1800s = 30min
    features.append(time_normalized)

    # 2. Position (2 features) - normalize to 0-1 (map is ~15000x15000)
    x_normalized = current_position[0] / 15000.0
    y_normalized = current_position[1] / 15000.0
    features.extend([x_normalized, y_normalized])

    # 3. Last camp one-hot (20 features)
    last_camp_onehot = np.zeros(len(ALL_CAMPS))
    if last_camp in CAMP_TO_IDX:
        last_camp_onehot[CAMP_TO_IDX[last_camp]] = 1.0
    features.extend(last_camp_onehot)

    # 4. Recent camp history (last 3 camps, 60 features)
    for i in range(3):
        camp_onehot = np.zeros(len(ALL_CAMPS))
        if i < len(recent_camps) and recent_camps[-(i+1)] in CAMP_TO_IDX:
            camp_onehot[CAMP_TO_IDX[recent_camps[-(i+1)]]] = 1.0
        features.extend(camp_onehot)

    # 5. Powerspike features (10 features) - NEW!
    if powerspike_features:
        # Jungler spike score
        features.append(powerspike_features.get('jungler_spike', 0.5))

        # Lane spike scores (6 features)
        features.append(powerspike_features.get('ally_top_spike', 0.5))
        features.append(powerspike_features.get('ally_mid_spike', 0.5))
        features.append(powerspike_features.get('ally_bot_spike', 0.5))
        features.append(powerspike_features.get('enemy_top_spike', 0.5))
        features.append(powerspike_features.get('enemy_mid_spike', 0.5))
        features.append(powerspike_features.get('enemy_bot_spike', 0.5))

        # Gank priority predictions (3 features) - NEW!
        if GANK_PRIORITY_AVAILABLE and gank_priority_predictor:
            priorities = create_lane_priority_features(
                jungler_level=powerspike_features.get('jungler_level', 1),
                jungler_gold=powerspike_features.get('jungler_gold', 0),
                jungler_spike_score=powerspike_features.get('jungler_spike', 0.5),
                ally_top_spike=powerspike_features.get('ally_top_spike', 0.5),
                ally_mid_spike=powerspike_features.get('ally_mid_spike', 0.5),
                ally_bot_spike=powerspike_features.get('ally_bot_spike', 0.5),
                enemy_top_spike=powerspike_features.get('enemy_top_spike', 0.5),
                enemy_mid_spike=powerspike_features.get('enemy_mid_spike', 0.5),
                enemy_bot_spike=powerspike_features.get('enemy_bot_spike', 0.5),
                game_time_seconds=int(current_time),
                predictor=gank_priority_predictor
            )
            features.extend(priorities)  # top, mid, bot priorities
        else:
            # Fallback: simple spike differential priorities
            top_diff = powerspike_features.get('ally_top_spike', 0.5) - powerspike_features.get('enemy_top_spike', 0.5)
            mid_diff = powerspike_features.get('ally_mid_spike', 0.5) - powerspike_features.get('enemy_mid_spike', 0.5)
            bot_diff = powerspike_features.get('ally_bot_spike', 0.5) - powerspike_features.get('enemy_bot_spike', 0.5)

            # Convert to soft priorities using softmax
            diffs = np.array([top_diff, mid_diff, bot_diff])
            exp_diffs = np.exp(diffs * 2.0)
            priorities = exp_diffs / exp_diffs.sum()
            features.extend(priorities)
    else:
        # Backward compatibility: add dummy features if no powerspike data
        features.extend([0.5] * 10)  # 1 jungler spike + 6 lane spikes + 3 priorities

    return np.array(features, dtype=np.float32)


def extract_training_examples(jungle_path: List[Dict]) -> List[TrainingExample]:
    """
    Convert a jungle path into state-action training pairs.

    CORRECTED: State at camp_i predicts the NEXT action (camp_i+1).
    This ensures we're predicting the decision, not labeling where they already are.
    """
    examples = []
    recent_camps = []

    # Iterate through path, creating (state_at_i -> action_at_i+1) pairs
    for i in range(len(jungle_path) - 1):  # Stop before last element
        current_clear = jungle_path[i]
        next_clear = jungle_path[i + 1]

        # State: After clearing current camp
        timestamp = current_clear['timestamp']
        position = tuple(current_clear['position'])
        camp_name = current_clear['camp_name']

        # Get last camp (if exists)
        last_camp = recent_camps[-1] if recent_camps else "none"

        # Create state vector (where they are NOW)
        state = create_state_vector(
            current_time=timestamp,
            current_position=position,
            last_camp=last_camp,
            recent_camps=recent_camps[-3:]  # Last 3 camps
        )

        # Action: The NEXT camp they chose to go to
        next_camp_name = next_clear['camp_name']
        action = CAMP_TO_IDX.get(next_camp_name, 0)

        examples.append(TrainingExample(
            state=state,
            action=action,
            timestamp=timestamp
        ))

        # Update history (add current camp, since we just cleared it)
        recent_camps.append(camp_name)

    return examples


def load_and_process_data(data_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load collected jungle data and convert to training format.

    Returns:
        (states, actions, game_indices) - numpy arrays ready for training
        game_indices tracks which game each state came from (for proper train/val split)
    """
    with open(data_file) as f:
        data = json.load(f)

    all_examples = []
    game_id = 0

    for player in data['players']:
        for match in player['matches']:
            jungle_path = match['jungle_path']
            examples = extract_training_examples(jungle_path)

            # Tag each example with the game ID
            for ex in examples:
                all_examples.append((ex, game_id))

            game_id += 1

    # Convert to numpy arrays
    states = np.array([ex.state for ex, _ in all_examples])
    actions = np.array([ex.action for ex, _ in all_examples])
    game_indices = np.array([gid for _, gid in all_examples])

    return states, actions, game_indices


def print_dataset_info(states: np.ndarray, actions: np.ndarray):
    """Print information about the dataset."""
    print(f"\n{'='*70}")
    print(f"TRAINING DATASET")
    print(f"{'='*70}")
    print(f"Total examples: {len(states)}")
    print(f"State dimension: {states.shape[1]}")
    print(f"Action space: {len(ALL_CAMPS)} camps")
    print()
    print(f"Action distribution:")
    unique, counts = np.unique(actions, return_counts=True)
    for camp_idx, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        camp_name = ALL_CAMPS[camp_idx]
        print(f"  {camp_name:<20}: {count:>4} ({count/len(actions)*100:.1f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test data loading
    data_file = Path(__file__).parent.parent / "data" / "processed" / "challenger_jungle_data.json"

    if not data_file.exists():
        print("No data found. Run: python scripts/collect_from_leaderboard.py")
        exit(1)

    states, actions, game_indices = load_and_process_data(data_file)
    print_dataset_info(states, actions)

    print(f"\nGame-based split info:")
    print(f"  Total games: {len(np.unique(game_indices))}")
    print(f"  States per game (avg): {len(states) / len(np.unique(game_indices)):.1f}")

    # Save processed data
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "game_indices.npy", game_indices)

    print(f"\n✓ Saved preprocessed data to:")
    print(f"  - {output_dir / 'states.npy'}")
    print(f"  - {output_dir / 'actions.npy'}")
    print(f"  - {output_dir / 'game_indices.npy'}")
