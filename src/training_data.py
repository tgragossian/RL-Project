"""
Convert collected jungle paths into training data for behavior cloning.

State representation: What the jungler knows at each decision point
Action: Which camp they chose to clear next
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


# All possible camps
ALL_CAMPS = [
    "blue_blue", "blue_gromp", "blue_wolves", "blue_raptors", "blue_red", "blue_krugs",
    "red_red", "red_krugs", "red_raptors", "red_wolves", "red_gromp", "red_blue",
    "dragon", "baron", "herald", "scuttle_bot", "scuttle_top"
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
    recent_camps: List[str]
) -> np.ndarray:
    """
    Create a state representation from game information.

    State features (~50-dim):
    - Game time (normalized to 0-30 min)
    - Current position (x, y normalized)
    - One-hot encoding of last camp cleared (17 dims)
    - Recent camp history (last 3 camps, one-hot) (51 dims)

    Total: ~70 dimensions
    """
    features = []

    # 1. Game time (1 feature) - normalize to 0-1 for 0-30 minutes
    time_normalized = min(current_time / 1800.0, 1.0)  # 1800s = 30min
    features.append(time_normalized)

    # 2. Position (2 features) - normalize to 0-1 (map is ~15000x15000)
    x_normalized = current_position[0] / 15000.0
    y_normalized = current_position[1] / 15000.0
    features.extend([x_normalized, y_normalized])

    # 3. Last camp one-hot (17 features)
    last_camp_onehot = np.zeros(len(ALL_CAMPS))
    if last_camp in CAMP_TO_IDX:
        last_camp_onehot[CAMP_TO_IDX[last_camp]] = 1.0
    features.extend(last_camp_onehot)

    # 4. Recent camp history (last 3 camps, 51 features)
    for i in range(3):
        camp_onehot = np.zeros(len(ALL_CAMPS))
        if i < len(recent_camps) and recent_camps[-(i+1)] in CAMP_TO_IDX:
            camp_onehot[CAMP_TO_IDX[recent_camps[-(i+1)]]] = 1.0
        features.extend(camp_onehot)

    return np.array(features, dtype=np.float32)


def extract_training_examples(jungle_path: List[Dict]) -> List[TrainingExample]:
    """
    Convert a jungle path into state-action training pairs.

    Each clear becomes: (state_before, action_taken)
    """
    examples = []
    recent_camps = []

    for i in range(len(jungle_path)):
        clear = jungle_path[i]

        # Current state
        timestamp = clear['timestamp']
        position = tuple(clear['position'])
        camp_name = clear['camp_name']

        # Get last camp (if exists)
        last_camp = recent_camps[-1] if recent_camps else "none"

        # Create state vector
        state = create_state_vector(
            current_time=timestamp,
            current_position=position,
            last_camp=last_camp,
            recent_camps=recent_camps[-3:]  # Last 3 camps
        )

        # Action is the camp they chose
        action = CAMP_TO_IDX.get(camp_name, 0)

        examples.append(TrainingExample(
            state=state,
            action=action,
            timestamp=timestamp
        ))

        # Update history
        recent_camps.append(camp_name)

    return examples


def load_and_process_data(data_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load collected jungle data and convert to training format.

    Returns:
        (states, actions) - numpy arrays ready for training
    """
    with open(data_file) as f:
        data = json.load(f)

    all_examples = []

    for player in data['players']:
        for match in player['matches']:
            jungle_path = match['jungle_path']
            examples = extract_training_examples(jungle_path)
            all_examples.extend(examples)

    # Convert to numpy arrays
    states = np.array([ex.state for ex in all_examples])
    actions = np.array([ex.action for ex in all_examples])

    return states, actions


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

    states, actions = load_and_process_data(data_file)
    print_dataset_info(states, actions)

    # Save processed data
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "actions.npy", actions)

    print(f"✓ Saved preprocessed data to:")
    print(f"  - {output_dir / 'states.npy'}")
    print(f"  - {output_dir / 'actions.npy'}")
