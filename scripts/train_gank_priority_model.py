"""
Train Random Forest model to predict gank lane priorities.

This lightweight model predicts whether high-elo junglers choose to gank
Top/Mid/Bot given the current game state plus powerspike-derived lane features.

The model's output probabilities serve as a structured prior for gank efficiency
and timing, which feeds into both BC and RL training.

Features:
- Jungler level, gold, items
- Ally/enemy lane champion powerspike scores at current game time
- Lane HP states
- Game time
- Recent action history

Target:
- Which lane to gank (Top/Mid/Bot) or no gank (Farm)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent / "src"))
from powerspike_system import PowerspikeSystem


@dataclass
class GankExample:
    """Training example for gank priority prediction."""
    # Jungler state
    jungler_level: int
    jungler_gold: int
    jungler_spike_score: float

    # Lane powerspike scores (relative advantage)
    top_spike_diff: float  # ally - enemy
    mid_spike_diff: float
    bot_spike_diff: float

    # Absolute lane scores
    ally_top_spike: float
    ally_mid_spike: float
    ally_bot_spike: float
    enemy_top_spike: float
    enemy_mid_spike: float
    enemy_bot_spike: float

    # Game context
    game_time: int  # seconds

    # Target: which lane was ganked (or "farm" if no gank)
    target_lane: str  # "top", "mid", "bot", or "farm"


def load_jungle_data(jungle_data_path: Path) -> dict:
    """Load the jungle match data."""
    with open(jungle_data_path, 'r') as f:
        return json.load(f)


def load_powerspike_data(powerspike_data_path: Path) -> dict:
    """Load the powerspike match data."""
    with open(powerspike_data_path, 'r') as f:
        return json.load(f)


def find_powerspike_snapshot(
    match_id: str,
    participant_id: int,
    timestamp: int,
    powerspike_matches: List[dict],
    tolerance_ms: int = 5000
) -> Optional[dict]:
    """
    Find the closest powerspike snapshot for a given match/time.

    Args:
        match_id: Match ID
        participant_id: Participant ID
        timestamp: Timestamp in milliseconds
        powerspike_matches: List of powerspike match data
        tolerance_ms: Maximum time difference to accept

    Returns:
        Snapshot dict or None if not found
    """
    # Find the match
    target_match = None
    for match in powerspike_matches:
        if match['match_id'] == match_id:
            target_match = match
            break

    if not target_match:
        return None

    # Find closest snapshot for this participant
    best_snapshot = None
    best_time_diff = float('inf')

    for snapshot in target_match['snapshots']:
        if snapshot['participant_id'] != participant_id:
            continue

        time_diff = abs(snapshot['timestamp'] - timestamp)
        if time_diff < best_time_diff and time_diff <= tolerance_ms:
            best_time_diff = time_diff
            best_snapshot = snapshot

    return best_snapshot


def extract_gank_examples(
    jungle_data: dict,
    powerspike_matches: List[dict],
    powerspike_system: PowerspikeSystem
) -> List[GankExample]:
    """
    Extract training examples from jungle paths with powerspike data.

    For each action in the jungle path, we create a training example:
    - If action is a gank → target is that lane
    - If action is a camp clear → target is "farm"

    This teaches the model when junglers choose to gank vs farm.
    """
    examples = []

    print(f"\n{'='*70}")
    print("EXTRACTING GANK EXAMPLES")
    print(f"{'='*70}\n")

    total_players = len(jungle_data['players'])

    for player_idx, player in enumerate(jungle_data['players']):
        print(f"Player {player_idx + 1}/{total_players}: {player['summoner_name']}")

        for match in player['matches']:
            match_id = match['match_id']
            participant_id = match['participant_id']
            jungle_path = match['jungle_path']

            match_examples = 0

            for action in jungle_path:
                timestamp = action['timestamp']
                camp_name = action['camp_name']

                # Find powerspike snapshot for this timestamp
                snapshot = find_powerspike_snapshot(
                    match_id,
                    participant_id,
                    timestamp,
                    powerspike_matches
                )

                if not snapshot:
                    continue

                # Get jungler state
                jungler_level = snapshot['level']
                jungler_gold = snapshot['gold_earned']
                jungler_spike = snapshot['overall_spike_score']

                # We need to find all 10 players' snapshots at this timestamp
                # to calculate lane spike scores
                all_snapshots = []
                for match_ps in powerspike_matches:
                    if match_ps['match_id'] == match_id:
                        for snap in match_ps['snapshots']:
                            if abs(snap['timestamp'] - timestamp) <= 5000:
                                all_snapshots.append(snap)
                        break

                if len(all_snapshots) < 10:
                    continue  # Not enough data

                # Determine teams (participants 1-5 are team 100, 6-10 are team 200)
                jungler_team = 100 if participant_id <= 5 else 200

                # Map participants to roles (simplified assumption)
                # In real data, we'd need to detect roles properly
                # For now: assume IDs 1,6=top, 2,7=jungle, 3,8=mid, 4,9=adc, 5,10=support

                team_role_map = {
                    100: {1: 'top', 2: 'jungle', 3: 'mid', 4: 'bot', 5: 'bot'},
                    200: {6: 'top', 7: 'jungle', 8: 'mid', 9: 'bot', 10: 'bot'}
                }

                # Get lane powerspike scores
                lane_scores = {}

                for snap in all_snapshots:
                    pid = snap['participant_id']
                    team = 100 if pid <= 5 else 200

                    # Get role
                    if pid in team_role_map.get(team, {}):
                        role = team_role_map[team][pid]

                        if team == jungler_team:
                            key = f'ally_{role}'
                        else:
                            key = f'enemy_{role}'

                        if role != 'jungle':  # Don't include jungler in lane scores
                            lane_scores[key] = snap['overall_spike_score']

                # Extract lane scores (with defaults)
                ally_top = lane_scores.get('ally_top', 0.5)
                ally_mid = lane_scores.get('ally_mid', 0.5)
                ally_bot = lane_scores.get('ally_bot', 0.5)
                enemy_top = lane_scores.get('enemy_top', 0.5)
                enemy_mid = lane_scores.get('enemy_mid', 0.5)
                enemy_bot = lane_scores.get('enemy_bot', 0.5)

                # Calculate spike differentials
                top_diff = ally_top - enemy_top
                mid_diff = ally_mid - enemy_mid
                bot_diff = ally_bot - enemy_bot

                # Determine target
                if 'gank_top' in camp_name:
                    target = 'top'
                elif 'gank_mid' in camp_name:
                    target = 'mid'
                elif 'gank_bot' in camp_name:
                    target = 'bot'
                else:
                    target = 'farm'  # Camp clear

                example = GankExample(
                    jungler_level=jungler_level,
                    jungler_gold=jungler_gold,
                    jungler_spike_score=jungler_spike,
                    top_spike_diff=top_diff,
                    mid_spike_diff=mid_diff,
                    bot_spike_diff=bot_diff,
                    ally_top_spike=ally_top,
                    ally_mid_spike=ally_mid,
                    ally_bot_spike=ally_bot,
                    enemy_top_spike=enemy_top,
                    enemy_mid_spike=enemy_mid,
                    enemy_bot_spike=enemy_bot,
                    game_time=timestamp // 1000,  # Convert to seconds
                    target_lane=target
                )

                examples.append(example)
                match_examples += 1

            if match_examples > 0:
                print(f"  ✓ {match_id[:15]}: {match_examples} examples")

    print(f"\n✓ Extracted {len(examples)} total examples")
    return examples


def examples_to_arrays(examples: List[GankExample]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert examples to feature matrix and target array.

    Features (13-dim):
    - jungler_level (normalized)
    - jungler_gold (normalized)
    - jungler_spike_score
    - top_spike_diff
    - mid_spike_diff
    - bot_spike_diff
    - ally_top_spike
    - ally_mid_spike
    - ally_bot_spike
    - enemy_top_spike
    - enemy_mid_spike
    - enemy_bot_spike
    - game_time (normalized)

    Targets: 0=farm, 1=top, 2=mid, 3=bot
    """
    X = []
    y = []

    target_map = {'farm': 0, 'top': 1, 'mid': 2, 'bot': 3}

    for ex in examples:
        features = [
            ex.jungler_level / 18.0,  # Normalize to 0-1
            ex.jungler_gold / 20000.0,  # Normalize (typical max ~20k)
            ex.jungler_spike_score,
            ex.top_spike_diff,
            ex.mid_spike_diff,
            ex.bot_spike_diff,
            ex.ally_top_spike,
            ex.ally_mid_spike,
            ex.ally_bot_spike,
            ex.enemy_top_spike,
            ex.enemy_mid_spike,
            ex.enemy_bot_spike,
            ex.game_time / 1800.0,  # Normalize to 0-1 (30 min)
        ]

        X.append(features)
        y.append(target_map[ex.target_lane])

    return np.array(X), np.array(y)


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train Random Forest classifier for gank priority prediction.

    We use a lightweight RF (not deep) so it generalizes well and
    provides smooth probability estimates.
    """
    print(f"\n{'='*70}")
    print("TRAINING RANDOM FOREST")
    print(f"{'='*70}\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} examples")
    print(f"Test set: {len(X_test)} examples")

    # Distribution
    unique, counts = np.unique(y_train, return_counts=True)
    label_names = ['farm', 'top', 'mid', 'bot']
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label_names[label]:10s}: {count:>5d} ({count/len(y_train)*100:.1f}%)")

    # Train model
    # Use moderate complexity to avoid overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )

    print(f"\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}\n")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    # Detailed metrics
    y_pred = model.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Feature importance
    print(f"\nFeature Importance:")
    feature_names = [
        'jungler_level', 'jungler_gold', 'jungler_spike',
        'top_spike_diff', 'mid_spike_diff', 'bot_spike_diff',
        'ally_top_spike', 'ally_mid_spike', 'ally_bot_spike',
        'enemy_top_spike', 'enemy_mid_spike', 'enemy_bot_spike',
        'game_time'
    ]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {feature_names[idx]:20s}: {importances[idx]:.3f}")

    return model


def main():
    print("="*70)
    print("GANK PRIORITY MODEL TRAINING")
    print("="*70)

    # Paths
    base_dir = Path(__file__).parent.parent
    jungle_data_path = base_dir / "data" / "processed" / "challenger_jungle_data.json"
    powerspike_data_path = base_dir / "data" / "processed" / "powerspike_match_data.json"
    powerspike_csv_path = base_dir / "data" / "raw" / "Champ Powerspikes _ Updated Patch 14.21 - Powerspikes.csv"

    # Check files exist
    if not jungle_data_path.exists():
        print(f"❌ Jungle data not found: {jungle_data_path}")
        print("Run: python scripts/collect_from_leaderboard.py")
        return

    if not powerspike_data_path.exists():
        print(f"❌ Powerspike data not found: {powerspike_data_path}")
        print("Run: python scripts/collect_powerspike_data.py")
        return

    if not powerspike_csv_path.exists():
        print(f"❌ Powerspike CSV not found: {powerspike_csv_path}")
        return

    # Load data
    print("\nLoading data...")
    jungle_data = load_jungle_data(jungle_data_path)
    powerspike_data = load_powerspike_data(powerspike_data_path)
    powerspike_system = PowerspikeSystem(powerspike_csv_path)

    print(f"✓ Jungle matches: {jungle_data['total_matches']}")
    print(f"✓ Powerspike matches: {powerspike_data['total_matches']}")

    # Extract examples
    examples = extract_gank_examples(
        jungle_data,
        powerspike_data['matches'],
        powerspike_system
    )

    if len(examples) == 0:
        print("❌ No training examples extracted!")
        return

    # Convert to arrays
    X, y = examples_to_arrays(examples)

    # Train model
    model = train_random_forest(X, y)

    # Save model
    output_path = base_dir / "models" / "gank_priority_rf.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n{'='*70}")
    print(f"✅ MODEL SAVED")
    print(f"{'='*70}")
    print(f"Path: {output_path}")
    print(f"\nThis model can now be used to predict gank priorities based on:")
    print(f"  - Champion powerspike scores (early/late game advantage)")
    print(f"  - Lane spike differentials (which lane has advantage)")
    print(f"  - Jungler state (level, gold)")
    print(f"  - Game time")


if __name__ == "__main__":
    main()
