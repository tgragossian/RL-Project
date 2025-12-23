"""
Gank priority prediction system.

Uses the trained Random Forest model to predict gank priorities for each lane
based on champion powerspikes and game state. These priorities are used as:
1. Additional state features for the BC model
2. Reward shaping bonuses for the RL agent
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class GankPriorityPredictor:
    """
    Predicts gank priorities for Top/Mid/Bot lanes.

    Uses a trained Random Forest to estimate P(gank_lane | state, powerspikes).
    """

    def __init__(self, model_path: Path):
        """
        Load the trained Random Forest model.

        Args:
            model_path: Path to the pickled RandomForestClassifier
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict_lane_priorities(
        self,
        jungler_level: int,
        jungler_gold: int,
        jungler_spike_score: float,
        ally_top_spike: float,
        ally_mid_spike: float,
        ally_bot_spike: float,
        enemy_top_spike: float,
        enemy_mid_spike: float,
        enemy_bot_spike: float,
        game_time_seconds: int
    ) -> Tuple[float, float, float]:
        """
        Predict gank priority probabilities for each lane.

        Args:
            jungler_level: Current jungler level (1-18)
            jungler_gold: Jungler's total gold earned
            jungler_spike_score: Jungler's powerspike score (0-1)
            ally_top_spike: Ally top laner's spike score
            ally_mid_spike: Ally mid laner's spike score
            ally_bot_spike: Ally bot laner's spike score
            enemy_top_spike: Enemy top laner's spike score
            enemy_mid_spike: Enemy mid laner's spike score
            enemy_bot_spike: Enemy bot laner's spike score
            game_time_seconds: Current game time in seconds

        Returns:
            (top_priority, mid_priority, bot_priority) - Probabilities 0-1
        """
        # Calculate spike differentials
        top_diff = ally_top_spike - enemy_top_spike
        mid_diff = ally_mid_spike - enemy_mid_spike
        bot_diff = ally_bot_spike - enemy_bot_spike

        # Create feature vector (same as training)
        features = np.array([[
            jungler_level / 18.0,
            jungler_gold / 20000.0,
            jungler_spike_score,
            top_diff,
            mid_diff,
            bot_diff,
            ally_top_spike,
            ally_mid_spike,
            ally_bot_spike,
            enemy_top_spike,
            enemy_mid_spike,
            enemy_bot_spike,
            game_time_seconds / 1800.0,
        ]])

        # Get probability distribution over [farm, top, mid, bot]
        probs = self.model.predict_proba(features)[0]

        # Extract lane priorities (indices 1, 2, 3 = top, mid, bot)
        # Normalize to exclude "farm" probability
        top_prob = probs[1]
        mid_prob = probs[2]
        bot_prob = probs[3]

        # Renormalize lane probabilities to sum to 1
        total = top_prob + mid_prob + bot_prob
        if total > 0:
            top_prob /= total
            mid_prob /= total
            bot_prob /= total
        else:
            # Default to uniform if all zero
            top_prob = mid_prob = bot_prob = 1.0 / 3.0

        return top_prob, mid_prob, bot_prob

    def get_best_gank_lane(
        self,
        jungler_level: int,
        jungler_gold: int,
        jungler_spike_score: float,
        ally_top_spike: float,
        ally_mid_spike: float,
        ally_bot_spike: float,
        enemy_top_spike: float,
        enemy_mid_spike: float,
        enemy_bot_spike: float,
        game_time_seconds: int
    ) -> str:
        """
        Get the recommended gank lane.

        Returns:
            "top", "mid", or "bot"
        """
        top_p, mid_p, bot_p = self.predict_lane_priorities(
            jungler_level, jungler_gold, jungler_spike_score,
            ally_top_spike, ally_mid_spike, ally_bot_spike,
            enemy_top_spike, enemy_mid_spike, enemy_bot_spike,
            game_time_seconds
        )

        if top_p >= mid_p and top_p >= bot_p:
            return "top"
        elif mid_p >= bot_p:
            return "mid"
        else:
            return "bot"


def create_lane_priority_features(
    jungler_level: int,
    jungler_gold: int,
    jungler_spike_score: float,
    ally_top_spike: float,
    ally_mid_spike: float,
    ally_bot_spike: float,
    enemy_top_spike: float,
    enemy_mid_spike: float,
    enemy_bot_spike: float,
    game_time_seconds: int,
    predictor: Optional[GankPriorityPredictor] = None
) -> np.ndarray:
    """
    Create lane priority features for state representation.

    If predictor is provided, uses the trained model.
    Otherwise, uses a simple heuristic based on spike differentials.

    Returns:
        3-dim array: [top_priority, mid_priority, bot_priority]
    """
    if predictor:
        return np.array(predictor.predict_lane_priorities(
            jungler_level, jungler_gold, jungler_spike_score,
            ally_top_spike, ally_mid_spike, ally_bot_spike,
            enemy_top_spike, enemy_mid_spike, enemy_bot_spike,
            game_time_seconds
        ))
    else:
        # Fallback heuristic: prioritize lanes with spike advantage
        top_diff = ally_top_spike - enemy_top_spike
        mid_diff = ally_mid_spike - enemy_mid_spike
        bot_diff = ally_bot_spike - enemy_bot_spike

        # Convert to priorities (positive diff = higher priority)
        # Use softmax to convert to probabilities
        diffs = np.array([top_diff, mid_diff, bot_diff])
        exp_diffs = np.exp(diffs * 2.0)  # Scale for sensitivity
        priorities = exp_diffs / exp_diffs.sum()

        return priorities


if __name__ == "__main__":
    # Test the gank priority predictor
    model_path = Path(__file__).parent.parent / "models" / "gank_priority_rf.pkl"

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Run: python scripts/train_gank_priority_model.py")
        exit(1)

    predictor = GankPriorityPredictor(model_path)

    print("="*70)
    print("GANK PRIORITY PREDICTION TEST")
    print("="*70)

    # Test scenarios
    scenarios = [
        {
            "name": "Early game - Top lane has level advantage",
            "jungler_level": 3,
            "jungler_gold": 800,
            "jungler_spike": 0.6,
            "ally_top": 0.8, "ally_mid": 0.5, "ally_bot": 0.5,
            "enemy_top": 0.4, "enemy_mid": 0.6, "enemy_bot": 0.6,
            "time": 180
        },
        {
            "name": "Mid game - Bot lane has item spike",
            "jungler_level": 9,
            "jungler_gold": 4500,
            "jungler_spike": 0.7,
            "ally_top": 0.5, "ally_mid": 0.5, "ally_bot": 0.9,
            "enemy_top": 0.6, "enemy_mid": 0.6, "enemy_bot": 0.4,
            "time": 720
        },
        {
            "name": "Late game - Even lanes",
            "jungler_level": 16,
            "jungler_gold": 12000,
            "jungler_spike": 0.8,
            "ally_top": 0.7, "ally_mid": 0.7, "ally_bot": 0.7,
            "enemy_top": 0.7, "enemy_mid": 0.7, "enemy_bot": 0.7,
            "time": 1500
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 70)

        top_p, mid_p, bot_p = predictor.predict_lane_priorities(
            jungler_level=scenario['jungler_level'],
            jungler_gold=scenario['jungler_gold'],
            jungler_spike_score=scenario['jungler_spike'],
            ally_top_spike=scenario['ally_top'],
            ally_mid_spike=scenario['ally_mid'],
            ally_bot_spike=scenario['ally_bot'],
            enemy_top_spike=scenario['enemy_top'],
            enemy_mid_spike=scenario['enemy_mid'],
            enemy_bot_spike=scenario['enemy_bot'],
            game_time_seconds=scenario['time']
        )

        print(f"  Jungler: Level {scenario['jungler_level']}, {scenario['jungler_gold']}g")
        print(f"  Lane Spikes:")
        print(f"    Top: Ally {scenario['ally_top']:.2f} vs Enemy {scenario['enemy_top']:.2f}")
        print(f"    Mid: Ally {scenario['ally_mid']:.2f} vs Enemy {scenario['enemy_mid']:.2f}")
        print(f"    Bot: Ally {scenario['ally_bot']:.2f} vs Enemy {scenario['enemy_bot']:.2f}")
        print(f"\n  Gank Priorities:")
        print(f"    Top: {top_p:.3f} ({top_p*100:.1f}%)")
        print(f"    Mid: {mid_p:.3f} ({mid_p*100:.1f}%)")
        print(f"    Bot: {bot_p:.3f} ({bot_p*100:.1f}%)")

        best_lane = predictor.get_best_gank_lane(
            scenario['jungler_level'], scenario['jungler_gold'],
            scenario['jungler_spike'],
            scenario['ally_top'], scenario['ally_mid'], scenario['ally_bot'],
            scenario['enemy_top'], scenario['enemy_mid'], scenario['enemy_bot'],
            scenario['time']
        )
        print(f"  → Recommended gank: {best_lane.upper()}")

    print(f"\n{'='*70}")
    print("✓ Test complete!")
