"""
Train RL agent (PPO) with behavior cloning warm-start.

This script:
1. Loads the trained BC model
2. Creates a Gymnasium jungle environment
3. Initializes PPO agent with BC weights
4. Trains using PPO to improve beyond imitation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from jungle_gym_env import JungleGymEnv
from jungle_model import JungleNet
from training_data import ALL_CAMPS


class BCWarmStartPolicy:
    """
    Wrapper to initialize PPO policy with BC model weights.
    """

    @staticmethod
    def load_bc_weights(bc_model_path: str, ppo_policy):
        """
        Load BC model weights into PPO policy network.

        The BC model and PPO policy have similar architectures,
        so we can transfer the feature extractor weights.
        """
        print(f"Loading BC model from {bc_model_path}...")

        # Load BC checkpoint
        checkpoint = torch.load(bc_model_path, map_location='cpu')
        bc_state_dict = checkpoint['model_state_dict']

        # Get PPO policy state dict
        ppo_state_dict = ppo_policy.state_dict()

        # Map BC layers to PPO layers
        # BC: network.0 (Linear 71->128), network.3 (Linear 128->128), etc.
        # PPO: mlp_extractor.policy_net.0 (similar structure)

        bc_layer_mapping = {
            'network.0.weight': 'mlp_extractor.policy_net.0.weight',
            'network.0.bias': 'mlp_extractor.policy_net.0.bias',
            'network.3.weight': 'mlp_extractor.policy_net.2.weight',
            'network.3.bias': 'mlp_extractor.policy_net.2.bias',
            'network.6.weight': 'mlp_extractor.policy_net.4.weight',
            'network.6.bias': 'mlp_extractor.policy_net.4.bias',
            # Final action layer
            'network.8.weight': 'action_net.weight',
            'network.8.bias': 'action_net.bias',
        }

        # Transfer weights
        transferred = 0
        for bc_key, ppo_key in bc_layer_mapping.items():
            if bc_key in bc_state_dict and ppo_key in ppo_state_dict:
                ppo_state_dict[ppo_key] = bc_state_dict[bc_key]
                transferred += 1
                print(f"  ✓ Transferred {bc_key} -> {ppo_key}")

        # Load updated state dict
        ppo_policy.load_state_dict(ppo_state_dict)

        print(f"✓ Transferred {transferred} layers from BC model to PPO policy")
        return ppo_policy


def make_env():
    """Create and wrap environment."""
    def _init():
        env = JungleGymEnv(max_episode_steps=50)  # ~10 minutes of game time
        env = Monitor(env)
        return env
    return _init


def main():
    print("="*70)
    print("RL AGENT TRAINING WITH BC WARM-START")
    print("="*70)

    # Paths
    bc_model_path = Path(__file__).parent.parent / "models" / "jungle_net_best.pt"
    output_dir = Path(__file__).parent.parent / "models" / "rl"
    output_dir.mkdir(exist_ok=True, parents=True)

    if not bc_model_path.exists():
        print(f"⚠ BC model not found at {bc_model_path}")
        print("  Run train_behavior_cloning.py first!")
        return

    # Create vectorized environment
    print("\n1. Creating environment...")
    env = DummyVecEnv([make_env()])

    # Create PPO agent
    print("\n2. Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,  # Steps per environment per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy bonus for exploration
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    # Load BC weights into PPO policy
    print("\n3. Loading BC weights...")
    try:
        BCWarmStartPolicy.load_bc_weights(str(bc_model_path), model.policy)
        print("✓ BC warm-start successful!")
    except Exception as e:
        print(f"⚠ Could not load BC weights: {e}")
        print("  Training from scratch instead...")

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_jungle",
    )

    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train
    print("\n4. Training PPO agent...")
    print(f"{'='*70}")

    total_timesteps = 100_000  # Start with 100k steps (~2 hours on CPU)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")

    # Save final model
    final_model_path = output_dir / "ppo_jungle_final.zip"
    model.save(str(final_model_path))

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {output_dir / 'best_model.zip'}")
    print(f"TensorBoard logs: {output_dir / 'tensorboard'}")
    print()
    print("To view training progress:")
    print(f"  tensorboard --logdir {output_dir / 'tensorboard'}")

    # Test the trained agent
    print(f"\n{'='*70}")
    print("TESTING TRAINED AGENT")
    print(f"{'='*70}")

    test_env = JungleGymEnv(render_mode="human")
    obs, info = test_env.reset()

    episode_reward = 0
    steps = 0

    for _ in range(20):  # Run for 20 steps
        action, _states = model.predict(obs, deterministic=True)
        camp_name = ALL_CAMPS[action]

        print(f"\nStep {steps + 1}: Choosing {camp_name}")
        obs, reward, terminated, truncated, info = test_env.step(action)

        episode_reward += reward
        steps += 1

        test_env.render()

        if terminated or truncated:
            print("\nEpisode finished!")
            break

    print(f"\nTotal reward: {episode_reward:.2f}")
    print(f"Camps cleared: {info.get('camps_cleared', 0)}")
    print(f"Gold earned: {info.get('total_gold', 0):.0f}")


if __name__ == "__main__":
    main()
