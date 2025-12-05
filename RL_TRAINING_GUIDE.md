# RL Training Guide

## 🎯 What We Built

You now have a complete reinforcement learning pipeline for League of Legends jungling:

1. **Behavior Cloning Model** (60.5% accuracy, 90.3% top-3)
   - Trained on 34 Challenger jungle matches
   - Predicts which camp to clear next
   - Path: `models/jungle_net_best.pt`

2. **Gymnasium Environment** (`JungleGymEnv`)
   - Simulates jungle clearing with realistic combat
   - 71-dim state space (same as BC model)
   - 17 action space (all camps + objectives)
   - Reward based on gold/XP efficiency

3. **PPO Training Script** with BC warm-start
   - Initializes RL agent with your BC model weights
   - Fine-tunes using Proximal Policy Optimization
   - Saves checkpoints and best model

---

## 🚀 How to Train the RL Agent

### Quick Start (100k steps, ~30 minutes)

```bash
cd "/Users/thomas/Desktop/RL Project/RL-Project"
python scripts/train_rl_agent.py
```

This will:
- Load your BC model weights
- Train for 100k steps (~2000 episodes)
- Save checkpoints every 10k steps
- Save best model based on evaluation
- Log to TensorBoard

### Monitor Training Progress

In a separate terminal:
```bash
cd "/Users/thomas/Desktop/RL Project/RL-Project"
tensorboard --logdir models/rl/tensorboard
```

Then open http://localhost:6006 in your browser.

### Key Metrics to Watch

- **`rollout/ep_rew_mean`**: Average episode reward (should increase)
- **`train/approx_kl`**: KL divergence (should stay low, < 0.1)
- **`train/policy_loss`**: Policy loss (should decrease)
- **`eval/mean_reward`**: Evaluation reward (should increase)

---

## 📊 Expected Performance

| Training Stage | Steps | Time | Expected Reward | Description |
|----------------|-------|------|-----------------|-------------|
| **BC warm-start** | 0 | 0 min | ~5-10 | Initial BC policy |
| **Early RL** | 50k | 15 min | ~15-20 | Learning to avoid bad actions |
| **Mid RL** | 100k | 30 min | ~25-35 | Optimizing efficiency |
| **Late RL** | 500k | 2-3 hrs | ~40-50 | Near-optimal pathing |

**Note:** Random policy gets ~-5 reward, so BC starts way ahead!

---

## 🎮 Testing the Trained Agent

After training, test your agent:

```python
from stable_baselines3 import PPO
from jungle_gym_env import JungleGymEnv

# Load trained model
model = PPO.load("models/rl/best_model.zip")

# Run in environment
env = JungleGymEnv(render_mode="human")
obs, info = env.reset()

for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        break

print(f"Final stats: {info}")
```

---

## ⚙️ Tuning the Environment

### Adjust Episode Length

In `jungle_gym_env.py`:
```python
env = JungleGymEnv(max_episode_steps=100)  # Longer episodes
```

### Modify Reward Function

In `jungle_gym_env.py`, edit `_calculate_reward()`:

```python
def _calculate_reward(self, ...):
    # Current: gold/XP efficiency

    # Add bonus for objective control:
    if "dragon" in camp_name:
        reward += 2.0  # Prioritize dragons

    # Add penalty for inefficient pathing:
    if time_spent > 30.0:
        reward -= 0.5  # Encourage faster clears

    return reward
```

### Change Starting Conditions

```python
env = JungleGymEnv(
    max_episode_steps=50,
    starting_level=3,  # Adjust starting level
)
```

---

## 📈 Scaling Up Training

### Longer Training (Better Performance)

Edit `scripts/train_rl_agent.py`:

```python
total_timesteps = 500_000  # 5x longer (2-3 hours)
```

### Parallel Environments (Faster Training)

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Use multiple parallel environments
env = SubprocVecEnv([make_env() for _ in range(4)])  # 4 parallel
```

### Hyperparameter Tuning

In `scripts/train_rl_agent.py`:

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,  # Lower = more stable
    n_steps=4096,        # More steps = better estimates
    batch_size=128,      # Larger = more stable
    ent_coef=0.02,       # Higher = more exploration
    ...
)
```

---

## 🐛 Troubleshooting

### Agent Dies Too Often

- Reduce starting damage of camps in `spawn_camp()`
- Increase champion HP in `JungleGymEnv.reset()`
- Add HP recovery between camps

### Agent Doesn't Explore

- Increase `ent_coef` (entropy coefficient) in PPO
- Reduce `clip_range` for more policy updates
- Train longer (exploration takes time)

### Training is Slow

- Reduce `max_episode_steps` (shorter episodes)
- Use parallel environments (`SubprocVecEnv`)
- Train on GPU if available

---

## 🎯 Next Steps

1. **Run initial training** (100k steps)
2. **Analyze results** in TensorBoard
3. **Tune reward function** based on behavior
4. **Scale up** to 500k-1M steps
5. **Add complexity:**
   - Lane state awareness
   - Enemy jungler tracking
   - Objective timing
   - Vision control (later)

---

## 📁 Key Files

```
RL-Project/
├── src/
│   ├── jungle_gym_env.py       # Gymnasium environment
│   ├── jungle_model.py          # BC model architecture
│   ├── combatState.py           # Combat simulation
│   └── gameStates.py            # Camp spawn tracking
├── scripts/
│   ├── train_behavior_cloning.py   # BC training
│   └── train_rl_agent.py           # RL training (PPO + BC)
├── models/
│   ├── jungle_net_best.pt      # Trained BC model
│   └── rl/
│       ├── best_model.zip      # Best RL checkpoint
│       └── ppo_jungle_final.zip # Final RL model
└── data/
    └── processed/
        ├── challenger_jungle_data.json  # Raw match data
        ├── states.npy                   # Training states
        └── actions.npy                  # Training actions
```

---

## 🚀 Ready to Train!

You're all set. Run:

```bash
python scripts/train_rl_agent.py
```

And watch your agent learn to jungle like a Challenger player! 🎮
