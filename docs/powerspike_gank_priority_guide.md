# Powerspike-Aware Gank Priority System

## Overview

This system integrates champion powerspike data into the jungler BC/RL pipeline to teach the agent **when** to gank based on lane win-conditions, not just **how** to path efficiently.

### The Problem

Basic jungling models learn camp clearing patterns but don't understand:
- **Champion matchup dynamics**: Kayle is weak early vs Draven is strong early
- **Powerspike timing**: Some champions spike at level 3, others at level 6 or item completion
- **Gank value**: Which lanes are worth investing time vs farming

Without this, agents either:
1. Never gank (pure farming is safer)
2. Perma-gank regardless of lane state (inefficient)

### The Solution

We use a **two-model approach**:

1. **Random Forest Gank Priority Model** - Predicts which lane high-elo junglers choose to gank
   - Input: Game state + powerspike scores for all 10 champions
   - Output: P(gank_top), P(gank_mid), P(gank_bot)
   - Trained on actual high-elo jungler decisions

2. **Enhanced BC/RL Pipeline** - Uses gank priorities for:
   - **State features**: Lane priority scores inform the neural network
   - **Reward shaping**: Bonus rewards for ganking high-priority lanes
   - **Anti-perma-gank**: Penalties for excessive ganking (maintains clear-tempo)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Collection (Patch 14.21+)                           │
│    - Collect 2000+ high-elo jungle matches                  │
│    - Extract powerspike data (levels, items, champions)     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Random Forest Training                                   │
│    - Features: jungler state + lane powerspike scores       │
│    - Target: Which lane was ganked (or farm)                │
│    - Output: Gank priority probabilities                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Enhanced State Representation                            │
│    - Base: position, time, recent camps (83-dim)            │
│    - NEW: Powerspike scores (7-dim)                         │
│    - NEW: Gank priorities (3-dim)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Behavior Cloning + RL                                    │
│    - BC learns baseline pathing from experts                │
│    - RL fine-tunes with reward shaping:                     │
│      • Gank bonus: +0.2 × lane_priority                     │
│      • Farm penalty: -0.3 if gank_ratio > 70%               │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage

### Step 1: Collect Data

```bash
# Collect 2000+ jungle matches from last 5 patches
python scripts/collect_from_leaderboard.py

# Collect powerspike snapshots for those matches
python scripts/collect_powerspike_data.py
```

**Output**:
- `data/processed/challenger_jungle_data.json` (~2000 matches)
- `data/processed/powerspike_match_data.json` (champion states)

### Step 2: Train Gank Priority Model

```bash
python scripts/train_gank_priority_model.py
```

This trains a Random Forest to predict gank lane choices based on:
- Jungler level, gold, spike score
- Lane powerspike differentials (ally vs enemy)
- Game time

**Output**: `models/gank_priority_rf.pkl`

**Expected Performance**:
- ~60-70% accuracy on which lane to gank
- Feature importance: spike differentials > game time > jungler state

### Step 3: Train Enhanced BC Model

The BC model now has an **83-dimensional state space**:

| Feature Group | Dims | Description |
|---------------|------|-------------|
| Time & Position | 3 | Game time, (x, y) position |
| Camp History | 80 | Last camp + recent 3 camps (one-hot) |
| Powerspikes | 7 | Jungler spike + 6 lane spikes |
| **Gank Priorities** | **3** | **Top/Mid/Bot priority scores** |

```python
# The model automatically learns:
# - When Draven is ahead → gank bot priority high
# - When Kayle is level 16 → avoid ganking top
# - When mid has level 6 spike → prioritize mid ganks
```

Train as normal:
```bash
python scripts/train_behavior_cloning.py
```

### Step 4: Train RL Agent with Reward Shaping

The RL environment now uses **powerspike-aware rewards**:

#### Reward Components

1. **Base Reward** (Behavior Cloning)
   - +1.0 for matching expert action
   - -0.1 for mismatch

2. **Gank Priority Bonus** (NEW!)
   - +0 to +0.2 based on lane priority
   - Example: Gank top when top_priority = 0.8 → +0.16 bonus
   - Encourages ganking lanes with powerspike advantage

3. **Farm Efficiency Penalty** (NEW!)
   - -0.3 if gank ratio > 70% of last 10 actions
   - -0.15 if gank ratio > 50%
   - Prevents "perma-gank Draven" failure mode

#### Training

```python
from stable_baselines3 import PPO
from jungle_rl_env import JungleRLEnv
from pathlib import Path

# Create environment with gank priority model
env = JungleRLEnv(
    jungle_data_path=Path("data/processed/challenger_jungle_data.json"),
    powerspike_data_path=Path("data/processed/powerspike_match_data.json"),
    gank_priority_model_path=Path("models/gank_priority_rf.pkl"),
    # Reward shaping hyperparameters
    gank_priority_bonus_weight=0.2,  # Max bonus for high-priority ganks
    farm_efficiency_weight=0.3,  # Penalty for over-ganking
    min_farm_actions=3  # Minimum farm actions per window
)

# Train RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("models/rl/jungle_powerspike_agent")
```

---

## How It Works: Example Scenarios

### Scenario 1: Early Game Draven Advantage

**State**:
- Game time: 4:00
- Jungler: Level 3, 800g
- Bot lane: Draven (ally, 0.8 spike) vs Vayne (enemy, 0.3 spike)
- Top lane: Kayle (ally, 0.3 spike) vs Darius (enemy, 0.7 spike)

**Gank Priority Prediction**:
- Top: 0.15 (avoid - Kayle is weak, Darius strong)
- Mid: 0.25 (neutral)
- **Bot: 0.60 (HIGH - Draven has huge early advantage)**

**Agent Behavior**:
- BC model sees high bot_gank_priority feature → learns to gank bot
- RL agent gets +0.12 reward bonus for ganking bot (0.60 × 0.2)
- Farm penalty = 0 (not over-ganking yet)

### Scenario 2: Late Game Kayle Spike

**State**:
- Game time: 25:00
- Jungler: Level 16, 12000g
- Top lane: Kayle (ally, 0.95 spike at level 16) vs tank (enemy, 0.6 spike)
- Bot lane: Even matchup

**Gank Priority Prediction**:
- **Top: 0.55 (HIGH - Kayle can carry now)**
- Mid: 0.25
- Bot: 0.20

**Agent Behavior**:
- Learned that Kayle level 16 = strong → gank top to enable carry
- +0.11 bonus for top gank

### Scenario 3: Perma-Gank Prevention

**State**:
- Last 10 actions: [gank, gank, gank, farm, gank, gank, gank, gank, farm, gank]
- Gank ratio: 80%

**Reward**:
- Agent ganks again → gets -0.3 farm penalty
- Total reward likely negative even if it's a good gank
- Agent learns to balance ganking with clearing tempo

---

## Configuration

### Reward Shaping Hyperparameters

Adjust in `JungleRLEnv` constructor:

```python
env = JungleRLEnv(
    ...,
    gank_priority_bonus_weight=0.2,  # Default: 0.2
    # Higher → more aggressive ganking
    # Lower → more conservative

    farm_efficiency_weight=0.3,  # Default: 0.3
    # Higher → stronger anti-perma-gank penalty
    # Lower → allows more ganking

    min_farm_actions=3  # Default: 3
    # Minimum farm actions per 10-action window
)
```

**Tuning Tips**:
- If agent farms too much: Increase `gank_priority_bonus_weight` to 0.3-0.4
- If agent perma-ganks: Increase `farm_efficiency_weight` to 0.4-0.5
- For aggressive junglers: Lower both weights
- For farming junglers: Raise `farm_efficiency_weight`, lower bonus

---

## Expected Results

### Random Forest Gank Priority Model

| Metric | Expected Value |
|--------|----------------|
| Train Accuracy | 65-75% |
| Test Accuracy | 60-70% |
| Top-3 Accuracy | 85-90% |

**Important Features** (by importance):
1. Lane spike differentials (30-40%)
2. Game time (15-25%)
3. Jungler spike score (10-15%)
4. Absolute lane spikes (5-10% each)

### Behavior Cloning with Powerspike Features

| Metric | Without Powerspikes | With Powerspikes |
|--------|---------------------|------------------|
| Top-1 Accuracy | 60% | 65-70% |
| Top-3 Accuracy | 90% | 93-95% |
| Gank Prediction | Poor | Good |

**Improvement**: 5-10% boost from lane-aware features

### RL Agent Performance

| Metric | Without Reward Shaping | With Reward Shaping |
|--------|------------------------|---------------------|
| Avg Episode Reward | 15-20 | 25-35 |
| Gank Success Rate | Low (blind ganks) | High (targeted ganks) |
| Farm Efficiency | High (too safe) | Balanced |

**Key Behaviors Learned**:
- Gank losing lanes when they have powerspike advantage
- Avoid weak lanes (e.g., early Kayle)
- Maintain clear tempo between ganks
- Time ganks with level/item spikes

---

## Debugging

### Check Gank Priority Predictions

```python
from gank_priority import GankPriorityPredictor
from pathlib import Path

predictor = GankPriorityPredictor(Path("models/gank_priority_rf.pkl"))

top_p, mid_p, bot_p = predictor.predict_lane_priorities(
    jungler_level=4,
    jungler_gold=1000,
    jungler_spike_score=0.6,
    ally_top_spike=0.3,  # Weak early (Kayle)
    ally_mid_spike=0.6,
    ally_bot_spike=0.8,  # Strong early (Draven)
    enemy_top_spike=0.7,
    enemy_mid_spike=0.6,
    enemy_bot_spike=0.4,
    game_time_seconds=240
)

print(f"Top: {top_p:.2f}, Mid: {mid_p:.2f}, Bot: {bot_p:.2f}")
# Expected: Top: 0.15, Mid: 0.25, Bot: 0.60
```

### Monitor Reward Components

During RL training, log reward info:

```python
obs, reward, done, truncated, info = env.step(action)

print(f"Base: {info['base_reward']:.3f}")
print(f"Gank bonus: {info['gank_bonus']:.3f}")
print(f"Farm penalty: {info['farm_penalty']:.3f}")
print(f"Total: {info['total_reward']:.3f}")
```

### Visualize Gank Patterns

```python
# Collect agent decisions
ganks_by_lane = {"top": 0, "mid": 0, "bot": 0}
for episode in range(100):
    obs, info = env.reset()
    for step in range(20):
        action, _ = model.predict(obs)
        action_name = env.processor.ALL_ACTIONS[action]

        if "gank" in action_name:
            if "top" in action_name:
                ganks_by_lane["top"] += 1
            elif "mid" in action_name:
                ganks_by_lane["mid"] += 1
            elif "bot" in action_name:
                ganks_by_lane["bot"] += 1

        obs, _, done, _, _ = env.step(action)
        if done:
            break

print("Gank distribution:", ganks_by_lane)
# Should reflect lane priorities, not be uniform
```

---

## Limitations & Future Work

### Current Limitations

1. **Lane Assignment Approximation**
   - Uses participant ID heuristics (1=top, 2=jungle, etc.)
   - Real roles can vary (lane swaps, funnel comps)
   - **Solution**: Use position-based role detection

2. **Simplified Powerspike Model**
   - Doesn't account for matchup-specific spikes
   - Treats all "level 6" spikes equally (Malphite ult ≠ Garen)
   - **Solution**: Add matchup-specific multipliers

3. **No Lane State Awareness**
   - Doesn't know lane HP, wave position, vision
   - Can't predict gank success probability
   - **Solution**: Add lane state features from timeline data

4. **Reward Hacking Risk**
   - Agent might learn to "fake gank" for bonus without impact
   - **Solution**: Add gank outcome rewards (kill/assist/flash)

### Future Enhancements

1. **Dynamic Reward Weights**
   - Adjust weights based on game time
   - Early: emphasize farming, Late: emphasize objectives

2. **Multi-Objective RL**
   - Separate value heads for farming, ganking, objectives
   - Learn trade-offs dynamically

3. **Gank Outcome Prediction**
   - Train classifier: P(kill | gank, state)
   - Only reward successful ganks

4. **Champion-Specific Policies**
   - Train specialist models: Lee Sin (early gank), Karthus (farm)
   - Use champion as context in policy network

---

## Files Created/Modified

### New Files
- `scripts/train_gank_priority_model.py` - Train Random Forest
- `src/gank_priority.py` - Gank priority predictor module
- `docs/powerspike_gank_priority_guide.md` - This guide

### Modified Files
- `scripts/collect_from_leaderboard.py` - Increased data collection (2000+ matches)
- `src/training_data.py` - Enhanced state vector with powerspike + priority features
- `src/jungle_rl_env.py` - Added reward shaping with gank bonuses + farm penalties
- `src/jungle_model.py` - (Auto-adjusts to new state dim)

---

## Quick Start Checklist

- [ ] Run `python scripts/collect_from_leaderboard.py` (collect 2000+ matches)
- [ ] Run `python scripts/collect_powerspike_data.py` (get champion states)
- [ ] Run `python scripts/train_gank_priority_model.py` (train RF model)
- [ ] Run `python scripts/train_behavior_cloning.py` (train BC with new features)
- [ ] Run `python scripts/train_rl_agent.py` (train RL with reward shaping)
- [ ] Evaluate agent performance on test matches
- [ ] Tune reward weights if needed

---

## Questions?

See the main `README.md` or `RL_TRAINING_GUIDE.md` for general project info.

For powerspike system details, see `src/powerspike_system.py` docstrings.

**Key Insight**: This system doesn't just teach the agent **how** to jungle, it teaches **why** certain ganks are worth it based on champion matchups and power levels. This is the difference between a Gold and Diamond jungler.
