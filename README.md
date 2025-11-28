# RL Jungle Project
# League of Legends Jungle RL (WIP)

This project explores using reinforcement learning to control a jungler in a League of Legends–inspired environment.

The goal is **not** to build a bot that plays on Riot's servers, but to train agents inside a custom Python simulation that abstracts jungle mechanics (camps, time, HP, gold, XP, etc.).

## Current Stage: Project 1 – Amumu Full Clear

**Objective:**  
Train a PPO agent to learn an efficient early-game full clear for Amumu in a simplified jungle environment.

### Key ideas

- Custom `gymnasium` environment (`AmumuFullClearEnv`) with:
  - 6 camps (Blue, Gromp, Wolves, Raptors, Red, Krugs)
  - State includes HP, time, and which camps are cleared
  - Actions choose which camp to clear next
  - Reward balances:
    - camp value (gold/xp)
    - time cost
    - HP cost
    - bonus for successful full clear

- Agent:
  - Algorithm: PPO (Stable-Baselines3)
  - Trained entirely in simulation

### Tech stack

- Python 3.x  
- `gymnasium`
- `stable-baselines3`
- `numpy`

## Setup

```bash
# (optional) create and activate a venv/conda env first

pip install -r requirements.txt  # if you add one later

# or manually:
pip install gymnasium stable-baselines3 numpy
