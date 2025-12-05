# League of Legends Jungle RL Project

Reinforcement Learning agent for League of Legends jungling, focusing on macro decisions and optimal pathing.

## 🎯 Project Goal

Train an RL agent to make high-level jungling decisions:
- Optimal jungle pathing and camp clearing
- Gank timing based on lane states
- Objective control (Dragon, Herald, Baron)
- Gold/XP efficiency optimization

**Target Performance**: Gold-Plat level decision making

## 🏗️ Project Status

Currently in **Phase 1: Simulation & Data Collection**

- ✅ Core jungle simulation (camp clearing, HP/damage)
- ✅ Monster scaling system
- ✅ Map geometry and travel times
- ✅ Lane state and gank heuristics
- 🚧 Data collection pipeline
- ⏳ RL training environment
- ⏳ Behavior cloning from high-elo data
- ⏳ PPO/DQN fine-tuning

## 📁 Project Structure

```
RL-Project/
├── src/                    # Core simulation code
│   ├── combatState.py      # Camp clear simulation
│   ├── monster_scaling.py  # Dynamic monster stats
│   ├── map_geometry.py     # Travel time calculations
│   ├── gameStates.py       # Lane states, monster spawns
│   ├── jungle_env.py       # Environment integration
│   ├── envSim.py           # Demo simulation
│   └── RiotAPIs.py         # Riot Data Dragon API client
├── scripts/                # Utility scripts
│   └── test_riot_api.py    # API data exploration
├── docs/                   # Documentation
│   └── data_collection_options.md  # Data strategy analysis
├── data/                   # Training data (gitignored)
│   ├── raw/                # Raw match timelines
│   └── processed/          # Preprocessed training data
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- League of Legends understanding (jungling knowledge helpful)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RL-Project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Simulation

```bash
# Test camp clearing mechanics
python src/jungle_env.py

# Test environment simulation
python src/envSim.py
```

## 🧪 Current Simulation Features

### Combat System
- HP/damage modeling
- Smite mechanics (damage + heal)
- Camp-specific stats (HP, armor, DPS)
- Death tracking for both jungler and camps

### Monster Scaling
- Level-based stat interpolation
- 8 jungle camps (Blue, Red, Gromp, Wolves, Raptors, Krugs, Scuttle)
- 4 epic objectives (Dragon, Herald, Baron, Void Grubs)
- Accurate respawn timers

### Map Geometry
- Travel time calculations between camps
- Shortest path finding (Dijkstra)
- Recall + travel time simulation
- Movement speed scaling

### Game State
- Lane state abstraction (pushed, even, under tower)
- Win probability for gank opportunities
- Heuristic gank policy (baseline)
- Epic monster spawn/respawn tracking

## 📊 Data Collection Strategy

See [docs/data_collection_options.md](docs/data_collection_options.md) for detailed analysis.

**Current Plan**: Hybrid approach
1. **Phase 1**: Riot API + inference heuristics for camp clears
2. **Phase 2**: (If needed) Computer vision on replay recordings for perfect data

## 🛠️ Tech Stack

- **Simulation**: Python, NumPy
- **RL Framework**: Stable-Baselines3 (PPO/DQN)
- **Neural Networks**: PyTorch
- **Data Source**: Riot Games API
- **Training**: AWS (planned)

## 📈 Roadmap

### Phase 1: Data Collection (Current)
- [ ] Set up Riot API data pipeline
- [ ] Download 1000+ high-elo match timelines
- [ ] Build camp inference heuristics
- [ ] Extract state-action pairs

### Phase 2: Environment Setup
- [ ] Create Gymnasium-compatible environment
- [ ] Define state space (~1500 dimensions)
- [ ] Define action space (18 discrete actions)
- [ ] Implement reward function

### Phase 3: Behavior Cloning
- [ ] Train PyTorch network on expert data
- [ ] Validate imitation accuracy
- [ ] Establish baseline performance

### Phase 4: RL Fine-Tuning
- [ ] PPO training on top of behavior cloning
- [ ] Hyperparameter tuning
- [ ] Evaluation against heuristics

### Phase 5: Deployment
- [ ] AWS training pipeline
- [ ] Model checkpointing
- [ ] Performance analysis

## 🤝 Contributing

This is a personal research project, but feedback and suggestions are welcome!

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Riot Games for Data Dragon API
- League of Legends community for game knowledge
- Stable-Baselines3 team for RL framework

## 📧 Contact

[Your contact info here]

---

**Note**: This project is for educational/research purposes. It does not interact with live League of Legends games or servers.
