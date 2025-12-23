# League of Legends Jungle ML Project

Machine Learning models for League of Legends jungling, trained on high-elo replay data using computer vision and automated screenshot extraction.

## 🎯 Project Goal

Train ML models to make high-level jungling decisions using **complete game state data** extracted from replays:
- Optimal jungle pathing and camp clearing
- Gank timing based on lane states and fog of war
- Objective control (Dragon, Herald, Baron, Atakhan, Void Grubs)
- Gold/XP efficiency optimization
- Information asymmetry (visible vs. true enemy positions)

**Target Performance**: High-elo decision making using ensemble models

## 🏗️ Project Status

Currently in **Phase 1: Automated Replay Screenshot Collection Pipeline**

- ✅ Core jungle simulation (camp clearing, HP/damage)
- ✅ Monster scaling system
- ✅ Map geometry and travel times
- ✅ Lane state and gank heuristics
- 🚧 Screenshot-based data collection pipeline
- 🚧 Computer vision extraction system
- ⏳ Partition-based training workflow
- ⏳ Hyperparameter tuning with K-fold CV
- ⏳ Final model training (XGBoost, Extra Trees, Neural Networks)

## 📁 Project Structure

```
RL-Project/
├── src/                    # Core simulation & data processing
│   ├── combatState.py      # Camp clear simulation
│   ├── monster_scaling.py  # Dynamic monster stats
│   ├── map_geometry.py     # Travel time calculations
│   ├── gameStates.py       # Lane states, monster spawns
│   ├── jungle_env.py       # Environment integration
│   ├── envSim.py           # Demo simulation
│   ├── RiotAPIs.py         # Riot Data Dragon API client
│   └── (future) cv_extraction.py  # Computer vision pipeline
├── scripts/                # Collection & training scripts
│   ├── collect_replays.py  # Automated replay screenshot collection
│   ├── process_partition.py # CV extraction for partition
│   └── train_model.py      # Model training pipeline
├── docs/                   # Documentation
│   ├── data_collection_options.md  # Original data strategy analysis
│   └── screenshot_pipeline.md      # New CV-based approach
├── data/                   # Training data (gitignored)
│   ├── partitions/         # Partition-based collection
│   │   ├── partition_001/
│   │   │   ├── raw/        # Screenshots (deleted after processing)
│   │   │   └── processed/  # Extracted CSV data
│   │   └── partition_NNN/
│   ├── models/             # Trained models per partition + final
│   └── full_dataset.csv    # Merged data from all partitions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- League of Legends client (for replay collection)
- League of Legends accounts in target regions (NA, EUW, KR)
- ~200 GB free disk space (temporary, during partition processing)
- Understanding of jungling and LoL game mechanics

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

### Running Data Collection

```bash
# Collect one partition of replays (automated)
python scripts/collect_partition.py --region NA --partition 1 --games 100

# Process partition with computer vision
python scripts/process_partition.py --partition 1

# Train on completed partitions
python scripts/train_model.py --partitions 1-5
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

**NEW APPROACH**: Screenshot-based automated replay collection with computer vision extraction.

### Why Screenshots Over API?
- **Complete data**: All visible information (positions, items, gold, HP/mana, cooldowns, fog of war)
- **Fog of war tracking**: Both true positions and last-known positions
- **All epic monsters**: Dragon, Baron, Herald, Atakhan, Void Grubs
- **5-second intervals**: Much better temporal resolution than API (60s)
- **No API limitations**: Get everything visible on screen

### Collection Pipeline
1. **Automated replay navigation** (jump backwards in 5s intervals from end)
2. **Two screenshots per timestamp**:
   - Image 1: Fog of war ON + Tab (visible enemy positions)
   - Image 2: Fog of war OFF + Tab + X (true positions + gold data)
3. **Computer vision extraction**:
   - Champion positions from minimap (color blob detection)
   - Items, stats, gold from scoreboard (template matching + OCR)
   - Epic monster status (visual detection + API correlation)
   - Health/mana bars, ultimate/summoner cooldowns
4. **Partition-based processing** (100 games per partition):
   - Collect → Process → Save CSV → Delete images
   - Accumulate CSVs across partitions
   - Train final model on merged dataset

### Storage Efficiency
- **During partition**: ~180 GB (screenshots, temporary)
- **After processing**: ~72 MB (CSV per 100 games)
- **Final dataset**: ~360 MB for 500 games
- **No cloud storage needed**

See [docs/data_collection_options.md](docs/data_collection_options.md) for original API-based analysis.

## 🛠️ Tech Stack

- **Simulation**: Python, NumPy
- **Computer Vision**: OpenCV, pytesseract (OCR)
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM
- **Neural Networks**: PyTorch
- **Hyperparameter Tuning**: Optuna (Bayesian optimization)
- **Data Processing**: pandas, NumPy
- **Automation**: pyautogui, pynput (replay control)
- **Data Source**: League of Legends replays + Riot API (metadata)

## 📈 Roadmap

### Phase 1: Automated Screenshot Collection (Current)
- [ ] Build replay navigation automation (pyautogui)
- [ ] Implement screenshot capture at 5s intervals
- [ ] Set up fog of war toggle automation
- [ ] Test on 1-2 replays for validation
- [ ] Scale to partition-based collection (100 games/partition)

### Phase 2: Computer Vision Pipeline
- [ ] Champion position detection (minimap blob detection)
- [ ] Champion identification (portrait template matching)
- [ ] Item extraction (template matching + OCR)
- [ ] Gold/stats OCR (scoreboard parsing)
- [ ] Health/mana bar reading
- [ ] Ultimate/summoner cooldown detection
- [ ] Epic monster status tracking

### Phase 3: Partition-Based Data Collection
- [ ] Collect Partition 1 (100 games, ~33 hours)
- [ ] Process Partition 1 → CSV
- [ ] Collect & process Partitions 2-5
- [ ] Merge all partition CSVs
- [ ] Final dataset: 500 games, ~1.8M training examples

### Phase 4: Hyperparameter Tuning
- [ ] Set up Optuna Bayesian optimization
- [ ] Define search space (NN, XGBoost, Extra Trees)
- [ ] Run K-fold CV using partition splits
- [ ] Select best model configurations

### Phase 5: Final Model Training
- [ ] Train XGBoost on full dataset
- [ ] Train Extra Trees on full dataset
- [ ] Train deep neural network on full dataset
- [ ] Build stacked ensemble (optional)
- [ ] Evaluate on held-out test set

### Phase 6: Deployment & Analysis
- [ ] Model evaluation and performance metrics
- [ ] Feature importance analysis
- [ ] Decision visualization
- [ ] Documentation of results

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
