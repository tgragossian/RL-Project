# Screenshot-Based Data Collection Pipeline

**Status**: Current approach (December 2024)

This document describes the automated screenshot collection and computer vision extraction pipeline for gathering high-quality League of Legends training data from replays.

---

## Overview

### Why Screenshots Over API?

The Riot API has significant limitations:
- ❌ Only 60-second temporal resolution
- ❌ Missing individual camp clear data
- ❌ No fog of war information
- ❌ Limited epic monster details
- ❌ No cooldown tracking
- ❌ No health/mana states

**Our screenshot approach captures everything:**
- ✅ 5-second intervals (12x better resolution)
- ✅ Complete champion positions (minimap)
- ✅ Fog of war state (both visible and true positions)
- ✅ All items, gold, stats
- ✅ Health/mana bars
- ✅ Ultimate and summoner spell cooldowns
- ✅ All epic monsters (Dragon, Baron, Herald, Atakhan, Void Grubs)

---

## Collection Workflow

### Phase 1: Automated Replay Navigation

Using `pyautogui` or `pynput` to control the League client:

1. Load replay in spectator mode
2. Jump to end of game
3. Click "-15 seconds" button
4. Take screenshots (fog ON, fog OFF)
5. Click "-10 seconds" button
6. Repeat until game start

**Timing**: 5-second intervals by using: end, end-5, end-10, end-15, end-20, etc.

**Speed**: ~2 seconds per timestamp (includes navigation + screenshots)

**Total per game**: ~12 minutes for 30-minute game with 5s intervals

### Phase 2: Screenshot Capture

**Two images per timestamp:**

#### Image 1: Fog of War ON + Tab
- Minimap shows only visible enemies
- Scoreboard shows items, CS, KDA, vision score, level
- Captures what the team actually knows (perceived game state)

#### Image 2: Fog of War OFF + Tab + X
- Minimap shows ALL champion positions (ground truth)
- Press X to show gold columns instead of items
- Current gold and total gold for all players
- Captures true game state

**Resolution**: 1920×1080 (standard)

**File format**: PNG (lossless, good compression)

**Storage**: ~2.5 MB per image × 2 images × 360 timestamps = ~1.8 GB per game

---

## Computer Vision Extraction

### 1. Champion Position Detection

**Source**: Minimap (bottom-left or bottom-right corner)

**Method**: Color blob detection
```
Blue team: HSV range [100-130, 150-255, 150-255]
Red team: HSV range [0-10, 150-255, 150-255]

For each colored dot:
  - Find contours
  - Get centroid (x, y)
  - Convert from minimap pixels to game coordinates
```

**Difficulty**: ⭐ Easy (basic CV)

**Output**: 10 positions per timestamp

### 2. Champion Identity Matching

**Challenge**: Minimap only shows colored dots, not which champion is which

**Solution**: Multi-object tracking with momentum
```
For each timestamp:
  1. Predict where each champion should be (based on previous velocity)
  2. Match detected positions to predictions (Hungarian algorithm)
  3. Update positions and velocities
```

**Bootstrap** (first timestamp at 1:30 game time):
- Use lane position heuristics (top lane = top of map, etc.)
- Use API to get roles
- Match roles to positions

**Difficulty**: ⭐⭐⭐ Medium (well-solved tracking problem)

### 3. Item Extraction

**Source**: Scoreboard item slots (6 slots + trinket per champion)

**Method**: Template matching
```
For each item slot:
  - Crop fixed region
  - Match against database of ~200 item icons
  - Find best match (normalized cross-correlation)
```

**Database**: Extract item icons from Data Dragon API once

**Difficulty**: ⭐⭐ Easy-Medium (fixed UI positions)

**Output**: 70 item slots per timestamp (10 champions × 7 slots)

### 4. Stats Extraction (OCR)

**Source**: Scoreboard rows

**Method**: pytesseract OCR on fixed regions

**Values to extract**:
- CS (minion kills)
- Vision score
- Level
- KDA (kills/deaths/assists)
- Current gold (when X pressed)
- Total gold (when X pressed)

**Difficulty**: ⭐⭐ Medium (OCR on clean UI text)

**Optimization**: Pre-process images (threshold, increase contrast) for better OCR

### 5. Health/Mana Bar Reading

**Source**: Champion portraits in scoreboard

**Method**: Color-based fill percentage
```
For each champion portrait:
  - Crop health bar region (green bar below portrait)
  - Count green pixels / total width = health %
  - Crop mana bar region (blue bar)
  - Count blue pixels / total width = mana %
```

**Difficulty**: ⭐⭐ Easy-Medium (color detection)

**Output**: Health % and mana % for all 10 champions

### 6. Cooldown Detection

**Source**: Scoreboard or champion UI

**Method**: Icon detection

**Tracked**:
- Ultimate status (green dot = ready, no dot = on CD)
- Summoner spell 1 (Flash) status
- Summoner spell 2 (Teleport, Ignite, etc.) status

**Optional**: Q/W/E cooldowns (harder, requires clicking each champion)

**Difficulty**: ⭐⭐⭐ Medium (icon detection)

**Simplification**: Boolean ready/not ready vs. exact seconds remaining

### 7. Epic Monster Tracking

**Sources**: Multiple
1. API timeline (exact kill timestamps)
2. Buff icons on champion portraits (dragon buffs visible)
3. Map visuals (Atakhan spawns, soul effects)
4. Total gold spikes (Baron = ~1500g team gold increase)

**Monsters tracked**:
- Dragon (type, count per team, soul status)
- Elder Dragon (spawns after soul)
- Baron Nashor
- Rift Herald (track who has Eye in inventory)
- Atakhan (visible on map at 19:00+)
- Void Grubs (small gold increase, API correlation)

**Difficulty**: ⭐⭐⭐⭐ Hard (multi-modal detection)

### 8. Game Timer & Metadata

**Source**: Top UI

**Method**: OCR

**Values**:
- Game time (MM:SS)
- Dragon count (blue/red)
- Turret count
- Team gold totals (if visible)

**Difficulty**: ⭐⭐ Easy (OCR on large, clear text)

---

## Data Structure (Per Timestamp)

```json
{
  "timestamp": 300,
  "game_time": "5:00",

  "champions": [
    {
      "id": "LeeSin",
      "name": "Lee Sin",
      "team": "blue",
      "role": "jungle",

      "true_position": {"x": 1250, "y": 3400},
      "last_known_position": {"x": 1250, "y": 3400},
      "last_seen_timestamp": 300,

      "health_percent": 0.85,
      "mana_percent": 0.60,
      "level": 7,

      "ultimate_ready": true,
      "flash_ready": false,
      "summoner2_ready": true,

      "current_gold": 1850,
      "total_gold": 8200,
      "cs": 52,
      "vision_score": 12,
      "kills": 2,
      "deaths": 1,
      "assists": 3,

      "items": [
        {"slot": 0, "id": "item_1400"},
        {"slot": 1, "id": "item_3134"},
        {"slot": 2, "id": "item_1036"},
        {"slot": 3, "id": null},
        {"slot": 4, "id": null},
        {"slot": 5, "id": null},
        {"slot": 6, "id": "item_3340"}
      ]
    }
    // ... 9 more champions
  ],

  "objectives": {
    "dragon_count_blue": 1,
    "dragon_count_red": 0,
    "soul_active": false,
    "soul_type": "infernal",

    "baron_alive": false,
    "herald_eye_holder": "LeeSin",
    "atakhan_visible": false
  }
}
```

---

## Partition-Based Processing

### Strategy: Minimize Storage, Maximize Throughput

**Problem**: 500 games × 1.8 GB = 900 GB (too much!)

**Solution**: Process in partitions, delete images immediately after extraction

### Partition Workflow

```
Partition 1 (100 games):
  1. Collect 100 games → ~180 GB images
  2. Process each game:
     - Extract data to CSV rows
     - Delete images immediately
  3. Save partition_001.csv (~72 MB)
  4. Delete raw/ directory

Partition 2 (100 games):
  1. Collect 100 games → ~180 GB images
  2. Process → partition_002.csv
  3. Delete images

... repeat for K partitions

Final:
  1. Merge all partition CSVs
  2. Train final model on full dataset
```

**Peak storage**: ~180 GB (only during one partition)

**Final storage**: ~360 MB (all CSVs) + ~500 MB (models) = **~1 GB total**

### Timeline for 500 Games

**Collection** (parallel across 3 regions):
- 500 games ÷ 3 regions = 167 games per region
- 167 games × 12 min/game = 2,000 min = **33 hours per region**
- **Total: 33 hours** (parallel)

**Processing** (8 parallel workers):
- 100 games × 6 min / 8 workers = 75 minutes per partition
- 5 partitions = **6.25 hours total**

**Grand total: ~40 hours** (less than 2 days if run continuously)

---

## Training Strategy

### Why One Model > Ensemble of Partition Models

**Ensemble approach (worse)**:
- Each model sees only 100 games (20% of data)
- Limited pattern learning
- Rare events only seen 1-2 times per model
- Requires K forward passes at inference

**Unified model approach (better)**:
- Sees all 500 games = 1.8M training examples
- Strong statistics on all patterns
- Can use deeper architectures (more data = more capacity)
- Single forward pass at inference

### Using Partitions for K-Fold Cross-Validation

The partitions become natural CV folds:

```
Hyperparameter Search:
  For each config in search space:
    Fold 1: Train on [P2,P3,P4,P5], validate on [P1]
    Fold 2: Train on [P1,P3,P4,P5], validate on [P2]
    Fold 3: Train on [P1,P2,P4,P5], validate on [P3]
    Fold 4: Train on [P1,P2,P3,P5], validate on [P4]
    Fold 5: Train on [P1,P2,P3,P4], validate on [P5]

    Average validation scores
    Track best config

Final Training:
  Use best hyperparameters
  Train on ALL partitions merged
  Evaluate on held-out test set (or fresh games)
```

### Model Architectures

**With 1.8M training examples, we can use:**

1. **Deep Neural Network**
   - Input: 2000-3000 features
   - Architecture: [1024] → [512] → [256] → [128] → [output]
   - Regularization: Dropout, batch norm
   - Training: Adam optimizer, learning rate scheduling

2. **XGBoost**
   - Trees: 1000-2000
   - Max depth: 7-10
   - Learning rate: 0.01-0.1
   - Best for tabular data

3. **Extra Trees**
   - Trees: 500+
   - Excellent for feature importance analysis

4. **LightGBM**
   - Faster than XGBoost
   - Similar performance

5. **Stacked Ensemble** (optional)
   - Level 0: XGBoost, Extra Trees, LightGBM, Neural Net
   - Level 1: Simple logistic regression or small NN
   - Combines strengths of all models

---

## Implementation Checklist

### Phase 1: Automation Setup
- [ ] Replay navigation (jump backwards)
- [ ] Fog of war toggle
- [ ] Tab/X key presses
- [ ] Screenshot capture with proper timing
- [ ] File naming convention (match_id, timestamp, fog state)

### Phase 2: CV Pipeline
- [ ] Minimap extraction and preprocessing
- [ ] Color blob detection for champion positions
- [ ] Champion tracking with Hungarian algorithm
- [ ] Template matching for items
- [ ] OCR for stats (gold, CS, vision score)
- [ ] Health/mana bar reading
- [ ] Cooldown detection
- [ ] Epic monster tracking

### Phase 3: Data Processing
- [ ] Partition manager (create, collect, process, cleanup)
- [ ] CSV schema definition
- [ ] Data validation (sanity checks)
- [ ] Progress tracking and logging

### Phase 4: Training Pipeline
- [ ] CSV merging across partitions
- [ ] Train/val/test split (or K-fold setup)
- [ ] Hyperparameter search with Optuna
- [ ] Model training scripts (XGBoost, NN, etc.)
- [ ] Model evaluation and metrics

---

## Challenges & Solutions

### Challenge 1: Champion Identity Matching
**Problem**: Minimap shows dots, not which champion is which

**Solution**: Track movement frame-to-frame with Hungarian algorithm + bootstrap from lane positions

### Challenge 2: Overlapping Champions
**Problem**: In teamfights, dots overlap on minimap

**Solution**: Use momentum/velocity to predict positions, match nearest prediction

### Challenge 3: Epic Monster Detection
**Problem**: Can't always see when monsters die

**Solution**: Multi-modal approach (API timestamps + buff icons + gold spikes)

### Challenge 4: OCR Accuracy
**Problem**: Small text, varying fonts

**Solution**: Pre-process images (threshold, contrast), use large clear UI elements, validate against known ranges

### Challenge 5: Processing Speed
**Problem**: 500 games × 360 timestamps = 180k images to process

**Solution**: Parallelize across CPU cores, use efficient CV operations, process incrementally per partition

---

## Storage Breakdown

### Temporary (per partition)
- 100 games × 360 timestamps × 2 images × 2.5 MB = **180 GB**
- Deleted immediately after processing

### Permanent (cumulative)
- Partition CSVs: 5 × 72 MB = **360 MB**
- Merged dataset: **360 MB**
- Trained models: **500 MB - 2 GB**
- **Total: ~1-2.5 GB**

### Processing Speed
- CV extraction: ~1 second per timestamp
- 360 timestamps per game = 6 minutes per game
- 100 games ÷ 8 workers = 75 minutes per partition

---

## Next Steps

1. Build proof-of-concept on 1 game
2. Validate CV extraction accuracy
3. Implement partition manager
4. Collect Partition 1 (100 games)
5. Process → CSV
6. Repeat for remaining partitions
7. Hyperparameter tuning
8. Final model training

---

## Resources

- **OpenCV Docs**: https://docs.opencv.org/
- **pytesseract**: https://github.com/madmaze/pytesseract
- **PyAutoGUI**: https://pyautogui.readthedocs.io/
- **Optuna**: https://optuna.readthedocs.io/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Hungarian Algorithm**: scipy.optimize.linear_sum_assignment
