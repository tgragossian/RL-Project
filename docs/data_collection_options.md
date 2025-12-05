# Data Collection Strategy Comparison

Based on research into Riot API, replay files, and scraping options.

---

## Option 1: Riot Official API ⭐⭐⭐☆☆

### What You Get
- ✅ Position snapshots every 60 seconds (x, y coordinates)
- ✅ Gold, XP, level per minute
- ✅ Cumulative jungle minions killed
- ✅ Elite monster kills (Dragon, Baron, Herald, Void Grubs)
- ✅ Champion kills, item purchases
- ❌ **Individual camp clears removed by Riot**
- ❌ Only 1-minute granularity

### Implementation
```python
# Free API key from developer.riotgames.com
# Rate limits: 20 requests/second (development key)
# Can request production key for higher limits

import requests

url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
headers = {"X-Riot-Token": API_KEY}
timeline = requests.get(url, headers=headers).json()
```

### Pros
- Easy to implement (REST API)
- Free and legal
- Reliable data
- Can query thousands of high-elo games
- Good for macro decisions (ganks, objectives)

### Cons
- **Missing exact camp clear data**
- Low temporal resolution (60s intervals)
- Must infer camps from position + minion count
- ~20-30% uncertainty in exact pathing

### Best For
- Training on **macro decisions** (gank timings, objective control)
- Getting initial dataset quickly
- Proof of concept

---

## Option 2: .ROFL Replay File Parsing ⭐⭐☆☆☆

### What You Get
- ✅ Metadata (players, champions, outcome)
- ✅ Can launch replay in client
- ❌ **Game packets are obfuscated** (anti-cheat)
- ❌ Obfuscation changes every patch

### Research Findings
From [roflxd GitHub](https://github.com/fraxiinus/roflxd) and [ROFL parser](https://github.com/Mowokuma/ROFL):
- Replay files contain encrypted game packets
- Only metadata is easily parsable
- Packet payload requires reverse engineering
- **Most parsers archived or only support old patches**

### Recent Development
From [League Data Scraping blog](https://maknee.github.io/blog/2025/League-Data-Scraping/):
- Developer released 1.4M replays on Hugging Face
- Built custom emulator to decrypt/decompress packets
- 15-minute replay → 135MB JSON in ~3 seconds
- **This is extremely complex engineering**

### Pros
- Complete game state if you can decrypt it
- Replays downloadable from Riot client
- Perfect temporal resolution

### Cons
- **Extremely difficult** (packet obfuscation)
- Breaks every patch (14-day update cycle)
- Requires game client to download replays
- Would need to build/maintain emulator
- Legal gray area

### Best For
- Research projects with dedicated team
- Not recommended for solo projects

---

## Option 3: Third-Party Site Scraping (OP.GG, U.GG) ⭐⭐⭐⭐☆

### What You Get
From [OP.GG Help Center](https://help.op.gg/hc/en-us/articles/31091405109401):
- ✅ Same data as Riot API (they use official API)
- ✅ Pre-processed visualizations
- ✅ Timeline graphs (gold, XP, objectives)
- ❌ Still no individual camp data (API limitation)

### NEW: OP.GG MCP Server (2025)
From [OP.GG Gaming Data MCP Server](https://skywork.ai/skypage/en/real-time-game-insights/1981912306334928896):
- Official OP.GG GraphQL API
- TypeScript/Node.js interface
- Structured data access
- **This is basically Riot API with better UI**

### Pros
- Web scraping is allowed (non-commercial)
- Could extract additional analysis
- Multiple sites to cross-reference

### Cons
- **Same data limitations as Riot API**
- Rate limiting + anti-scraping measures
- More fragile than official API
- Still missing camp-level data

### Best For
- If you need OP.GG's derived metrics
- Not worth it if you can use Riot API directly

---

## Option 4: Computer Vision on Replay Recording ⭐⭐⭐⭐⭐

### What You Get
- ✅ **EVERYTHING visible on screen**
- ✅ Exact camp clears in real-time
- ✅ Minimap positions
- ✅ HP/mana bars, gold counts
- ✅ Lane states, ward placements
- ✅ Fog of war state

### Implementation Strategy

#### Step 1: Record Replays
```bash
# Watch replay in client at 8x speed
# Use OBS or game capture to record screen
# 1 game (~30 min at 8x) = 3.75 minutes recording
# 1000 games = ~62 hours of recording (automated)
```

#### Step 2: Computer Vision Pipeline
```python
import cv2
import pytesseract
from PIL import Image

# Extract from minimap
def detect_jungler_position(frame):
    minimap = frame[y1:y2, x1:x2]  # Crop minimap region
    # Use color detection for champion icon
    # Return (x, y) coordinates

def detect_jungle_camp_status(frame):
    # Check camp indicator on minimap
    # Camps show as icons when alive
    # Return dict of {camp_name: alive/dead}

def extract_game_timer(frame):
    # OCR on timer region
    return pytesseract.image_to_string(timer_region)

def extract_gold(frame):
    # OCR on gold display
    return int(pytesseract.image_to_string(gold_region))
```

#### Step 3: State Extraction
```python
# Process recorded video frame-by-frame
for frame in video:
    state = {
        'time': extract_game_timer(frame),
        'jungler_pos': detect_jungler_position(frame),
        'camps_alive': detect_jungle_camp_status(frame),
        'gold': extract_gold(frame),
        'hp': extract_hp_bar(frame),
        'lane_states': detect_lane_positions(frame),
    }

    # Detect state transitions
    if camps_alive['blue_buff'] != prev_state['blue_buff']:
        action = 'CLEAR_BLUE_BUFF'
        training_data.append((prev_state, action, state))
```

### Challenges
1. **UI Detection Robustness**
   - Minimap location fixed, but UI changes
   - Need to handle different resolutions
   - Patch updates may move UI elements

2. **OCR Accuracy**
   - Game timer, gold, CS counts
   - Need clean font recognition
   - Can validate against known ranges

3. **Fog of War**
   - Can only see what jungler sees
   - Enemy jungler position unknown (realistic!)
   - This is actually good for training

4. **Automation**
   - Need to automate: download replay → launch → record → process
   - Riot client API/automation required
   - Could use AutoHotkey/pyautogui

### Pros
- **Complete ground truth data**
- Exact camp clear timings
- Natural fog of war (matches RL training)
- Can capture decision context visually
- Future-proof (works regardless of API changes)

### Cons
- Engineering complexity (CV pipeline)
- Need to record many games
- Potential errors in detection
- More time to implement
- Requires game client + automation

### Best For
- **High-quality training data**
- When exact pathing matters
- Research projects with CV expertise
- **My recommendation if you want best results**

---

## Option 5: Hybrid Approach ⭐⭐⭐⭐⭐ (RECOMMENDED)

### Strategy: Start Simple, Add Complexity

**Phase 1: Riot API Baseline (Week 1-2)**
```python
# Quick implementation
# Train on macro decisions (ganks, objectives)
# Establish baseline performance
# Cost: Free, Time: Fast
```

**Phase 2: Infer Camps from API (Week 3-4)**
```python
# Use position + jungle CS delta to infer camps
# Example:
#   t=90s: jg_minions=0, pos=(blue_buff_coords)
#   t=150s: jg_minions=6, pos=(gromp_coords)
#   → Inferred: Cleared Blue buff
# ~70-80% accuracy possible with good heuristics
```

**Phase 3: CV for Fine-Tuning (Week 5-8, if needed)**
```python
# If API baseline is insufficient:
# Record 100-200 high-quality games with CV
# Use for fine-tuning / validation
# Mix API (quantity) + CV (quality) data
```

### Implementation Timeline

| Week | Task | Output |
|------|------|--------|
| 1 | Get API key, download 1000 match timelines | Raw API data |
| 2 | Build inference heuristics for camps | ~1000 games with inferred paths |
| 3 | Train baseline behavior cloning model | 70% accuracy jungling model |
| 4 | Test in simulation, evaluate | Metrics on performance |
| 5 | (Optional) Build CV pipeline if needed | Prototype camp detector |
| 6-8 | (Optional) Record + process 200 games | High-quality dataset |

---

## My Strong Recommendation

### Start with: **Riot API + Inference Heuristics**

**Why:**
1. **Fast to implement** - Working in 1-2 days
2. **Get actual data** - See shape of problem immediately
3. **Train baseline model** - Proof of concept
4. **Measure gap** - Is it good enough? Or do we need CV?
5. **Decide intelligently** - Only invest in CV if API is insufficient

**Then:**
- If API + inference gives 70%+ accuracy → Ship it!
- If not → Build CV pipeline for 200-500 games

### Code Strategy
```python
# Week 1: API data collector
def collect_matches(num_matches=1000):
    # Query high-elo matches
    # Download timelines
    # Save to database

# Week 2: Camp inference
def infer_camps_from_timeline(timeline):
    # Position tracking
    # Jungle minion delta
    # Nearest camp logic
    # Return: [(state, action, next_state), ...]

# Week 3: Train model
def train_behavior_cloning(training_data):
    # PyTorch neural network
    # Input: game state
    # Output: action probabilities
    # Loss: cross-entropy with expert actions

# Week 4: Evaluate
def evaluate_in_simulation(model):
    # Run in your existing jungle sim
    # Compare to heuristic baseline
    # Measure: gold/min, XP/min, gank success
```

---

## Next Steps - Your Choice!

**Option A: Fast Start (My Recommendation)**
1. I'll help you get Riot API key
2. Write data collector script
3. Download 100 sample timelines today
4. Analyze actual data shape
5. Build inference heuristics
6. **You'll have training data by tomorrow**

**Option B: CV from Start (Higher Risk, Higher Reward)**
1. Set up screen recording automation
2. Build minimap CV detector
3. Record 10 test games
4. Validate detection accuracy
5. Scale to 1000 games
6. **You'll have perfect data in ~2 weeks**

**Option C: Both in Parallel**
1. I work on API collector
2. You experiment with recording setup
3. Compare data quality
4. Choose best approach
5. **Hedge your bets**

---

## What do you want to do?

Vote for:
- **A** - API + inference (fast, pragmatic)
- **B** - CV from start (slow, perfect data)
- **C** - Both in parallel (most work, most safety)
- **D** - Something else?
