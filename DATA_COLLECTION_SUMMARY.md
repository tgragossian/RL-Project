# Data Collection Strategy - Updated

**Previous Approach:** Riot API-based collection (deprecated due to API limitations)

**New Approach:** Screenshot-based automated replay collection with computer vision extraction

**Documentation:** See [docs/screenshot_pipeline.md](docs/screenshot_pipeline.md) for complete details

---

## Why We Changed Approaches

### API Limitations (Previous Approach)
- ❌ Only 60-second temporal resolution
- ❌ No individual camp clear data
- ❌ No fog of war information
- ❌ Limited epic monster tracking
- ❌ No cooldown data
- ❌ No health/mana states

### Screenshot Benefits (New Approach)
- ✅ 5-second intervals (12x better resolution)
- ✅ Complete game state (positions, items, gold, HP/mana, cooldowns)
- ✅ Fog of war tracking (both visible and true positions)
- ✅ All epic monsters (Dragon, Baron, Herald, Atakhan, Void Grubs)
- ✅ ~1.8M training examples from 500 games

---

## New Data Collection Pipeline

### Collection Workflow
1. **Automated replay navigation** (pyautogui)
   - Jump backwards in 5-second intervals from game end
   - ~12 minutes to collect one 30-minute game

2. **Two screenshots per timestamp**
   - Image 1: Fog ON + Tab (visible enemies only)
   - Image 2: Fog OFF + Tab + X (true positions + gold)

3. **Computer vision extraction**
   - Champion positions (minimap blob detection)
   - Items, stats, gold (template matching + OCR)
   - Health/mana bars, cooldowns
   - Epic monster status

4. **Partition-based processing**
   - Collect 100 games → Process → Save CSV → Delete images
   - Repeat for K partitions
   - Merge all CSVs for final training

### Storage Efficiency
- **During partition**: ~180 GB (temporary)
- **After processing**: ~72 MB CSV per 100 games
- **Final dataset**: ~360 MB for 500 games
- **No cloud storage needed**

---

## Old API-Based Collection (Archived)

### ✅ Improvements Implemented:

1. **Jungle Player Verification (5-match check)**
   - Checks first 5 matches to verify player is actually a jungler
   - Saves API calls by skipping non-jungle players
   - Only scans remaining matches if jungle role detected

2. **API Key Rotation Verified**
   - Two API keys properly rotating:
     - `RGAPI-81751001-babd-49a5-ae02-d00305d382db`
     - `RGAPI-7012b0bb-e73a-4043-889d-a7a6e2a03621`
   - Round-robin rotation using `_current_key_index`

3. **Clean Console Output**
   - **Before:**
     ```
     Processing match: NA1_5421699749
       ✗ Player not jungling in this match
     Processing match: NA1_5421639550
       ✗ Player not jungling in this match
     ```
   - **After:**
     ```
     [Player 1] SummonerName
       Checking if jungle player...
       ✓ Jungler detected, scanning matches...
       ✓ KhaZix jungle match
         (35 frames)
         [1/10] collected
     ```

---

## Data Collected

### Per Match:
- **Match metadata:** ID, version, duration
- **Jungler info:** Participant ID, PUUID, champion

### Per Participant (All 10 players):
- **Identity:** Participant ID, champion, team, role, PUUID
- **Match outcome:** Win/loss
- **Combat stats:** Kills, deaths, assists, damage dealt/taken (total, physical, magic, true)
- **Farm stats:** CS, jungle CS
- **Objective stats:** Dragons, barons, turrets, inhibitors
- **Gold/XP:** Total earned, final level, experience
- **Items:** Final build (7 item slots)
- **Vision:** Vision score, wards placed/killed

### Per Frame (Every 60 seconds, All 10 players):
- **Core stats:** Level, XP, position (x, y)
- **Gold:** Current gold, total gold, gold per second
- **Farm:** CS, jungle CS
- **Champion stats (25 fields):**
  - Combat: HP, max HP, HP regen, armor, MR, AD, AS, AP, ability haste
  - Sustain: Lifesteal, omnivamp, physical vamp, spell vamp
  - Penetration: Armor pen (flat/%), magic pen (flat/%)
  - Other: Mana/energy, max mana, mana regen, CC reduction, movement speed
- **Damage stats (12 fields):**
  - Total damage done/taken
  - Damage to champions (total, physical, magic, true)
  - Damage taken (physical, magic, true)
- **CC:** Time enemy spent controlled

### Per Frame Events:
- **ELITE_MONSTER_KILL:** Dragons, barons, heralds (type, killer, position, timestamp)
- **CHAMPION_KILL:** Kills (killer, victim, assists, position, timestamp)
- **BUILDING_KILL:** Turrets, inhibitors (type, killer, team, position)
- **LEVEL_UP:** Level up events (participant, level, timestamp)
- **ITEM_PURCHASED:** Item buys (participant, item ID, timestamp)

---

## Data Structure

```json
{
  "collection_date": "2025-12-18 HH:MM:SS",
  "total_matches": 10,
  "matches": [
    {
      "match_id": "NA1_5434752489",
      "game_version": "15.1.123.4567",
      "game_duration": 1850,
      "jungler_participant_id": 3,
      "jungler_puuid": "...",
      "participants": [
        {
          "participant_id": 1,
          "champion_name": "KSante",
          "team_id": 100,
          "team_position": "TOP",
          "win": false,
          "kills": 1,
          "deaths": 3,
          "assists": 4,
          "total_damage_dealt_to_champions": 11547,
          "total_damage_taken": 28641,
          "total_minions_killed": 232,
          "neutral_minions_killed": 4,
          "baron_kills": 0,
          "dragon_kills": 0,
          "gold_earned": 10980,
          "champ_level": 17,
          "items": [3065, 2502, 0, 6665, 3111, 0, 3364],
          "vision_score": 27
        },
        // ... 9 more participants
      ],
      "frames": [
        {
          "timestamp": 60000,
          "participants": {
            "1": {
              "participant_id": 1,
              "level": 2,
              "xp": 450,
              "position_x": 1234,
              "position_y": 5678,
              "current_gold": 125,
              "total_gold": 625,
              "minions_killed": 8,
              "jungle_minions_killed": 0,
              "health": 650,
              "health_max": 750,
              "armor": 45,
              "magic_resist": 32,
              "attack_damage": 68,
              "attack_speed": 95,
              "movement_speed": 345,
              "total_damage_done": 1250,
              "total_damage_taken": 380
              // ... 30+ more fields per player
            },
            // ... participants 2-10
          },
          "events": [
            {
              "type": "LEVEL_UP",
              "timestamp": 62000,
              "participant_id": 1,
              "level": 2
            },
            {
              "type": "CHAMPION_KILL",
              "timestamp": 65000,
              "killer_id": 3,
              "victim_id": 7,
              "assisting_participant_ids": [1],
              "position_x": 5000,
              "position_y": 9000
            }
          ]
        },
        // ... ~30 more frames (one per minute)
      ]
    }
  ]
}
```

---

## File Size Estimate

**Per match:**
- ~10 participants × 145 fields = 1,450 values
- ~30 frames × 10 participants × 50 fields = 15,000 values
- ~30 frames × ~10 events × 7 fields = 2,100 values
- **Total:** ~18,550 data points per match

**For 10 matches:**
- ~185,500 data points
- **Estimated file size:** ~8-15 MB (JSON with indent)

---

## How to Run

```bash
cd /Users/thomas/Desktop/RL\ Project/RL-Project
python3 scripts/collect_camp_classification_data.py
```

**Expected output:**
```
======================================================================
CAMP CLASSIFICATION DATA COLLECTION
Using 2 API key(s)
======================================================================

======================================================================
COLLECTING CAMP CLASSIFICATION DATA
Platform: NA1
Target: 10 jungle matches
======================================================================

Fetching Challenger leaderboard...
✓ Found 300 Challenger players

[Player 1] SomeJungler
  Checking if jungle player...
  ✓ Jungler detected, scanning matches...
    ✓ KhaZix jungle match
      (35 frames)
      [1/10] collected
    ✓ Graves jungle match
      (28 frames)
      [2/10] collected

[Player 2] SomeLaner
  Checking if jungle player...
  ✗ Not a jungle player, skipping

[Player 3] AnotherJungler
  Checking if jungle player...
  ✓ Jungler detected, scanning matches...
    ✓ Lee Sin jungle match
      (31 frames)
      [3/10] collected
...

✓ Reached target of 10 matches!

======================================================================
✅ COLLECTION COMPLETE
======================================================================
Matches collected: 10
Saved to: data/processed/camp_classification_data.json

Data summary:
  Total frames: 315
  Avg frames per match: 31.5
  Frame interval: 60 seconds
```

---

## API Usage Optimization

### Before (wasteful):
- Check 20 matches per player
- Many players are laners (80% waste)
- ~100 API calls to find 10 jungle matches

### After (efficient):
- Check first 5 matches to verify jungler (5 calls)
- Skip non-junglers entirely
- Only scan remaining 15 matches if jungler detected
- **Saves ~40-60% of API calls**

---

## Notes

### Enemy Jungler Data:
- **No fog of war violations** - all data is what the API provides
- Frame data shows ALL 10 players every frame (API limitation)
- In your ML model, you should only use enemy data that would be "visible" in-game
- Consider adding fog of war simulation later if needed

### Fog of War Simulation (Future):
```python
def is_visible(enemy_position, ally_positions, ward_positions):
    """Check if enemy is visible based on vision range."""
    vision_range = 1200  # Standard vision range

    # Check if in range of any ally
    for ally_pos in ally_positions:
        if distance(enemy_position, ally_pos) < vision_range:
            return True

    # Check if in range of any ward
    for ward_pos in ward_positions:
        if distance(enemy_position, ward_pos) < vision_range:
            return True

    return False
```

This would filter enemy data to only what's actually visible, making it more realistic.
