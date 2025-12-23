# Riot API Fields Reference - Complete Column List

**Generated:** 2025-12-18
**API Version:** Match-v5
**Purpose:** Reference for reducing columns in jungle_data_processor.py

---

## Match Info API - Participant Fields (145 total fields)

### Currently Used in jungle_data_processor.py ✅

| Field Name | Type | Current Usage | Line |
|------------|------|---------------|------|
| `participantId` | int | Identify jungler in timeline | 219 |
| `level` (from timeline) | int | `jungler_level` state feature | 258 |
| `goldEarned` | int | `jungler_gold` state feature | 259 |
| `item0-6` | int | `jungler_items` (converted to names) | 260 |
| `championName` | str | For powerspike calculation | - |
| `teamPosition` | str | Filter for "JUNGLE" role | - |
| `puuid` | str | Match player to summoner | - |

### Available But NOT Used ⚠️

#### Combat Stats
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `kills` | int | Total kills | Win condition / performance metric |
| `deaths` | int | Total deaths | Survivability metric |
| `assists` | int | Total assists | Teamplay metric |
| `champLevel` | int | Final level | End-game state |
| `champExperience` | int | Total XP earned | Progression tracking |
| `doubleKills`, `tripleKills`, etc. | int | Multikills | Performance metric |

#### Damage Stats
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `totalDamageDealtToChampions` | int | Damage to champions | Fight contribution |
| `physicalDamageDealtToChampions` | int | Physical damage | Damage type breakdown |
| `magicDamageDealtToChampions` | int | Magic damage | Damage type breakdown |
| `trueDamageDealtToChampions` | int | True damage | Damage type breakdown |
| `totalDamageTaken` | int | Damage taken | Tankiness metric |
| `damageSelfMitigated` | int | Damage mitigated | Effective HP |
| `damageDealtToObjectives` | int | Objective damage | Objective focus |
| `damageDealtToTurrets` | int | Turret damage | Splitpush metric |

#### Vision Stats
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `visionScore` | int | Total vision score | Map control metric |
| `wardsPlaced` | int | Wards placed | Vision control |
| `wardsKilled` | int | Enemy wards killed | Vision denial |
| `visionWardsBoughtInGame` | int | Control wards bought | Vision investment |
| `detectorWardsPlaced` | int | Control wards placed | Deep vision |

#### Objective Stats
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `baronKills` | int | Barons killed | Epic objective participation |
| `dragonKills` | int | Dragons killed | Epic objective participation |
| `turretKills` | int | Turrets destroyed | Tower pressure |
| `inhibitorKills` | int | Inhibitors destroyed | Game-ending push |
| `objectivesStolen` | int | Objectives stolen | Clutch plays |
| `objectivesStolenAssists` | int | Assisted steals | Team coordination |

#### Farm Stats
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `totalMinionsKilled` | int | Total CS | Farm efficiency |
| `neutralMinionsKilled` | int | Jungle monsters | Jungle CS (already tracked in timeline) |
| `totalAllyJungleMinionsKilled` | int | Own jungle CS | Territory control |
| `totalEnemyJungleMinionsKilled` | int | Enemy jungle CS | Counter-jungling metric |

#### Gold Stats
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `goldSpent` | int | Gold spent | Economy efficiency |
| `consumablesPurchased` | int | Potions/wards bought | Resource management |
| `itemsPurchased` | int | Total items bought | Recall frequency |

#### CC/Utility Stats
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `timeCCingOthers` | int | Time CCing enemies | Crowd control contribution |
| `totalTimeCCDealt` | int | Total CC duration | CC effectiveness |
| `timeEnemySpentControlled` | int | Total CC taken | Getting caught |
| `totalTimeSpentDead` | int | Death time | Availability penalty |

#### Healing/Shielding
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `totalHeal` | int | Self healing | Sustain metric |
| `totalHealsOnTeammates` | int | Healing to team | Support metric |
| `totalDamageShieldedOnTeammates` | int | Shields to team | Support metric |

#### Communication (Pings)
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `allInPings` | int | All-in pings | Aggression signals |
| `assistMePings` | int | Assist pings | Help requests |
| `enemyMissingPings` | int | Missing pings | Communication |
| `dangerPings` | int | Danger pings | Warning signals |
| `onMyWayPings` | int | OMW pings | Coordination |
| `retreatPings` | int | Retreat pings | Defensive calls |

#### Spell Usage
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `spell1Casts`, `spell2Casts`, etc. | int | Ability casts | Spell efficiency |
| `summoner1Casts`, `summoner2Casts` | int | Summoner spell casts | Flash/Smite usage |
| `summoner1Id`, `summoner2Id` | int | Summoner spell IDs | Verify Smite presence |

#### Game State
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `win` | bool | Did player win? | **Critical for RL rewards** |
| `timePlayed` | int | Match duration | Game length |
| `teamId` | int | Team (100 or 200) | Side identification |
| `individualPosition` | str | Lane/role | Role verification |

---

## Timeline API - Participant Frame Fields (12 fields per frame)

### Currently Used in jungle_data_processor.py ✅

| Field | Type | Current Usage |
|-------|------|---------------|
| `level` | int | `jungler_level` |
| `totalGold` | int | `jungler_gold` (or goldEarned from match info) |
| `position.x`, `position.y` | int | `jungler_position` |

### Available But NOT Used ⚠️

| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `currentGold` | int | Unspent gold | Recall timing decision |
| `goldPerSecond` | int | Current GPM | Income rate |
| `minionsKilled` | int | Lane CS | Farm tracking |
| `jungleMinionsKilled` | int | Jungle CS | **Used in camp detection!** |
| `xp` | int | Current XP | Level-up prediction |
| `timeEnemySpentControlled` | int | CC taken | Combat state |

#### championStats (25 sub-fields)
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `health` | int | Current HP | **Critical for fight decisions** |
| `healthMax` | int | Max HP | HP percentage calculation |
| `healthRegen` | int | HP regen | Sustain metric |
| `armor` | int | Armor stat | **Tankiness for fights** |
| `magicResist` | int | MR stat | **Tankiness for fights** |
| `attackDamage` | int | AD stat | Clear speed |
| `attackSpeed` | int | AS stat | Clear speed |
| `abilityPower` | int | AP stat | AP jungler scaling |
| `abilityHaste` | int | Ability haste | CDR metric |
| `movementSpeed` | int | Movement speed | **Pathing speed** |
| `lifesteal` | int | Lifesteal | Sustain |
| `omnivamp` | int | Omnivamp | Sustain |
| `armorPen`, `magicPen` | int | Penetration stats | Damage optimization |

#### damageStats (12 sub-fields)
| Field | Type | Description | Potential Use |
|-------|------|-------------|---------------|
| `totalDamageDone` | int | Cumulative damage | Combat activity |
| `totalDamageDoneToChampions` | int | Champion damage | Fight participation |
| `totalDamageTaken` | int | Damage taken | **Being invaded detection** |
| `physicalDamageDone` | int | Physical damage | Damage breakdown |
| `magicDamageDone` | int | Magic damage | Damage breakdown |
| `trueDamageDone` | int | True damage (Smite!) | Smite usage tracking |

---

## Timeline API - Event Types

### Currently Used ✅

| Event Type | Fields | Current Usage |
|------------|--------|---------------|
| (None directly used - inferred from CS changes) | - | - |

### Available Event Types ⚠️

| Event Type | Key Fields | Description | Potential Use |
|------------|------------|-------------|---------------|
| `ELITE_MONSTER_KILL` | `monsterType`, `killerId`, `timestamp`, `position` | Dragon/Baron/Herald kills | **Objective timing, rewards** |
| `CHAMPION_KILL` | `killerId`, `victimId`, `assistingParticipantIds`, `position` | Champion kills | **Gank detection (current)** |
| `BUILDING_KILL` | `buildingType`, `teamId`, `position` | Tower/Inhibitor kills | Objective priority |
| `WARD_PLACED` | `wardType`, `creatorId`, `timestamp` | Vision control | Vision patterns |
| `WARD_KILL` | `killerId`, `timestamp` | Ward denial | Vision denial |
| `LEVEL_UP` | `participantId`, `level`, `timestamp` | Level ups | **Powerspike timing** |
| `SKILL_LEVEL_UP` | `participantId`, `skillSlot`, `timestamp` | Skill points | Ability powerspikes |
| `ITEM_PURCHASED` | `participantId`, `itemId`, `timestamp` | Item buys | **Item powerspikes** |
| `ITEM_DESTROYED` | `participantId`, `itemId`, `timestamp` | Item sells | Build changes |
| `ITEM_UNDO` | `participantId`, `itemId`, `timestamp` | Item refunds | Mistakes? |
| `TURRET_PLATE_DESTROYED` | `teamId`, `position`, `timestamp` | Plate gold | Lane pressure |
| `PAUSE_END` | `timestamp` | Game paused | Anomaly detection |

---

## Recommendations for jungle_data_processor.py

### ✅ Keep These (Essential):
1. **Jungler State:**
   - `level` - Power scaling
   - `totalGold` or `goldEarned` - Economic state
   - `position` - Location for pathing
   - `items` - Powerspike calculation

2. **Lane Powerspikes:**
   - Already using `overall_spike_score` (good!)

3. **Actions:**
   - Camp clears (from CS changes)
   - Ganks (from kill events)

### 🔥 Add These (High Value):

#### From Participant Frames (Timeline):
1. **`health` / `healthMax`** - HP% is critical for:
   - Deciding to fight/invade
   - Recall timing
   - Risk assessment

2. **`currentGold`** - Unspent gold indicates:
   - Ready to recall
   - Holding for item spike

3. **`jungleMinionsKilled`** - Already using for camp detection (keep!)

4. **`movementSpeed`** - For accurate travel time calculations

5. **`damageStats.totalDamageTaken`** - Detect being invaded:
   - Sudden damage = combat = possible invade

#### From Match Info (End-game):
1. **`win`** - **CRITICAL for RL training rewards!**
   - Positive reward for wins
   - Negative for losses
   - Learn winning patterns

2. **`totalEnemyJungleMinionsKilled`** - Counter-jungle metric
   - How much you invaded

3. **`visionScore`** - Vision control quality

### ❌ Can Remove (Low Value):

1. **Ping stats** - Not useful for decision making
2. **Spell cast counts** - Too granular
3. **Multikill stats** - Outcome, not input
4. **PlayerScore0-11** - Usually 0 (legacy fields)
5. **Tournament/augment fields** - Not applicable to ranked
6. **Placement/subteam** - Arena mode only

---

## Minimal High-Value Feature Set

If reducing to bare minimum, use these **25 features**:

### Jungler (7 features):
1. `level` - Power level
2. `gold` - Economy
3. `health_percent` - HP state
4. `current_gold` - Unspent gold
5. `movement_speed` - Mobility
6. `num_items` - Item count
7. `spike_score` - Overall power

### Lanes (12 features):
8-13. `ally_top/mid/bot_spike` - Ally power (3)
14-19. `enemy_top/mid/bot_spike` - Enemy power (3)
20-25. `ally_top/mid/bot_hp_percent` - Ally HP (3)
26-31. `enemy_top/mid/bot_hp_percent` - Enemy HP (3)

### Context (6 features):
32. `game_time` - Timing
33-34. `position_x`, `position_y` - Location
35. `last_camp` (one-hot) - Memory
36. `time_since_last_camp` - Pacing
37. `damage_taken_recently` - Combat detection

### Rewards (from match info):
- `win` - For RL training outcome

---

## Current vs. Recommended State Vector

### Current (jungle_data_processor.py):
```python
obs = [
    state.jungler_level / 18.0,           # 1 feature
    state.jungler_gold / 20000.0,         # 1 feature
    len(state.jungler_items) / 6.0,       # 1 feature
    state.jungler_spike_score,            # 1 feature
    state.ally_top_spike,                 # 1 feature
    state.ally_mid_spike,                 # 1 feature
    state.ally_bot_spike,                 # 1 feature
    state.enemy_top_spike,                # 1 feature
    state.enemy_mid_spike,                # 1 feature
    state.enemy_bot_spike,                # 1 feature
]
# Total: 10 features
```

### Recommended (enhanced):
```python
obs = [
    # Jungler core (5)
    state.jungler_level / 18.0,
    state.jungler_gold / 20000.0,
    state.jungler_health_percent,         # NEW!
    state.jungler_current_gold / 5000.0,  # NEW!
    len(state.jungler_items) / 6.0,

    # Jungler powerspike (1)
    state.jungler_spike_score,

    # Lane powerspikes (6)
    state.ally_top_spike,
    state.ally_mid_spike,
    state.ally_bot_spike,
    state.enemy_top_spike,
    state.enemy_mid_spike,
    state.enemy_bot_spike,

    # Lane HP states (6) - NEW!
    state.ally_top_hp_percent,
    state.ally_mid_hp_percent,
    state.ally_bot_hp_percent,
    state.enemy_top_hp_percent,
    state.enemy_mid_hp_percent,
    state.enemy_bot_hp_percent,

    # Context (3) - NEW!
    state.game_time / 1800.0,             # 0-30 min
    state.damage_taken_recently / 1000.0, # Combat detection
    state.movement_speed / 500.0,         # Mobility
]
# Total: 21 features (up from 10)
```

---

## API Field Access Pattern

```python
# From Match Info (match_data['info']['participants'][i])
win = participant['win']
gold_earned = participant['goldEarned']
items = [participant[f'item{i}'] for i in range(7)]
enemy_jungle_cs = participant['totalEnemyJungleMinionsKilled']

# From Timeline Frame (timeline_data['info']['frames'][t]['participantFrames'][str(pid)])
level = pf['level']
position = (pf['position']['x'], pf['position']['y'])
health = pf['championStats']['health']
health_max = pf['championStats']['healthMax']
hp_percent = health / health_max
current_gold = pf['currentGold']
movement_speed = pf['championStats']['movementSpeed']
damage_taken = pf['damageStats']['totalDamageTaken']
```

---

**Summary:** You have access to 145 match-level fields and 49 frame-level fields (12 top-level + 25 championStats + 12 damageStats). Currently using only ~7 of them. Recommended to add HP%, current_gold, and win for better decision-making.
