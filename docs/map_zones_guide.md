# Map Zone System Guide

## Overview

The map zone system discretizes Summoner's Rift into macro-regions for jungle decision-making. This abstraction reduces spatial complexity while preserving the core jungle role identity (clears, ganks, objectives).

Based on `annotated_map.png` visual reference.

## Zone Categories

### Lanes (3 zones)
- **Top Lane**: High Y region along top edge
- **Mid Lane**: Diagonal corridor from bot-left to top-right
- **Bot Lane**: Low Y region along bottom edge

### Gank Regions (3 zones)
- **Top Gank**: Between top jungle and top lane
- **Mid Gank**: Between mid jungle and mid lane
- **Bot Gank**: Between bot jungle and bot lane

### River (3 zones)
- **Top River**: Upper river area
- **Mid River**: Center river area
- **Bot River**: Lower river area

### Objectives (2 zones)
- **Baron Pit**: Baron Nashor area
- **Dragon Pit**: Dragon area

### Jungle Quadrants (4 zones)
- **Blue Top Jungle**: Blue team's top-side jungle
- **Blue Bot Jungle**: Blue team's bot-side jungle
- **Red Top Jungle**: Red team's top-side jungle
- **Red Bot Jungle**: Red team's bot-side jungle

### Bases (2 zones)
- **Blue Base**: Blue team fountain/base
- **Red Base**: Red team fountain/base

## Usage

### Basic Zone Detection

```python
from map_zones import get_zone, zone_to_string

# Get zone for a position
x, y = 5000, 10500  # Baron pit area
zone = get_zone(x, y)
print(zone_to_string(zone))  # "Baron Pit"
```

### Position Features for ML

```python
from position_features import position_to_zone_features

# Get 5-dimensional categorical features
features = position_to_zone_features(x, y)
# Returns: [is_gank, is_objective, is_jungle, is_lane, is_river]
```

### Comprehensive Context

```python
from position_features import get_positional_context

context = get_positional_context(x, y)
# Returns dict with:
# - zone: Current zone enum
# - zone_type: Type category (gank/objective/jungle/lane/river/base)
# - nearest_gank: Nearest gank zone
# - distance_to_baron: Distance to Baron
# - distance_to_dragon: Distance to Dragon
```

### Integration with Training Data

To add positional features to your jungle training data:

```python
from jungle_data_processor import JungleDataProcessor
from position_features import position_to_zone_features
import numpy as np

# Process matches as usual
processor = JungleDataProcessor(jungle_data, powerspike_data)
states = processor.process_all_matches()

# Add positional features
enhanced_observations = []
for state in states:
    if not state.next_action:
        continue

    # Original features (10 dims)
    base_obs = [
        state.jungler_level / 18.0,
        state.jungler_gold / 20000.0,
        len(state.jungler_items) / 6.0,
        state.jungler_spike_score,
        state.ally_top_spike,
        state.ally_mid_spike,
        state.ally_bot_spike,
        state.enemy_top_spike,
        state.enemy_mid_spike,
        state.enemy_bot_spike,
    ]

    # Add positional features (5 dims)
    x, y = state.jungler_position
    pos_features = position_to_zone_features(x, y)

    # Combine: 15-dimensional observation
    enhanced_obs = base_obs + pos_features.tolist()
    enhanced_observations.append(enhanced_obs)

observations = np.array(enhanced_observations, dtype=np.float32)
```

## Action → Zone Mapping

Actions in the jungle model map to specific zones:

| Action | Zone Type | Target Zone |
|--------|-----------|-------------|
| `blue_buff` | Camp | Blue Top Jungle |
| `red_buff` | Camp | Blue/Red Bot Jungle |
| `gromp` | Camp | Blue Top Jungle |
| `wolves` | Camp | Blue Top Jungle |
| `raptors` | Camp | Blue Bot Jungle |
| `krugs` | Camp | Blue/Red Bot Jungle |
| `gank_top` | Gank | Top Gank |
| `gank_mid` | Gank | Mid Gank |
| `gank_bot` | Gank | Bot Gank |

## Design Philosophy

This zone system is designed to:

1. **Reduce complexity**: Instead of raw (x, y) coordinates, use meaningful regions
2. **Preserve semantics**: Zones map to actual jungle concepts (camps, ganks, objectives)
3. **Support decision-making**: Gank zones represent macro intent, not micro positioning
4. **Enable RL**: Discrete zones work better for action spaces than continuous coordinates

## Files

- `src/map_zones.py`: Core zone definitions and detection
- `src/position_features.py`: Feature extraction for ML models
- `annotated_map.png`: Visual reference for zone boundaries
- `docs/map_zones_guide.md`: This guide

## Future Enhancements

Potential additions:
- Camp-specific zones (each jungle camp as its own zone)
- Invade zones (enemy jungle)
- Lane push states (per-lane minion wave positions)
- Vision zones (common ward locations)
- Pathing hints (optimal routes between zones)
