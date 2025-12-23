"""
Champion powerspike system for gank/objective timing.

Parses the powerspike CSV and generates "spike scores" for each champion
based on their current level and items. Higher spike scores indicate the
champion is at a power advantage (good for ganking or fighting).

These scores are used as state features to help the model learn when to:
- Gank lanes (when allies have spike advantage)
- Avoid ganks (when enemies have spike advantage)
- Contest objectives (when team has overall spike advantage)
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class ChampionSpikes:
    """Powerspike information for a champion."""
    champion: str
    level_spikes: List[int]  # Levels where champion spikes (e.g., [3, 6, 9])
    item_spikes: List[str]   # Key items that spike (e.g., ["Eclipse", "Sterak's Gage"])


class PowerspikeSystem:
    """
    System for calculating champion powerspike scores.

    The spike score is a 0-1 value indicating how close a champion is to
    their optimal power state based on level and items.
    """

    def __init__(self, csv_path: Path):
        """Load and parse powerspike data from CSV."""
        self.champion_spikes: Dict[str, ChampionSpikes] = {}
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: Path):
        """Parse the powerspike CSV file."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                champion = row.get('Champion', '').strip()
                if not champion or champion == '':
                    continue

                # Parse level spikes (e.g., "4 / 6 / 9" -> [4, 6, 9])
                level_str = row.get('Level Spikes', '')
                level_spikes = self._parse_levels(level_str)

                # Parse item spikes (e.g., "Eclipse / Sterak's Gage" -> ["Eclipse", "Sterak's Gage"])
                item_str = row.get('Notable Item Power Spikes', '')
                item_spikes = self._parse_items(item_str)

                self.champion_spikes[champion.lower()] = ChampionSpikes(
                    champion=champion,
                    level_spikes=level_spikes,
                    item_spikes=item_spikes
                )

    def _parse_levels(self, level_str: str) -> List[int]:
        """Extract level numbers from string like '4 / 6 / 9' or 'lvl 6'."""
        if not level_str:
            return []

        # Extract all numbers
        numbers = re.findall(r'\d+', level_str)
        return [int(n) for n in numbers if int(n) <= 18]

    def _parse_items(self, item_str: str) -> List[str]:
        """Extract item names from string like 'Eclipse / Sterak's Gage'."""
        if not item_str:
            return []

        # Split by / and clean up
        items = [item.strip() for item in item_str.split('/')]

        # Remove common prefixes/suffixes
        cleaned_items = []
        for item in items:
            # Remove things like "Tier 2" or "(when finished)"
            item = re.sub(r'Tier \d+\s*', '', item)
            item = re.sub(r'\(.*?\)', '', item)
            item = re.sub(r'⚠️', '', item)
            item = item.strip()
            if item and len(item) > 2:  # Ignore very short strings
                cleaned_items.append(item.lower())

        return cleaned_items

    def calculate_level_spike_score(self, champion: str, current_level: int) -> float:
        """
        Calculate how close the champion is to their next level spike.

        Returns:
            0.0-1.0 where 1.0 = at a major level spike, 0.0 = far from spikes
        """
        champion = champion.lower()

        if champion not in self.champion_spikes:
            # Default spike score for unknown champions
            # Assume common spikes at 3, 6, 11, 16
            level_spikes = [3, 6, 11, 16]
        else:
            level_spikes = self.champion_spikes[champion].level_spikes

        if not level_spikes:
            # If no specific spikes, use ultimate levels
            level_spikes = [6, 11, 16]

        # Check if at a spike level
        if current_level in level_spikes:
            return 1.0

        # Find closest spike levels
        lower_spike = max([s for s in level_spikes if s < current_level], default=0)
        upper_spike = min([s for s in level_spikes if s > current_level], default=18)

        if upper_spike == 18 and current_level > max(level_spikes, default=0):
            # Past all spikes, gradual decay
            return 0.5

        # Linear interpolation between spikes
        if upper_spike == lower_spike:
            return 0.3  # Before first spike

        progress = (current_level - lower_spike) / (upper_spike - lower_spike)

        # Score peaks at spike levels, decreases linearly after
        # 1.0 at spike, decays to 0.3 before next spike
        score = 1.0 - (0.7 * progress)

        return max(0.3, min(1.0, score))

    def calculate_item_spike_score(self, champion: str, items: List[str]) -> float:
        """
        Calculate item spike score based on completed key items.

        Args:
            champion: Champion name
            items: List of item names the champion currently has

        Returns:
            0.0-1.0 based on how many key items are completed
        """
        champion = champion.lower()

        if champion not in self.champion_spikes:
            # Unknown champion, use generic item count
            # More items = stronger
            return min(1.0, len(items) * 0.2)

        spike_items = self.champion_spikes[champion].item_spikes

        if not spike_items:
            # No specific spike items, use generic
            return min(1.0, len(items) * 0.2)

        # Check how many spike items are completed
        items_lower = [item.lower() for item in items]

        completed_spikes = 0
        for spike_item in spike_items:
            # Check if any owned item contains the spike item name
            for owned_item in items_lower:
                if spike_item in owned_item or owned_item in spike_item:
                    completed_spikes += 1
                    break

        # Score based on proportion of spike items completed
        max_spikes = min(len(spike_items), 3)  # Cap at 3 major items
        score = completed_spikes / max_spikes if max_spikes > 0 else 0.5

        return min(1.0, score)

    def calculate_overall_spike_score(self, champion: str, level: int, items: List[str]) -> float:
        """
        Calculate overall powerspike score combining level and items.

        Returns:
            0.0-1.0 overall power state score
        """
        level_score = self.calculate_level_spike_score(champion, level)
        item_score = self.calculate_item_spike_score(champion, items)

        # Weighted combination: level spikes matter more early, items more late
        level_weight = max(0.3, 1.0 - (level / 18) * 0.5)  # 1.0 early -> 0.5 late
        item_weight = 1.0 - level_weight

        overall = (level_score * level_weight) + (item_score * item_weight)

        return overall

    def get_champion_names(self) -> List[str]:
        """Get list of all known champion names."""
        return list(self.champion_spikes.keys())


def create_gank_opportunity_score(
    ally_spike: float,
    enemy_spike: float,
    ally_hp_percent: float,
    enemy_hp_percent: float
) -> float:
    """
    Calculate gank opportunity score for a lane.

    Args:
        ally_spike: Ally laner's powerspike score (0-1)
        enemy_spike: Enemy laner's powerspike score (0-1)
        ally_hp_percent: Ally HP percentage (0-1)
        enemy_hp_percent: Enemy HP percentage (0-1)

    Returns:
        0-1 score indicating gank opportunity quality
    """
    # Spike advantage (we're stronger = better gank)
    spike_advantage = ally_spike - enemy_spike

    # HP advantage (enemy low HP = better gank)
    hp_advantage = ally_hp_percent - enemy_hp_percent

    # Combine factors
    # Positive spike advantage and enemy low HP = high score
    score = 0.5  # Base score
    score += spike_advantage * 0.3  # +/- 0.3 from spike diff
    score -= hp_advantage * 0.2     # Enemy lower HP = bonus

    return max(0.0, min(1.0, score))


if __name__ == "__main__":
    # Test the powerspike system
    csv_path = Path(__file__).parent.parent / "data" / "raw" / "Champ Powerspikes _ Updated Patch 14.21 - Powerspikes.csv"

    if not csv_path.exists():
        print(f"Powerspike CSV not found at: {csv_path}")
        exit(1)

    system = PowerspikeSystem(csv_path)

    print(f"Loaded powerspike data for {len(system.champion_spikes)} champions")
    print("\nExample champion data:")

    # Test a few champions
    test_cases = [
        ("Lee Sin", 3, []),
        ("Lee Sin", 6, ["Eclipse"]),
        ("Ahri", 6, ["Malignance"]),
        ("Aatrox", 9, ["Eclipse", "Black Cleaver"]),
    ]

    for champ, level, items in test_cases:
        level_score = system.calculate_level_spike_score(champ, level)
        item_score = system.calculate_item_spike_score(champ, items)
        overall = system.calculate_overall_spike_score(champ, level, items)

        print(f"\n{champ} (Lvl {level}, Items: {items if items else 'None'})")
        print(f"  Level Score: {level_score:.2f}")
        print(f"  Item Score:  {item_score:.2f}")
        print(f"  Overall:     {overall:.2f}")

    # Test gank opportunity
    print("\n" + "="*60)
    print("Gank Opportunity Examples:")
    print("="*60)

    scenarios = [
        ("Ahead laner vs behind enemy", 0.8, 0.4, 1.0, 0.5),
        ("Even laners, enemy low HP", 0.6, 0.6, 0.8, 0.3),
        ("Behind laner vs ahead enemy", 0.4, 0.8, 0.7, 1.0),
    ]

    for desc, ally_spike, enemy_spike, ally_hp, enemy_hp in scenarios:
        score = create_gank_opportunity_score(ally_spike, enemy_spike, ally_hp, enemy_hp)
        print(f"\n{desc}")
        print(f"  Ally Spike: {ally_spike:.2f}, Enemy Spike: {enemy_spike:.2f}")
        print(f"  Ally HP: {ally_hp:.2f}, Enemy HP: {enemy_hp:.2f}")
        print(f"  → Gank Score: {score:.2f}")
