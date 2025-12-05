"""
Dynamic monster scaling system for jungle camps.

This module provides realistic stat scaling for all jungle camps and epic objectives
based on the average level of champions in the game. Stats interpolate linearly from
base values (level 1) to maximum values (level 18).

Includes:
    - 6 regular jungle camps (Blue, Red, Gromp, Wolves, Raptors, Krugs, Scuttle)
    - 4 epic objectives (Dragon, Herald, Baron, Void Grubs)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

MAX_CHAMP_LEVEL = 18  # LoL levels 1–18


@dataclass
class MonsterScaling:
    """
    Defines how a monster's stats scale from base (level 1) to max (level 18).

    Attributes:
        hp: (base, max) health points
        armor: (base, max) armor value
        mr: (base, max) magic resistance
        attack_damage: (base, max) attack damage
        attack_speed: Attacks per second (flat, doesn't scale)
        gold: (base, max) gold reward for clearing
        xp: (base, max) experience reward for clearing
        notes: Additional information about the camp
    """
    hp: Tuple[float, float]
    armor: Tuple[float, float]
    mr: Tuple[float, float]
    attack_damage: Tuple[float, float]
    attack_speed: float
    gold: Tuple[float, float]
    xp: Tuple[float, float]
    notes: str = ""


MONSTER_SCALING: Dict[str, MonsterScaling] = {
    # ---------- Normal jungle camps ---------- #

    "Blue Sentinel": MonsterScaling(
        hp=(2300.0, 6210.0),
        armor=(42.0, 42.0),
        mr=(42.0, 42.0),
        attack_damage=(66.0, 198.0),
        attack_speed=0.493,
        gold=(90.0, 90.0),
        xp=(95.0, 142.5),
        notes="large_monster; generic blue buff.",
    ),
    "Murk Wolves (Greater)": MonsterScaling(
        hp=(1600.0, 3760.0),
        armor=(42.0, 42.0),
        mr=(42.0, 42.0),
        attack_damage=(30.0, 90.0),
        attack_speed=0.625,
        gold=(55.0, 55.0),
        xp=(50.0, 75.0),
        notes="large_monster; wolves.",
    ),
    "Gromp": MonsterScaling(
        hp=(2050.0, 4817.5),
        armor=(42.0, 42.0),
        mr=(42.0, 42.0),
        attack_damage=(70.0, 210.0),
        attack_speed=0.425,
        gold=(80.0, 80.0),
        xp=(120.0, 180.0),
        notes="large_monster; gromp.",
    ),
    "Crimson Raptor": MonsterScaling(
        hp=(1200.0, 2820.0),
        armor=(42.0, 42.0),
        mr=(42.0, 42.0),
        attack_damage=(17.0, 51.0),
        attack_speed=0.667,
        gold=(35.0, 35.0),
        xp=(20.0, 30.0),
        notes="large_monster; raptor.",
    ),
    "Ancient Krug": MonsterScaling(
        hp=(1350.0, 3172.5),
        armor=(42.0, 42.0),
        mr=(42.0, 42.0),
        attack_damage=(57.0, 171.0),
        attack_speed=0.613,
        gold=(15.0, 15.0),  # camp total gold/xp is higher; this is per unit
        xp=(15.0, 22.5),
        notes="large_monster; krugs; sim ignores splitting details.",
    ),
    "Red Brambleback": MonsterScaling(
        hp=(2300.0, 6210.0),
        armor=(42.0, 42.0),
        mr=(42.0, 42.0),
        attack_damage=(66.0, 198.0),
        attack_speed=0.493,
        gold=(90.0, 90.0),
        xp=(95.0, 142.5),
        notes="large_monster; generic red buff.",
    ),
    "Rift Scuttler": MonsterScaling(
        hp=(1550.0, 3425.5),
        armor=(42.0, 42.0),
        mr=(42.0, 42.0),
        attack_damage=(35.0, 35.0),  # mostly flat in sim
        attack_speed=0.638,
        gold=(55.0, 121.0),
        xp=(100.0, 150.0),
        notes="large_monster; scuttle; sim ignores shield/flee mechanics.",
    ),

    # ---------- Epic camps / objectives ---------- #

    # Generic elemental dragon (you can split by type later if you want)
    "Dragon": MonsterScaling(
        hp=(3500.0, 9000.0),
        armor=(21.0, 60.0),
        mr=(30.0, 60.0),
        attack_damage=(80.0, 180.0),
        attack_speed=0.75,
        gold=(125.0, 150.0),
        xp=(180.0, 300.0),
        notes="epic_monster; generic elemental dragon approximation.",
    ),

    # Void Grubs – early epic objective that can spawn multiple times.
    # Numbers kept a bit lower than dragon; they're an early fight.
    "Void Grub": MonsterScaling(
        hp=(2200.0, 4500.0),
        armor=(30.0, 45.0),
        mr=(30.0, 45.0),
        attack_damage=(45.0, 90.0),
        attack_speed=0.7,
        gold=(35.0, 50.0),   # per grub; total camp reward is higher
        xp=(80.0, 120.0),
        notes="epic_monster; early game objective; sim treats a single 'grub' unit.",
    ),

    "Rift Herald": MonsterScaling(
        hp=(7500.0, 14000.0),
        armor=(40.0, 70.0),
        mr=(40.0, 70.0),
        attack_damage=(90.0, 200.0),
        attack_speed=0.8,
        gold=(100.0, 150.0),
        xp=(250.0, 350.0),
        notes="epic_monster; sim ignores eye/back mechanics.",
    ),

    "Baron Nashor": MonsterScaling(
        hp=(11800.0, 19700.0),
        armor=(120.0, 120.0),
        mr=(70.0, 70.0),
        attack_damage=(150.0, 520.0),
        attack_speed=0.625,
        gold=(300.0, 300.0),
        xp=(600.0, 800.0),
        notes="epic_monster; late-game objective; stats approximated.",
    ),
}


def monster_level_from_champs(champ_levels: List[int]) -> int:
    """
    Calculate monster level based on average champion level in the game.

    In League of Legends, jungle camps scale with the average level of all
    champions in the game to maintain relevance throughout the match.

    Args:
        champ_levels: List of all 10 champion levels in the game

    Returns:
        Monster level (1-18), rounded from average champion level
    """
    if not champ_levels:
        return 1
    avg = sum(champ_levels) / len(champ_levels)
    lvl = int(round(avg))
    return max(1, min(MAX_CHAMP_LEVEL, lvl))


def lerp_stat(base: float, max_val: float, level: int, max_level: int = MAX_CHAMP_LEVEL) -> float:
    """
    Linearly interpolate a stat from base to max based on level.

    Args:
        base: Base value at level 1
        max_val: Maximum value at max_level
        level: Current level
        max_level: Maximum level (default: 18)

    Returns:
        Interpolated stat value
    """
    if max_level <= 1 or base == max_val:
        return base
    t = (level - 1) / (max_level - 1)  # 0 at level 1, 1 at level max_level
    return base + t * (max_val - base)


def get_monster_stats(camp_name: str, champ_levels: List[int]) -> dict:
    """
    Get scaled stats for a jungle camp based on current champion levels.

    Args:
        camp_name: Name of the camp (e.g., "Blue Sentinel", "Dragon")
        champ_levels: List of all champion levels in the game

    Returns:
        Dictionary with scaled stats: hp, armor, mr, attack_damage,
        attack_speed, gold, xp, and notes

    Raises:
        KeyError: If camp_name is not recognized
    """
    if camp_name not in MONSTER_SCALING:
        raise KeyError(f"Unknown camp: {camp_name}")

    scaling = MONSTER_SCALING[camp_name]
    lvl = monster_level_from_champs(champ_levels)

    hp = lerp_stat(scaling.hp[0], scaling.hp[1], lvl)
    armor = lerp_stat(scaling.armor[0], scaling.armor[1], lvl)
    mr = lerp_stat(scaling.mr[0], scaling.mr[1], lvl)
    ad = lerp_stat(scaling.attack_damage[0], scaling.attack_damage[1], lvl)
    atk_spd = scaling.attack_speed
    gold = lerp_stat(scaling.gold[0], scaling.gold[1], lvl)
    xp = lerp_stat(scaling.xp[0], scaling.xp[1], lvl)

    return {
        "level": lvl,
        "hp": hp,
        "armor": armor,
        "mr": mr,
        "attack_damage": ad,
        "attack_speed": atk_spd,
        "gold": gold,
        "xp": xp,
        "notes": scaling.notes,
    }
