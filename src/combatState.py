"""
Combat state simulation for jungle camp clearing.

This module handles the core combat mechanics between a jungler and jungle camps,
including HP tracking, damage calculation, and smite mechanics.
"""

from dataclasses import dataclass
from enum import Enum


class CampType(str, Enum):
    """Type of jungle camp (normal camps vs epic objectives)."""
    NORMAL = "normal"
    EPIC = "epic"


@dataclass
class ChampionStats:
    """
    Represents the jungler's champion state during gameplay.

    Attributes:
        level: Champion level (1-18)
        max_hp: Maximum health points
        hp: Current health points
        attack_damage: Base attack damage
        attack_speed: Attacks per second
        armor: Armor value (for future damage reduction)
        smite_charges: Number of smite charges available (typically 2)
        smite_damage: Damage dealt to monsters per smite
        smite_heal: HP restored to champion per smite
        smite_cooldown: Seconds until next charge regenerates
        time: Current game time in seconds
    """
    level: int
    max_hp: float
    hp: float
    attack_damage: float
    attack_speed: float  # attacks per second
    armor: float

    smite_charges: int
    smite_damage: float        # damage to monsters
    smite_heal: float          # heal to champ
    smite_cooldown: float      # seconds until next charge (if < max)
    time: float = 0.0          # game time in seconds


@dataclass
class CampStats:
    """
    Represents a jungle camp's state.

    Attributes:
        name: Camp identifier (e.g., "Blue Sentinel", "Dragon")
        max_hp: Maximum HP of the camp
        hp: Current HP
        dps_to_champ: Damage per second dealt to the jungler
        armor: Armor value of the camp
        respawn_time: Seconds until camp respawns after being cleared
        camp_type: NORMAL (regular camps) or EPIC (objectives)
        gold_reward: Gold granted when camp is cleared
        xp_reward: Experience granted when camp is cleared
        alive: Whether the camp is currently alive
        next_spawn_time: Game time when camp will respawn
    """
    name: str
    max_hp: float
    hp: float
    dps_to_champ: float    # how much damage per second to our jungler
    armor: float           # if you want monsters to be tanky
    respawn_time: float    # respawn delay after death, in seconds
    camp_type: CampType = CampType.NORMAL
    gold_reward: float = 0.0
    xp_reward: float = 0.0
    alive: bool = True
    next_spawn_time: float = 0.0


def simulate_camp_clear(
    champ: ChampionStats,
    camp: CampStats,
    use_smite: bool = True,
    dt: float = 0.1,
) -> tuple[ChampionStats, CampStats, float, bool]:
    """
    Simulate fighting a camp until either the camp or champion dies.

    This uses a simple time-step simulation where both the champion and camp
    deal damage to each other. Smite is automatically used when the camp is
    low enough (within 110% of smite damage).

    Args:
        champ: The champion's current stats
        camp: The jungle camp's stats
        use_smite: Whether to use smite during the fight
        dt: Time step for simulation (seconds)

    Returns:
        tuple of (updated_champ, updated_camp, time_spent, champ_died)
            - updated_champ: Champion stats after the fight
            - updated_camp: Camp stats after the fight
            - time_spent: Total seconds spent fighting
            - champ_died: True if the champion died
    """
    time_spent = 0.0

    # If camp isn't alive or champ is already dead, nothing happens
    if not camp.alive or champ.hp <= 0:
        return champ, camp, time_spent, champ.hp <= 0

    while camp.hp > 0 and champ.hp > 0:
        # Super simplified DPS model
        champ_dps = champ.attack_damage * champ.attack_speed
        camp_dps = camp.dps_to_champ

        # Optionally smite when camp low enough
        if use_smite and champ.smite_charges > 0 and camp.hp <= champ.smite_damage * 1.1:
            camp.hp -= champ.smite_damage
            champ.hp = min(champ.max_hp, champ.hp + champ.smite_heal)
            champ.smite_charges -= 1

            if camp.hp <= 0:
                break

        # Apply damage for this small time slice dt
        camp.hp -= champ_dps * dt
        champ.hp -= camp_dps * dt

        time_spent += dt
        champ.time += dt

        if champ.hp <= 0:
            champ.hp = 0
            return champ, camp, time_spent, True

    # Camp died
    if camp.hp <= 0:
        camp.hp = 0
        camp.alive = False
        camp.next_spawn_time = champ.time + camp.respawn_time

    return champ, camp, time_spent, False
