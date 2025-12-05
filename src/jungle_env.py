# jungle_env.py

from __future__ import annotations
from typing import List

from combatState import ChampionStats, CampStats, CampType, simulate_camp_clear
from gameStates import MonstersState
from monster_scaling import get_monster_stats
from map_geometry import shortest_travel_time, recall_and_travel_time


EPIC_NAMES = {"Dragon", "Void Grub", "Rift Herald", "Baron Nashor"}


def spawn_camp(camp_name: str, champ_levels: List[int], time_now: float) -> CampStats:
    """
    Convert (camp_name + current champ levels) into a CampStats object
    using the monster_scaling table.
    """
    s = get_monster_stats(camp_name, champ_levels)
    dps_to_champ = s["attack_damage"] * s["attack_speed"]
    camp_type = CampType.EPIC if camp_name in EPIC_NAMES else CampType.NORMAL

    # Simple respawn rule: 150s for normal camps, 300s for epic objectives
    respawn = 300.0 if camp_type == CampType.EPIC else 150.0

    return CampStats(
        name=camp_name,
        max_hp=s["hp"],
        hp=s["hp"],
        dps_to_champ=dps_to_champ,
        armor=s["armor"],
        respawn_time=respawn,
        camp_type=camp_type,
        gold_reward=s["gold"],
        xp_reward=s["xp"],
        alive=True,
        next_spawn_time=time_now,
    )


def current_ms(champ: ChampionStats) -> float:
    """
    For now, assume jungler always has boots = 390 MS.
    Later, compute from items/base MS.
    """
    return 390.0


def demo_blue_clear_then_recall():
    """
    Simple end-to-end demo:
      - Start jungler at blue buff.
      - Spawn a Blue Sentinel using monster_scaling.
      - Clear it with simulate_camp_clear.
      - Recall to base and walk back to blue.
    """
    # 1) Fake champ + levels
    champ = ChampionStats(
        level=3,
        max_hp=650.0,
        hp=650.0,
        attack_damage=60.0,
        attack_speed=0.7,
        armor=30.0,
        smite_charges=2,
        smite_damage=600.0,
        smite_heal=90.0,
        smite_cooldown=15.0,
        time=90.0,  # start at 1:30 when camps first spawn
    )
    champ_levels = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # 10 champs, all level 3

    # 2) Spawn Blue Sentinel using scaling
    camp_name = "Blue Sentinel"
    camp = spawn_camp(camp_name, champ_levels, time_now=champ.time)

    print(f"Spawned {camp.name} at t={champ.time:.1f}s:")
    print(f"  HP={camp.hp:.1f}, DPS to champ≈{camp.dps_to_champ:.1f}, respawn={camp.respawn_time}s")

    # 3) Simulate clearing the camp
    champ_before = champ.hp
    champ, camp, t_spent, champ_died = simulate_camp_clear(champ, camp, use_smite=True)

    print(f"\nAfter clear:")
    print(f"  Time spent fighting: {t_spent:.2f}s")
    print(f"  Game time: {champ.time:.2f}s")
    print(f"  Champ HP: {champ.hp:.1f} / {champ.max_hp:.1f} (lost {champ_before - champ.hp:.1f})")
    print(f"  Camp HP: {camp.hp:.1f} (alive={camp.alive})")
    print(f"  Champ died? {champ_died}")

    # 4) Recall to base, then walk back to blue
    position = "blue_blue"  # we started and fought at blue
    dt_recall = recall_and_travel_time(position, "blue_blue", ms=current_ms(champ))
    champ.time += dt_recall
    champ.hp = champ.max_hp  # heal to full on base

    print(f"\nAfter recall and return to blue:")
    print(f"  Extra time: {dt_recall:.2f}s")
    print(f"  Game time: {champ.time:.2f}s")
    print(f"  Champ HP: {champ.hp:.1f} / {champ.max_hp:.1f}")


if __name__ == "__main__":
    demo_blue_clear_then_recall()