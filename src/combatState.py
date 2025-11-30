from dataclasses import dataclass


@dataclass
class ChampionStats:
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
    name: str
    max_hp: float
    hp: float
    dps_to_champ: float    # how much damage per second to our jungler
    armor: float           # if you want monsters to be tanky
    respawn_time: float    # respawn delay after death, in seconds
    alive: bool = True
    next_spawn_time: float = 0.0


def simulate_camp_clear(
    champ: ChampionStats,
    camp: CampStats,
    use_smite: bool = True,
    dt: float = 0.1,
) -> tuple[ChampionStats, CampStats, float, bool]:
    """
    Simulate fighting a camp until:
      - camp dies, or
      - champ dies.

    Returns:
      (updated_champ, updated_camp, time_spent, champ_died)
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
