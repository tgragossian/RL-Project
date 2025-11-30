from combatState import ChampionStats
from gameStates import LaneState, LaneType, PushedState, JungleHeuristicPolicy
from map_geometry import shortest_travel_time, recall_and_travel_time


def current_ms(champ: ChampionStats) -> float:
    """
    For now, assume jungler always has boots = 390 MS.
    Later, compute from items/base MS.
    """
    return 390.0


def demo_sim():
    # -------------------------------
    # 1. Set up a fake jungler state
    # -------------------------------
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
        time=0.0,
    )

    # Start at blue buff
    position = "blue_blue"

    # -----------------------------------
    # 2. Move to blue_raptors using map
    # -----------------------------------
    dt_move = shortest_travel_time(position, "blue_raptors", ms=current_ms(champ))
    champ.time += dt_move
    position = "blue_raptors"
    print(f"Moved to {position} in {dt_move:.2f}s, game time={champ.time:.2f}")

    # ------------------------
    # 3. Recall back to base,
    #    then back to blue
    # ------------------------
    dt_recall = recall_and_travel_time(position, "blue_blue", ms=current_ms(champ))
    champ.time += dt_recall
    position = "blue_blue"
    champ.hp = champ.max_hp
    print(
        f"Recalled and returned to {position} in {dt_recall:.2f}s, "
        f"game time={champ.time:.2f}, hp={champ.hp}/{champ.max_hp}"
    )

    # ------------------------------------------
    # 4. Lane state + heuristic gank decision
    # ------------------------------------------
    policy = JungleHeuristicPolicy()

    lanes = [
        LaneState(
            lane_type=LaneType.TOP,
            win_prob=0.58,
            pushed_state=PushedState.UNDER_TOWER,
            ally_hp=0.6,
            enemy_hp=0.7,
        ),
        LaneState(
            lane_type=LaneType.MID,
            win_prob=0.48,
            pushed_state=PushedState.EVEN,
            ally_hp=0.7,
            enemy_hp=0.5,
        ),
        LaneState(
            lane_type=LaneType.BOT,
            win_prob=0.65,
            pushed_state=PushedState.PUSHING,
            ally_hp=0.8,
            enemy_hp=0.3,
        ),
    ]

    jungler_hp_frac = champ.hp / champ.max_hp
    action = policy.choose_action(lanes, jungler_hp=jungler_hp_frac)
    print(f"Heuristic policy recommends: {action}")


if __name__ == "__main__":
    demo_sim()
