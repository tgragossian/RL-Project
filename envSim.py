from gameStates import LaneState, LaneType, PushedState, JungleHeuristicPolicy

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

action = policy.choose_action(lanes, jungler_hp=0.75)
# e.g. "GANK_BOT" or "FARM_OR_RESET"
