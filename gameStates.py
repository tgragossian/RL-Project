from dataclasses import dataclass, field
import numpy as np

class player_state:
    def __init__(self, gold, items, health, mana):
        self.gold = gold
        self.items = items
        self.health = health
        self.mana = mana


@dataclass
class MonstersState:
    """
    Tracks jungle camp spawn/respawn state.

    - Nothing is alive at t = 0.
    - Each camp has a first_spawn and respawn_interval (in seconds).
    - You call `update(time_now)` every step to refresh which camps are alive.
    - When a camp is cleared, call `kill(camp_name, time_now)`.
    """

    # Config per camp: first spawn time + respawn interval (seconds)
    camp_configs: dict = field(default_factory=lambda: {
        # Blue side
        "blue_blue":   {"first_spawn": 90.0, "respawn": 150.0},  # 1:30, 2:30 respawn
        "blue_gromp":  {"first_spawn": 90.0, "respawn": 150.0},
        "blue_wolves": {"first_spawn": 90.0, "respawn": 150.0},

        # Red side
        "red_red":     {"first_spawn": 90.0, "respawn": 150.0},
        "red_krugs":   {"first_spawn": 90.0, "respawn": 150.0},
        "red_raptors": {"first_spawn": 90.0, "respawn": 150.0},

        # Neutral objectives (numbers approximate; tweak as you like)
        "dragon":      {"first_spawn": 300.0, "respawn": 300.0},   # 5:00, 5:00
        "herald":      {"first_spawn": 480.0, "respawn": 360.0},   # 8:00, etc.
        "baron":       {"first_spawn": 1200.0, "respawn": 360.0},  # 20:00
    })

    # These are filled in at reset / __post_init__
    camp_names: list = field(init=False)
    alive: dict = field(init=False)
    next_spawn_time: dict = field(init=False)

    def __post_init__(self):
        self.camp_names = list(self.camp_configs.keys())
        self.reset()

    # ---------------- core API ---------------- #

    def reset(self):
        """Call at the start of an episode."""
        # Nothing alive at game start
        self.alive = {name: False for name in self.camp_names}
        # Next spawn is the first_spawn for each camp
        self.next_spawn_time = {
            name: cfg["first_spawn"] for name, cfg in self.camp_configs.items()
        }

    def update(self, time_now: float):
        """
        Update alive flags given the current game time.

        Call this at every env.step *before* you use the alive mask.
        """
        for name in self.camp_names:
            if not self.alive[name] and time_now >= self.next_spawn_time[name]:
                self.alive[name] = True

    def kill(self, camp: str, time_now: float):
        """
        Called when you finish clearing a camp.
        Sets it to dead and schedules the next respawn.
        """
        if camp not in self.alive:
            raise ValueError(f"Unknown camp: {camp}")
        if not self.alive[camp]:
            # Already dead, nothing to do
            return
        self.alive[camp] = False
        respawn = self.camp_configs[camp]["respawn"]
        self.next_spawn_time[camp] = time_now + respawn

    def is_alive(self, camp: str) -> bool:
        if camp not in self.alive:
            raise ValueError(f"Unknown camp: {camp}")
        return self.alive[camp]

    def all_alive_mask(self) -> np.ndarray:
        """
        0/1 mask in fixed order, for RL observation vector.
        """
        return np.array(
            [1.0 if self.alive[name] else 0.0 for name in self.camp_names],
            dtype=np.float32,
        )

    def all_dead(self) -> bool:
        return not any(self.alive.values())
    
# gameStates.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class LaneType(str, Enum):
    TOP = "TOP"
    MID = "MID"
    BOT = "BOT"


class PushedState(str, Enum):
    UNDER_TOWER = "UNDER_TOWER"  # wave near your own tower
    EVEN = "EVEN"                # roughly in the middle
    PUSHING = "PUSHING"          # wave near enemy tower / overextended


@dataclass
class LaneState:
    """
    Abstract representation of a single lane at some time t.

    All numbers are normalized to [0, 1] so we don't tie this to
    specific HP or probability scales.
    """
    lane_type: LaneType          # TOP / MID / BOT
    win_prob: float              # P(win if jungler ganks right now), in [0, 1]
    pushed_state: PushedState    # UNDER_TOWER / EVEN / PUSHING
    ally_hp: float               # ally HP fraction [0, 1]
    enemy_hp: float              # enemy HP fraction [0, 1]

    def clamped(self) -> "LaneState":
        """
        Return a copy with all continuous values clamped into [0, 1].
        Useful if your simulator sometimes overshoots a bit.
        """
        return LaneState(
            lane_type=self.lane_type,
            win_prob=max(0.0, min(1.0, self.win_prob)),
            pushed_state=self.pushed_state,
            ally_hp=max(0.0, min(1.0, self.ally_hp)),
            enemy_hp=max(0.0, min(1.0, self.enemy_hp)),
        )


class JungleHeuristicPolicy:
    """
    Hand-crafted jungler policy to decide whether to gank, and which lane.

    This is *not* RL yet – it's a baseline "expert" signal that you can:
      - use directly to generate trajectories, or
      - compare against your learned policy later.
    """

    def __init__(
        self,
        gank_threshold: float = 1.5,
        low_hp_cleanup_threshold: float = 0.2,
        min_standard_win_prob: float = 0.5,
        min_dive_win_prob: float = 0.52,
        min_jungler_hp_for_gank: float = 0.35,
    ):
        # Tunable knobs
        self.gank_threshold = gank_threshold
        self.low_hp_cleanup_threshold = low_hp_cleanup_threshold
        self.min_standard_win_prob = min_standard_win_prob
        self.min_dive_win_prob = min_dive_win_prob
        self.min_jungler_hp_for_gank = min_jungler_hp_for_gank

    # -----------------------------
    # Core scoring function
    # -----------------------------
    def gank_score(self, lane: LaneState) -> float:
        """
        Compute a continuous "gank desirability" score for a single lane.

        Higher score = more attractive gank.
        """
        lane = lane.clamped()
        score = 0.0

        # 1. How favorable is the fight? (win_prob centered at 0.5)
        #    If win_prob = 0.5 → term ~ 0
        #    If win_prob = 0.8 → term ~ +0.9
        #    If win_prob = 0.2 → term ~ -0.9
        score += 3.0 * (lane.win_prob - 0.5)

        # 2. Enemy low HP makes gank more appealing:
        #    enemy_hp = 1.0 → +0
        #    enemy_hp = 0.0 → +2.0
        score += 2.0 * (1.0 - lane.enemy_hp)

        # 3. Ally HP helps; you don't want to gank with a 5% HP laner.
        score += 1.0 * lane.ally_hp

        # 4. Wave state:
        #    UNDER_TOWER → good for defensive/countergank.
        #    EVEN        → neutral.
        #    PUSHING     → good only if enemy is low; otherwise risky (dives).
        if lane.pushed_state == PushedState.UNDER_TOWER:
            score += 1.0
        elif lane.pushed_state == PushedState.PUSHING:
            # If enemy is low, pushing is good for a dive.
            score += 0.5 if lane.enemy_hp <= 0.3 else -0.5

        # 5. Lane type adjustment:
        #    Diving side lanes usually safer than MID (longer lane, fewer TPs).
        if lane.lane_type in (LaneType.TOP, LaneType.BOT):
            score += 0.3

        # 6. Bonus: “free kill” cleanup when enemy is *very* low
        if lane.enemy_hp <= self.low_hp_cleanup_threshold and lane.win_prob >= 0.5:
            score += 0.5

        return score

    # -----------------------------
    # High-level decision
    # -----------------------------
    def choose_action(
        self,
        lanes: List[LaneState],
        jungler_hp: float,
        can_gank_now: bool = True,
    ) -> str:
        """
        Decide a high-level action string like:
          - "GANK_TOP", "GANK_MID", "GANK_BOT"
          - "FARM_OR_RESET"

        `jungler_hp` is in [0, 1].
        """
        jungler_hp = max(0.0, min(1.0, jungler_hp))

        # Global safety check: too low HP or other reasons to avoid fights.
        if (jungler_hp < self.min_jungler_hp_for_gank) or (not can_gank_now):
            return "FARM_OR_RESET"

        if not lanes:
            return "FARM_OR_RESET"

        # Pick lane with best score
        best_lane = max(lanes, key=self.gank_score)
        best_score = self.gank_score(best_lane)

        if best_score >= self.gank_threshold:
            return f"GANK_{best_lane.lane_type.value}"
        else:
            return "FARM_OR_RESET"


        