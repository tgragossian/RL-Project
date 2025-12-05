"""
Gymnasium-compatible RL environment for League of Legends jungling.

This environment simulates jungle macro decisions:
- Which camp to clear next
- When to recall
- Basic objective timing

State space: 71-dim vector (same as BC model)
Action space: 17 camps (same as BC model)
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from combatState import ChampionStats, CampStats, simulate_camp_clear
from gameStates import MonstersState
from jungle_env import spawn_camp, current_ms
from map_geometry import shortest_travel_time, recall_and_travel_time
from training_data import ALL_CAMPS, CAMP_TO_IDX, create_state_vector


# Map our camp names to internal simulation names
CAMP_NAME_MAPPING = {
    "blue_blue": "Blue Sentinel",
    "blue_gromp": "Gromp",
    "blue_wolves": "Murk Wolves",
    "blue_raptors": "Crimson Raptor",
    "blue_red": "Red Brambleback",
    "blue_krugs": "Ancient Krug",
    "red_red": "Red Brambleback",
    "red_krugs": "Ancient Krug",
    "red_raptors": "Crimson Raptor",
    "red_wolves": "Murk Wolves",
    "red_gromp": "Gromp",
    "red_blue": "Blue Sentinel",
    "dragon": "Dragon",
    "baron": "Baron Nashor",
    "herald": "Rift Herald",
    # Scuttles - for now, just use a simple camp
    "scuttle_bot": "Crimson Raptor",  # placeholder
    "scuttle_top": "Crimson Raptor",  # placeholder
}


class JungleGymEnv(gym.Env):
    """
    Gymnasium environment for jungle decision-making.

    Observation: 71-dim state vector
    Action: Integer 0-16 (which camp to clear next)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_episode_steps: int = 100,
        starting_level: int = 3,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.starting_level = starting_level
        self.render_mode = render_mode

        # Action space: 17 camps
        self.action_space = gym.spaces.Discrete(len(ALL_CAMPS))

        # Observation space: 71-dim continuous
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(71,),
            dtype=np.float32
        )

        # Internal state
        self.champion: Optional[ChampionStats] = None
        self.monsters: Optional[MonstersState] = None
        self.current_position: str = "blue_blue"
        self.recent_camps: list = []
        self.step_count: int = 0

        # Track stats for reward
        self.total_gold: float = 0.0
        self.total_xp: float = 0.0
        self.camps_cleared: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Initialize champion
        self.champion = ChampionStats(
            level=self.starting_level,
            max_hp=650.0,
            hp=650.0,
            attack_damage=60.0,
            attack_speed=0.7,
            armor=30.0,
            smite_charges=2,
            smite_damage=600.0,
            smite_heal=90.0,
            smite_cooldown=15.0,
            time=90.0,  # Game starts at 1:30 when camps spawn
        )

        # Initialize monster spawns
        self.monsters = MonstersState()
        self.monsters.reset()
        self.monsters.update(self.champion.time)

        # Reset tracking
        self.current_position = "blue_blue"
        self.recent_camps = []
        self.step_count = 0
        self.total_gold = 0.0
        self.total_xp = 0.0
        self.camps_cleared = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one action (clear a camp).

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.champion is None:
            raise RuntimeError("Call reset() before step()")

        self.step_count += 1

        # Get camp name from action
        camp_name = ALL_CAMPS[action]

        # Check if camp is alive
        # Map to monster state key (remove team prefix for shared camps)
        monster_key = self._get_monster_key(camp_name)

        reward = 0.0

        # Update monster spawn states
        self.monsters.update(self.champion.time)

        if not self.monsters.is_alive(monster_key):
            # Camp not available - negative reward
            reward = -1.0
            info = self._get_info()
            info["invalid_action"] = True

            # Still get observation and continue
            obs = self._get_observation()
            terminated = False
            truncated = self.step_count >= self.max_episode_steps

            return obs, reward, terminated, truncated, info

        # Camp is alive - execute the clear

        # 1. Travel to camp
        travel_time = shortest_travel_time(
            self.current_position,
            camp_name,
            ms=current_ms(self.champion)
        )
        self.champion.time += travel_time

        # 2. Spawn the camp
        champ_levels = [self.champion.level] * 10  # Simplified: all same level
        sim_camp_name = CAMP_NAME_MAPPING.get(camp_name, "Blue Sentinel")
        camp = spawn_camp(sim_camp_name, champ_levels, self.champion.time)

        # 3. Clear the camp
        hp_before = self.champion.hp
        self.champion, camp, time_spent, died = simulate_camp_clear(
            self.champion, camp, use_smite=(self.champion.smite_charges > 0)
        )

        # 4. Update state
        self.current_position = camp_name
        self.recent_camps.append(camp_name)
        self.monsters.kill(monster_key, self.champion.time)

        # 5. Track rewards
        gold_gained = camp.gold_reward if not died else 0
        xp_gained = camp.xp_reward if not died else 0

        self.total_gold += gold_gained
        self.total_xp += xp_gained
        self.camps_cleared += 1

        # 6. Calculate reward
        reward = self._calculate_reward(
            gold_gained=gold_gained,
            xp_gained=xp_gained,
            hp_lost=hp_before - self.champion.hp,
            time_spent=time_spent + travel_time,
            died=died
        )

        # 7. Check termination
        terminated = died or self.champion.time >= 1200.0  # 20 minutes
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_observation()
        info = self._get_info()
        info["camp_cleared"] = camp_name
        info["gold_gained"] = gold_gained
        info["xp_gained"] = xp_gained
        info["died"] = died

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current state observation (71-dim vector)."""
        last_camp = self.recent_camps[-1] if self.recent_camps else "none"

        state = create_state_vector(
            current_time=self.champion.time,
            current_position=self._position_to_coords(self.current_position),
            last_camp=last_camp,
            recent_camps=self.recent_camps[-3:] if len(self.recent_camps) >= 3 else self.recent_camps
        )

        return state.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        return {
            "game_time": self.champion.time,
            "champion_hp": self.champion.hp,
            "total_gold": self.total_gold,
            "total_xp": self.total_xp,
            "camps_cleared": self.camps_cleared,
            "position": self.current_position,
            "step": self.step_count,
        }

    def _calculate_reward(
        self,
        gold_gained: float,
        xp_gained: float,
        hp_lost: float,
        time_spent: float,
        died: bool
    ) -> float:
        """
        Reward function for jungle performance.

        Goals:
        - Maximize gold/XP per minute
        - Minimize HP loss (efficient clearing)
        - Heavily penalize death
        """
        if died:
            return -10.0  # Death is very bad

        # Reward efficiency: gold per second
        gold_per_sec = gold_gained / max(time_spent, 1.0)

        # Base reward is gold gained (scaled)
        reward = gold_gained / 100.0  # Scale to ~0.3-0.8 per clear

        # Bonus for efficiency
        reward += gold_per_sec / 10.0

        # Small penalty for HP loss (encourage healthy clearing)
        reward -= hp_lost / 1000.0

        return reward

    def _get_monster_key(self, camp_name: str) -> str:
        """
        Map action camp name to monster state key.

        For camps that are team-specific in action space but shared in monster state.
        """
        # For now, assume monster state uses same keys
        # You may need to adjust based on your MonstersState implementation
        if camp_name.startswith("blue_") or camp_name.startswith("red_"):
            # These are team-specific camps
            return camp_name
        else:
            # Shared camps (dragon, baron, herald, scuttle)
            return camp_name

    def _position_to_coords(self, position: str) -> Tuple[int, int]:
        """
        Convert position name to (x, y) coordinates.

        Approximate positions on Summoner's Rift.
        """
        # Simplified coordinates (you can refine these)
        coords = {
            "blue_blue": (3800, 7900),
            "blue_gromp": (2200, 8400),
            "blue_wolves": (3800, 6500),
            "blue_raptors": (6900, 5400),
            "blue_red": (7000, 4200),
            "blue_krugs": (8400, 2700),
            "red_red": (10800, 10800),
            "red_krugs": (12600, 12300),
            "red_raptors": (10100, 9600),
            "red_wolves": (11200, 8500),
            "red_gromp": (12800, 6600),
            "red_blue": (11200, 7100),
            "dragon": (9800, 4000),
            "baron": (5200, 10200),
            "herald": (5200, 10200),
            "scuttle_bot": (8500, 5000),
            "scuttle_top": (6500, 10000),
        }
        return coords.get(position, (7500, 7500))  # Default to center

    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"\n{'='*50}")
            print(f"Game Time: {self.champion.time:.1f}s")
            print(f"Position: {self.current_position}")
            print(f"HP: {self.champion.hp:.0f}/{self.champion.max_hp:.0f}")
            print(f"Gold: {self.total_gold:.0f}")
            print(f"Camps cleared: {self.camps_cleared}")
            print(f"Recent path: {' -> '.join(self.recent_camps[-5:])}")
            print(f"{'='*50}")


if __name__ == "__main__":
    # Test the environment
    print("Testing JungleGymEnv...")

    env = JungleGymEnv(render_mode="human")
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")

    # Take a few random actions
    for i in range(5):
        action = env.action_space.sample()
        camp_name = ALL_CAMPS[action]
        print(f"\nStep {i+1}: Attempting to clear {camp_name}")

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Reward: {reward:.3f}")
        print(f"Info: {info}")

        if terminated or truncated:
            print("Episode ended!")
            break

    print("\n✓ Environment test passed!")
