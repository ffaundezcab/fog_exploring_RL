"""Microbe grid-world environment.

A small, fully-discrete environment used as the running example in Lecture 1
of the course. A microbe lives on a rectangular grid, starts from a fixed
"safe" cell, and must reach a "nutrient" cell while avoiding "toxic" cells.
Stepping on a toxic cell teleports the microbe back to the start with a
heavy penalty; reaching the nutrient terminates the episode.

The default configuration matches the Cliff Walking benchmark of
Sutton & Barto (Reinforcement Learning: An Introduction, Section 6.5).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces

# Tile types
EMPTY = 0
OBSTACLE = 1
TRAP = 2
GOAL = 3
FOG = 4

# Action encoding shared with Gymnasium's CliffWalking convention.
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

# Row-column displacement for each action.
_DELTA = {
    ACTION_UP: (-1, 0),
    ACTION_RIGHT: (0, +1),
    ACTION_DOWN: (+1, 0),
    ACTION_LEFT: (0, -1),
}

# For each action, its two perpendicular actions (used in slippery dynamics).
_PERPENDICULAR = {
    ACTION_UP: (ACTION_LEFT, ACTION_RIGHT),
    ACTION_RIGHT: (ACTION_UP, ACTION_DOWN),
    ACTION_DOWN: (ACTION_LEFT, ACTION_RIGHT),
    ACTION_LEFT: (ACTION_UP, ACTION_DOWN),
}


class FogGridEnv(gym.Env):
    """
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        start_pos: tuple[int, int] = (0, 0),
        goal_pos: tuple[int, int] = (7, 7),
        obstacle_tiles: Optional[list[tuple[int, int]]] = None,
        trap_tiles: Optional[list[tuple[int, int]]] = None,
        vision_radius: int = 2,
        step_cost: float = -1.0,
        trap_penalty: float = -15.0,
        goal_reward: float = 100.0,
        randomize_map: bool = False,
        obstacle_density: float = 0.15,
        trap_density: float = 0.1,
        local_view: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if obstacle_tiles is None:
            obstacle_tiles = []
            
        if trap_tiles is None:
            trap_tiles = []

        # map config
        self.height = int(height)
        self.width = int(width)
        self.start_pos = tuple(start_pos)
        self.goal_pos = tuple(goal_pos)
        self.trap_tiles = set(map(tuple, trap_tiles))
        self.obstacle_tiles = set(map(tuple, obstacle_tiles))
        self.randomize_map = bool(randomize_map)
        self.obstacle_density = float(obstacle_density)
        self.trap_density = float(trap_density)
        self.vision_radius = int(vision_radius)
        self.local_view = bool(local_view)
        
        # agent config
        self.vision_radius = int(vision_radius)
        self.step_cost = float(step_cost)
        self.trap_penalty = float(trap_penalty)
        self.goal_reward = float(goal_reward)

        self._validate_layout()

        # actions
        self.action_space = spaces.Discrete(4)
        
        # depending on local view obs space changes
        n_visible_cells = (2*self.vision_radius + 1)**2 # a square with two tiles distance from agent pos
        
        if not self.local_view:
            self.observation_space = spaces.Discrete(self.height * self.width)
        else:
            # this is where the state space explosion happens
            # having a local view with 2 tiles to any direction can increase the number of states the agent stores
            # given the nature of our random map generation many of these states will be visited very few
            self.observation_space = spaces.Tuple((spaces.Discrete(self.height),
                                                   spaces.Discrete(self.width),
                                                   *[spaces.Discrete(5) for _ in range(n_visible_cells)],))

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        # Episode state.
        self._agent_pos: Optional[tuple[int, int]] = None
        self._episode_return: float = 0.0

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _validate_layout(self) -> None:
        def in_bounds(pos: tuple[int, int]) -> bool:
            r, c = pos
            return 0 <= r < self.height and 0 <= c < self.width
        
        if not 0.0 <= self.obstacle_density < 1.0:
            raise ValueError("obstacle_density must be between 0 and 1")
        
        if not 0.0 <= self.trap_density < 1.0:
            raise ValueError("trap_density must be between 0 and 1")
        
        if self.vision_radius < 0:
            raise ValueError("vision_radius must be >= 0")

        if not in_bounds(self.start_pos):
            raise ValueError(f"start_pos {self.start_pos} is out of bounds.")
        if not in_bounds(self.goal_pos):
            raise ValueError(f"goal_pos {self.goal_pos} is out of bounds.")
        
        for tile in self.obstacle_tiles:
            if not in_bounds(tile):
                raise ValueError(f"Obstacle tile {tile} is out of bounds.")
            
        for tile in self.trap_tiles:
            if not in_bounds(tile):
                raise ValueError(f"Trap tile {tile} is out of bounds.")
                    
        if self.start_pos in self.obstacle_tiles:
            raise ValueError("start_pos cannot be an obstacle tile.")
        # both obstacle tile restrictions
        if self.goal_pos in self.obstacle_tiles:
            raise ValueError("goal_pos cannot be an obstacle tile.")
        
        if self.start_pos in self.trap_tiles:
            raise ValueError("start_pos cannot be a trap tile.")
        # both trap tile restrictions 
        if self.goal_pos in self.trap_tiles:
            raise ValueError("goal_pos cannot be a trap tile.")
        
        if self.start_pos == self.goal_pos:
            raise ValueError("start_pos and goal_pos must differ.")
        
        if self.obstacle_tiles & self.trap_tiles:
            raise ValueError("tile cannot be an obstacle and a trap at the same time.")

    def _generate_random_map(self) -> None:
        while True:
            # all available tiles: not start or goal tiles
            all_tiles = [
                (r, c)
                for r in range(self.height)
                for c in range(self.width)
                if (r, c) not in {self.start_pos, self.goal_pos}
            ]

            n_available = len(all_tiles)

            # based on the previously defined density how many of each should we have
            n_obstacles = int(self.obstacle_density*n_available)
            n_traps = int(self.trap_density*n_available)

            # shuffle all available tiles
            shuffled = self.np_random.permutation(len(all_tiles))

            # take the first n_obstacles as obstacle tiles
            obstacle_idx = shuffled[:n_obstacles]
            
            # then the same for trap tiles
            trap_idx = shuffled[n_obstacles:n_obstacles + n_traps]

            # define them as a passable dict
            self.obstacle_tiles = {
                all_tiles[i] for i in obstacle_idx
            }

            self.trap_tiles = {
                all_tiles[i] for i in trap_idx
            }
            
            if self._has_path_to_goal():
                break
        
    def _has_path_to_goal(self) -> bool:
        # BFS algorithm for finding if there's any path to the goal state
        queue = deque([self.start_pos])
        visited = {self.start_pos}

        while queue:
            pos = queue.popleft()

            if pos == self.goal_pos:
                return True

            # at each position, ask which tiles are reachable within 1 step (in all directions)
            for action in range(4):
                nxt = self._intended_landing(pos, action)
                
                # if obstacle then continue to the next action
                if nxt in self.obstacle_tiles:
                    continue
                # if state wasn't visited, then record it and continue exploring
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        return False
    
    def _get_local_view(self) -> tuple[int, ...]:
        if self._agent_pos is None:
            raise RuntimeError
        
        ar, ac = self._agent_pos
        view = []
        
        for dr in range(-self.vision_radius, self.vision_radius+1):
            for dc in range(-self.vision_radius, self.vision_radius +1):
                
                r = ar + dr
                c = ac + dc
                pos = (r,c)
                
                if not (0 <= r < self.height and 0 <= c < self.width):
                    tile = FOG
                elif pos in self.obstacle_tiles:
                    tile = OBSTACLE
                elif pos in self.obstacle_tiles:
                    tile = TRAP
                elif pos == self.goal_pos:
                    tile = GOAL
                else:
                    tile = EMPTY
        
        return tuple(view)
        
        
    
    def _pos_to_obs(self, pos: tuple[int, int]) -> int:
        """Row-major flattening: s = row * width + col."""
        return pos[0] * self.width + pos[1]
    
    def _get_obs(self):
        if self._agent_pos is None:
            raise RuntimeError("Observation needed before reset")
        
        # if local view is not activated for the env
        if not self.local_view:
            return self._pos_to_obs(self._agent_pos)
        
        r,c = self._agent_pos
        local_view = self._get_local_view()
        
        return (r,c, *local_view)

    def _intended_landing(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        """Apply action displacement, clipping at grid boundaries."""
        dr, dc = _DELTA[action]
        r = max(0, min(self.height - 1, pos[0] + dr))
        c = max(0, min(self.width - 1, pos[1] + dc))
        return (r, c)

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        if self.randomize_map:
            self._generate_random_map()

        self._agent_pos = self.start_pos
        self._episode_return = 0.0
        
        obs = self._get_obs()

        return obs, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self._agent_pos is None:
            raise RuntimeError("step() called before reset().")
        
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0..3.")

        landing = self._intended_landing(self._agent_pos, action)
        if landing in self.obstacle_tiles:
            # position remains unchanged
            reward = self.step_cost
            terminated = False
        elif landing == self.goal_pos:
            self._agent_pos = landing
            reward = self.goal_reward
            terminated = True
        elif landing in self.trap_tiles:
            self._agent_pos = landing
            reward = self.trap_penalty
            terminated = False
        else:
            self._agent_pos = landing
            reward = self.step_cost
            terminated = False

        truncated = False
        
        self._episode_return += reward
        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    # Color palette (RGB in [0, 1]).
    _COLOR_FREE = np.array([0.96, 0.96, 0.92])    # pale background
    _COLOR_TOXIC = np.array([0.55, 0.10, 0.55])   # purple band
    _COLOR_GOAL = np.array([0.20, 0.55, 0.20])    # nutrient green
    _COLOR_START = np.array([0.85, 0.85, 0.70])   # faint marker
    _COLOR_GRID = np.array([0.30, 0.30, 0.30])    # grid lines

    # Microbe color endpoints: from "healthy" to "depleted".
    _MICROBE_HEALTHY = np.array([0.10, 0.75, 0.20])  # vivid green
    _MICROBE_DEPLETED = np.array([0.55, 0.05, 0.05])  # dark red
    _ENERGY_DEPLETION_SCALE = 100.0  # cumulative cost at which the microbe
                                     # is considered fully depleted

    _CELL_PIXELS = 32
    _MARGIN = 1

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB image of the current grid as a NumPy array.

        The microbe is drawn as a filled disc whose color interpolates
        between healthy green and depleted red as a function of the
        cumulative reward collected since the last reset. Concretely,
        ``alpha = clip(-episode_return / _ENERGY_DEPLETION_SCALE, 0, 1)``
        and the microbe color is ``(1 - alpha) * healthy + alpha * depleted``.
        """
        if self.render_mode != "rgb_array":
            return None
        if self._agent_pos is None:
            raise RuntimeError("render() called before reset().")

        cell = self._CELL_PIXELS
        m = self._MARGIN
        h_px = self.height * cell
        w_px = self.width * cell

        # Background
        img = np.tile(self._COLOR_FREE, (h_px, w_px, 1))

        # Paint special cells.
        def paint_cell(pos: tuple[int, int], color: np.ndarray) -> None:
            r, c = pos
            r0, r1 = r * cell + m, (r + 1) * cell - m
            c0, c1 = c * cell + m, (c + 1) * cell - m
            img[r0:r1, c0:c1] = color

        paint_cell(self.start_pos, self._COLOR_START)
        for tc in self.toxic_cells:
            paint_cell(tc, self._COLOR_TOXIC)
        paint_cell(self.goal_pos, self._COLOR_GOAL)

        # Grid lines.
        for r in range(self.height + 1):
            img[min(r * cell, h_px - 1), :] = self._COLOR_GRID
        for c in range(self.width + 1):
            img[:, min(c * cell, w_px - 1)] = self._COLOR_GRID

        # Microbe color from cumulative episode return.
        # episode_return is non-positive, so -episode_return >= 0.
        alpha = float(np.clip(-self._episode_return / self._ENERGY_DEPLETION_SCALE,
                              0.0, 1.0))
        microbe_color = (1.0 - alpha) * self._MICROBE_HEALTHY + alpha * self._MICROBE_DEPLETED

        # Draw the microbe as a filled disc inside its current cell.
        ar, ac = self._agent_pos
        cy = ar * cell + cell // 2
        cx = ac * cell + cell // 2
        radius = cell // 2 - 2 * m
        yy, xx = np.ogrid[:h_px, :w_px]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        img[disc] = microbe_color

        return (img * 255).astype(np.uint8)

    def close(self) -> None:
        # No external resources to release.
        pass
    
    def print_map(self) -> None:
        """
        simple text representation of the map
        """
        for r in range(self.height):
            row = []
            for c in range(self.width):
                pos = (r, c)

                if pos == self._agent_pos:
                    symbol = "A"
                elif pos == self.start_pos:
                    symbol = "S"
                elif pos == self.goal_pos:
                    symbol = "G"
                elif pos in self.obstacle_tiles:
                    symbol = "O"
                elif pos in self.trap_tiles:
                    symbol = "X"
                else:
                    symbol = "."

                row.append(symbol)

            print(" ".join(row))