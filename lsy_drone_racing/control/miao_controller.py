"""Sequential Dynamic Waypoint Controller fully using obs for DroneRacing-v0 with XY-only gate extension."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ObsBasedWaypointController(Controller):
    """Follow gates sequentially using only obs, with gate exit extension only in XY."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # control parameters
        self._tolerance = 0.14     # waypoint reach tolerance
        self._gate_radius = 0.7    # gate fuzzy area radius
        self._step_length = 0.4    # waypoint interpolation step length
        self._extension_steps = 3  # gate exit extension steps

        # state variables
        self._current_wp_idx = 0
        self._finished = False
        self._extension_done = False  # current gate extension has been generated
        self._waypoints = self._generate_waypoints(obs)

    def _linear_interpolation(self, start, end):
        """Generate a list of linear interpolation waypoints from start to end."""
        vec = end - start
        dist = np.linalg.norm(vec)
        if dist == 0:
            return np.array([start])
        direction = vec / dist
        num_steps = max(int(dist / self._step_length), 1)
        return np.array([start + direction * (i * self._step_length) for i in range(1, num_steps + 1)])

    def _generate_waypoints(self, obs):
        """Generate initial waypoints based on obs."""
        start_pos = obs["pos"].copy()
        if "target_gate" in obs and "gates_pos" in obs:
            gate_idx = obs["target_gate"]
            gate_pos = obs["gates_pos"][gate_idx]
            return self._linear_interpolation(start_pos, gate_pos)
        else:
            return np.array([start_pos])

    def compute_control(self, obs, info=None):
        if self._finished:
            return np.zeros(13, dtype=np.float32)

        # current gate index and position
        if "target_gate" in obs and "gates_pos" in obs:
            gate_idx = obs["target_gate"]
            gate_pos = obs["gates_pos"][gate_idx]
        else:
            self._finished = True
            return np.zeros(13, dtype=np.float32)

        # If entering gate fuzzy area, update waypoint and reset extension flag
        dist_to_gate = np.linalg.norm(gate_pos - obs["pos"])
        if dist_to_gate < self._gate_radius:
            self._waypoints = self._linear_interpolation(obs["pos"], gate_pos)
            self._current_wp_idx = 0
            self._extension_done = False  # new gate, extension not generated yet

        # current waypoint
        target_pos = self._waypoints[self._current_wp_idx]
        pos_error = target_pos - obs["pos"]
        print(f"Current WP idx: {self._current_wp_idx}, Target Pos: {target_pos}, Pos Error: {pos_error}")

        # If waypoint is reached, proceed to next
        if np.linalg.norm(pos_error) < self._tolerance:
            self._current_wp_idx += 1

            if self._current_wp_idx >= len(self._waypoints):
                # Gate exit extension, generate only once
                if not self._extension_done:
                    last_wp = self._waypoints[-1]
                    prev_wp = self._waypoints[-2] if len(self._waypoints) >= 2 else obs["pos"]
                    direction = last_wp - prev_wp
                    norm = np.linalg.norm(direction[:2])  # only consider XY plane
                    if norm > 0:
                        direction_xy = direction[:2] / norm
                        extension = np.array([
                            [last_wp[0] + direction_xy[0] * self._step_length * i,
                             last_wp[1] + direction_xy[1] * self._step_length * i,
                             last_wp[2]]  # Z remains unchanged
                            for i in range(1, self._extension_steps + 1)
                        ])
                        self._waypoints = np.vstack([self._waypoints, extension])
                        self._extension_done = True
                        self._current_wp_idx = len(self._waypoints) - len(extension)

        # Update control output
        target_pos = self._waypoints[self._current_wp_idx]
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = target_pos
        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return self._finished

    def episode_callback(self):
        self._current_wp_idx = 0
        self._finished = False
        self._extension_done = False
        # Generate waypoints for each episode
        # The first compute_control will generate based on obs