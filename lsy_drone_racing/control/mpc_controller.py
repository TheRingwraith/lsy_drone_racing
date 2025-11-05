from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryGenerator:
    """Generates smooth trajectories using cubic splines with distance-based timing
    and clamped boundary conditions to reduce end wiggle.
    """

    def __init__(self, waypoints: NDArray[np.floating], total_time: float = 15.0):
        """
        Args:
            waypoints: Array of shape (N, 3) with [x, y, z] waypoints.
            total_time: Duration to traverse the trajectory.
        """
        self._t_total = total_time
        self._waypoints = waypoints.copy()
        self._times = self._compute_times(self._waypoints)
        self._spline = self._build_spline()
        self._updated = set()

    def _compute_times(self, waypoints: NDArray[np.floating]) -> NDArray[np.floating]:
        """Assign time stamps proportional to distance between waypoints."""
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cumdist = np.concatenate(([0.0], np.cumsum(distances)))
        times = cumdist / cumdist[-1] * self._t_total
        return times

    def _build_spline(self) -> CubicSpline:
        # Use clamped boundary conditions to reduce overshoot at ends
        return CubicSpline(self._times, self._waypoints, bc_type="clamped")

    def evaluate(self, t: float) -> NDArray[np.floating]:
        """Evaluate trajectory at time t."""
        t = np.clip(t, 0, self._t_total)
        return self._spline(t)

    def update_detection(self, idx: int, detected_pos: NDArray[np.floating]):
        """Update a waypoint when a gate/obstacle is detected."""
        if idx in self._updated:
            return
        self._waypoints[idx] = detected_pos
        self._times = self._compute_times(self._waypoints)
        self._spline = self._build_spline()
        self._updated.add(idx)

    def reset_to_nominal(self, waypoints: NDArray[np.floating]):
        """Reset to nominal trajectory (e.g. at episode reset)."""
        self._waypoints = waypoints.copy()
        self._times = self._compute_times(self._waypoints)
        self._spline = self._build_spline()
        self._updated.clear()


class StateController(Controller):
    """State controller following a generated trajectory with adaptive updates."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Example nominal waypoints (replace with gates/obstacles from config if available)
        nominal_waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )

        self._nominal_waypoints = nominal_waypoints
        self._trajectory = TrajectoryGenerator(nominal_waypoints, total_time=15.0)

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        t = min(self._tick / self._freq, self._trajectory._t_total)
        if t >= self._trajectory._t_total:
            self._finished = True

        # Update trajectory if gates/obstacles are detected
        if "gates" in obs:
            for idx, gate in enumerate(obs["gates"]):
                if gate.get("visible", False):
                    self._trajectory.update_detection(idx, np.array(gate["pos"]))

        if "obstacles" in obs:
            for idx, obs_obj in enumerate(obs["obstacles"]):
                if obs_obj.get("visible", False):
                    self._trajectory.update_detection(idx, np.array(obs_obj["pos"]))

        des_pos = self._trajectory.evaluate(t)
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._trajectory.reset_to_nominal(self._nominal_waypoints)
        self._finished = False
