import gymnasium as gym
from gymnasium import spaces
import numpy as np
from franka_arm import RoboticArm

class RoboticArmEnv(gym.Env):
    def __init__(self, n_joints=3, render_mode=None):
        super().__init__()
        self.arm = RoboticArm(link_lengths=[100] * n_joints)
        self.n_joints = n_joints
        self.render_mode = render_mode
        self.target = self._sample_target()
        self._target_reached = False

        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi] * n_joints + [-200, -200], dtype=np.float32),
            high=np.array([np.pi] * n_joints + [200, 200], dtype=np.float32),
            dtype=np.float32
        )

    def _sample_target(self):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(50, sum(self.arm.link_lengths))
        return radius * np.cos(angle), radius * np.sin(angle)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.arm.joint_angles = [0.0] * self.n_joints
        self.target = self._sample_target()
        self._target_reached = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.arm.joint_angles + list(self.target), dtype=np.float32)

    def step(self, action):
        if self._target_reached:
            # Freeze arm when target reached
            return self._get_obs(), 0.0, True, False, {}

        for i in range(self.n_joints):
            self.arm.joint_angles[i] += float(action[i])
            self.arm.joint_angles[i] = np.clip(self.arm.joint_angles[i], -np.pi, np.pi)

        joints = self.arm.get_joint_positions()
        end = joints[-1]
        dx = self.target[0] - end[0]
        dy = self.target[1] - end[1]
        dist = np.hypot(dx, dy)

        reward = -dist
        done = dist < 10
        if done:
            reward += 10
            self._target_reached = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        if self.render_mode == "human":
            print(f"Joint angles: {self.arm.joint_angles}")
