
from turtle import done
import gymnasium as gym
from gymnasium import spaces
from robot_arm import RoboticArm
import numpy as np

class RoboticArmEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.arm = RoboticArm()
        self.target = self._sample_target()
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -200, -200]),
            high=np.array([np.pi, np.pi, 200, 200]),
            dtype=np.float32
        )

    def _sample_target(self):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(40, 160)
        return radius * np.cos(angle), radius * np.sin(angle)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.arm.set_angles(0.0, 0.0)
        if not hasattr(self, "target") or self.target is None:
            self.target = self._sample_target()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.arm.theta1,
            self.arm.theta2,
            self.target[0],
            self.target[1]
        ], dtype=np.float32)

    def step(self, action):
        self.arm.theta1 = np.clip(self.arm.theta1 + float(action[0]), -np.pi, np.pi)
        self.arm.theta2 = np.clip(self.arm.theta2 + float(action[1]), -np.pi, np.pi)

        joints = self.arm.get_joint_positions()
        end_effector = joints[-1]
        dx = self.target[0] - end_effector[0]
        dy = self.target[1] - end_effector[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        
        joint_effort = np.sum(np.square(action))   # penalize large joint actions
        elbow_weight = 0.5  # Encourage use of elbow
        reward = -dist - 0.01 * joint_effort + elbow_weight * abs(action[1])


        if done:
            reward += 10

        return self._get_obs(), reward, done, False, {}

    def render(self):
        if self.render_mode == "human":
            print(f"Theta1: {self.arm.theta1:.2f}, Theta2: {self.arm.theta2:.2f}, Target: {self.target}")
