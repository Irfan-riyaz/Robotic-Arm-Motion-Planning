# test_agent.py
from stable_baselines3 import PPO
from robot_arm_env import RoboticArmEnv
from metrics import evaluate_agent
import time

env = RoboticArmEnv()
model = PPO.load("robotic_arm_rl_agent")

evaluate_agent(model, env)

# Optional: render agent trying to reach new targets
for _ in range(10):
    obs, _ = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.05)
        if done:
            break
