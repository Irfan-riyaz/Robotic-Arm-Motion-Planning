from stable_baselines3 import PPO
from franka_arm_env import RoboticArmEnv

# Initialize the environment with 3 joints
env = RoboticArmEnv(n_joints=3)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")

# Train the model
model.learn(total_timesteps=100000)

# Save the model (this creates franka_arm_rl_agent.zip)
model.save("franka_arm_rl_agent")
print("✅ Model saved successfully as franka_arm_rl_agent.zip")
