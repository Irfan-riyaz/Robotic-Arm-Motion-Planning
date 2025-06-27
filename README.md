# Robotic-Arm-Motion-Planning

# 🤖 Franka Arm Motion Planning with Reinforcement Learning

This project simulates a 2D Franka-style 3-joint robotic arm. It uses Forward and Inverse Kinematics, as well as Reinforcement Learning (PPO) to reach arbitrary targets with smooth motion.

---

## 🎯 Features

- ✅ 3-joint robotic arm simulation (Franka-style)
- ✅ Forward Kinematics (FK) and Gradient Descent Inverse Kinematics (IK)
- ✅ Reinforcement Learning using PPO (Stable-Baselines3)
- ✅ Pygame GUI with click-to-target interface
- ✅ Smooth target reaching without trembling
- ✅ Real-time visualization of joint positions and trajectory

---

## 🧠 Technologies Used

- Python 3.8+
- Pygame (for visualization)
- NumPy
- Gymnasium (custom RL environment)
- Stable-Baselines3 (PPO agent)

---

## 📁 Project Structure

franka_arm.py # Arm kinematics (FK & IK)
├── franka_arm_env.py # Gym environment for RL
├── train_agent.py # PPO training script
├── visualizer.py # GUI + trained agent controller
├── requirements.txt # Dependencies
├── README.md # This file

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Train the RL agent

python train_agent.py
This saves the model as: franka_arm_rl_agent.zip

3️⃣ Visualize the trained agent

python visualizer.py
