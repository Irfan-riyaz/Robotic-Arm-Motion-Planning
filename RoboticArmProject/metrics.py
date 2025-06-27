import numpy as np

def evaluate_agent(model, env, episodes=20):
    successes = 0
    total_distance = 0
    total_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            joints = env.arm.get_joint_positions()
            end = joints[-1]
            dist = np.linalg.norm([end[0] - env.target[0], end[1] - env.target[1]])
            total_distance += dist
            steps += 1
        if done:
            successes += 1
        total_steps += steps

    print(f"\nEvaluation Results:")
    print(f"✅ Success Rate: {successes / episodes * 100:.2f}%")
    print(f"📏 Avg Distance: {total_distance / episodes:.2f}")
    print(f"⏱️ Avg Steps: {total_steps / episodes:.2f}")
