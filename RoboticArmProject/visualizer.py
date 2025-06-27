import pygame
import numpy as np
from stable_baselines3 import PPO
from franka_arm_env import RoboticArmEnv

WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Franka Arm RL Visualizer")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

def to_screen(pos):
    x, y = pos
    return int(x + WIDTH // 2), int(HEIGHT // 2 - y)

def from_screen(x, y):
    return x - WIDTH // 2, HEIGHT // 2 - y

env = RoboticArmEnv(n_joints=3)
model = PPO.load("franka_arm_rl_agent")
obs, _ = env.reset()
trajectory = []

running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            tx, ty = from_screen(mx, my)
            env.target = (tx, ty)
            obs, _ = env.reset()
            trajectory.clear()

    # Only act if target not reached
    if not env._target_reached:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

    joints = env.arm.get_joint_positions()
    screen_joints = [to_screen(j) for j in joints]
    trajectory.append(screen_joints[-1])

    # Draw arm
    for i in range(len(screen_joints) - 1):
        pygame.draw.line(screen, (0, 255, 0), screen_joints[i], screen_joints[i + 1], 5)
        pygame.draw.circle(screen, (255, 0, 0), screen_joints[i], 6)
    pygame.draw.circle(screen, (0, 0, 255), screen_joints[-1], 8)

    # Draw path
    if len(trajectory) > 1:
        pygame.draw.lines(screen, (0, 100, 255), False, trajectory, 2)

    # Draw target
    tx, ty = env.target
    pygame.draw.circle(screen, (255, 255, 0), to_screen((tx, ty)), 7)
    screen.blit(font.render(f"Target: ({int(tx)}, {int(ty)})", True, (255, 255, 255)), (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
