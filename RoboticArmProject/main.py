import pygame
from robot_arm import RoboticArm
import numpy as np

# Pygame setup
WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robotic Arm with Accurate Hand Motion")
clock = pygame.time.Clock()

# Helper functions
def to_screen(pos):
    x, y = pos
    return int(x + WIDTH // 2), int(HEIGHT // 2 - y)

def from_screen(x, y):
    return x - WIDTH // 2, HEIGHT // 2 - y

# Robotic Arm instance
arm = RoboticArm()
target = None

running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            tx, ty = from_screen(mx, my)
            target = (tx, ty)
            arm.inverse_kinematics(tx, ty)

    # Get joint positions: shoulder → elbow → hand
    joints = arm.get_joint_positions()
    screen_joints = [to_screen(j) for j in joints]

    # Draw arm links and joints
    for i in range(len(screen_joints) - 1):
        pygame.draw.line(screen, (0, 255, 0), screen_joints[i], screen_joints[i + 1], 5)
        pygame.draw.circle(screen, (255, 0, 0), screen_joints[i], 6)

    # Draw fingertip (hand)
    pygame.draw.circle(screen, (0, 0, 255), screen_joints[-1], 8)

    # Draw target
    if target:
        pygame.draw.circle(screen, (255, 255, 0), to_screen(target), 6)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
