import numpy as np

class RoboticArm:
    def __init__(self, link_lengths=(100, 80, 60)):
        self.link_lengths = link_lengths
        self.joint_angles = [np.pi / 4] * len(link_lengths)

    def get_joint_positions(self):
        """Compute forward kinematics and return joint positions."""
        x, y, theta = 0, 0, 0
        positions = [(x, y)]
        for i in range(len(self.link_lengths)):
            theta += self.joint_angles[i]
            x += self.link_lengths[i] * np.cos(theta)
            y += self.link_lengths[i] * np.sin(theta)
            positions.append((x, y))
        return positions

    def inverse_kinematics(self, target_x, target_y, iterations=100, lr=0.01):
        """Basic gradient descent IK solver for N-joint arm."""
        for _ in range(iterations):
            joints = self.get_joint_positions()
            end_effector = joints[-1]
            dx = target_x - end_effector[0]
            dy = target_y - end_effector[1]
            error = np.hypot(dx, dy)

            if error < 1:
                break

            for i in reversed(range(len(self.joint_angles))):
                joint = joints[i]
                effector = joints[-1]

                r_dx = effector[0] - joint[0]
                r_dy = effector[1] - joint[1]

                perp = np.array([-r_dy, r_dx])
                grad = (dx * perp[0] + dy * perp[1]) / (np.linalg.norm([r_dx, r_dy]) + 1e-8)

                self.joint_angles[i] += lr * grad
                self.joint_angles[i] = np.clip(self.joint_angles[i], -np.pi, np.pi)
