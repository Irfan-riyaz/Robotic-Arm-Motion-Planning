import numpy as np

class RoboticArm:
    def __init__(self, link_lengths=(100, 80)):
        self.l1, self.l2 = link_lengths
        self.theta1 = np.pi / 4  # shoulder angle
        self.theta2 = np.pi / 4  # elbow angle

    def get_joint_positions(self):
        # Shoulder to elbow
        x1 = self.l1 * np.cos(self.theta1)
        y1 = self.l1 * np.sin(self.theta1)

        # Elbow to fingertip (hand)
        x2 = x1 + self.l2 * np.cos(self.theta1 + self.theta2)
        y2 = y1 + self.l2 * np.sin(self.theta1 + self.theta2)

        return [(0, 0), (x1, y1), (x2, y2)]

    def inverse_kinematics(self, target_x, target_y):
        x, y = target_x, target_y
        l1, l2 = self.l1, self.l2
        dist = np.hypot(x, y)

        if dist > (l1 + l2):
            print("Target is out of reach.")
            return

        # Law of cosines
        cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2 = np.arccos(cos_theta2)

        # Law of sines
        k1 = l1 + l2 * np.cos(theta2)
        k2 = l2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        self.theta1 = theta1
        self.theta2 = theta2
