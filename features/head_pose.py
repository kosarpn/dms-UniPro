import cv2
import numpy as np
from collections import deque

class HeadPoseEstimator:
    def __init__(self):
        self.pitch_history = deque(maxlen=10)
        self.yaw_history = deque(maxlen=10)
        self.roll_history = deque(maxlen=10)
        self.head_down_counter = 0
        self.head_left_counter = 0
        self.head_right_counter = 0
    def estimate(self,landmarks,image_shape):
        h,w=image_shape[:2]
        image_points = np.array([
        landmarks[1][:2],      # nose
        landmarks[152][:2],    # chin
        landmarks[33][:2],     # left eye
        landmarks[263][:2],    # right eye
        landmarks[61][:2],     # left mouth
        landmarks[291][:2],    # right mouth
    ], dtype=np.float64)
        model_points = np.array([
        (0.0,0.0,0.0),
        (0.0,-63.6,-12.5),
        (-43.3,32.7,-26),
        (43.3,32.7,-26),
        (-28.9,-28.9,-24.1),
        (28.9,-28.9,-24.1)
    ])
        focal_length = w
        camera_matrix = np.array([
            [focal_length,0,w/2],
            [0,focal_length,h/2],
            [0,0,1]
        ],dtype=np.float64)
        dist_coeffs=np.zeros((4,1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(
            rotation_matrix[0,0] ** 2 +
            rotation_matrix[1,0] ** 2
        )

        singular = sy < 1e-6


        if not singular:

            pitch = np.arctan2(
                rotation_matrix[2,1],
                rotation_matrix[2,2]
            )

            yaw = np.arctan2(
                -rotation_matrix[2,0],
                sy
            )

            roll = np.arctan2(
                rotation_matrix[1,0],
                rotation_matrix[0,0]
            )

        else:

            pitch = np.arctan2(
                -rotation_matrix[1,2],
                rotation_matrix[1,1]
            )

            yaw = np.arctan2(
                -rotation_matrix[2,0],
                sy
            )

            roll = 0
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.roll_history.append(roll)
        pitch = np.mean(self.pitch_history)
        yaw = np.mean(self.yaw_history)
        roll = np.mean(self.roll_history)
        HEAD_DOWN = 18
        HEAD_LEFT = -25
        HEAD_RIGHT = 25
        head_down = False
        head_left = False
        head_right = False
        if pitch > HEAD_DOWN:
            self.head_down_counter+=1
        else:
            self.head_down_counter=0
        if yaw < HEAD_LEFT:
            self.head_left_counter += 1
        else:
            self.head_left_counter = 0
        if yaw > HEAD_RIGHT:
            self.head_right_counter += 1
        else:
            self.head_right_counter = 0
        head_down = self.head_down_counter >= 15
        head_left = self.head_left_counter >= 15
        head_right = self.head_right_counter >= 15
        print(
            f"Pitch={pitch:.1f}  "
            f"Yaw={yaw:.1f}  "
            f"Roll={roll:.1f}  "
            f"Down={self.head_down_counter}"
        )
        return {
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
        "head_down": head_down,
        "head_left": head_left,
        "head_right": head_right,
        "head_down_frames": self.head_down_counter,
        "head_left_frames": self.head_left_counter,
        "head_right_frames": self.head_right_counter,
    }
            
