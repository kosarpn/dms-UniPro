from dataclasses import dataclass
import numpy as np

from core.data_types import (
    EyeData,
    MouthData,
    HeadPoseData
)
@dataclass
class FeatureVector:
    # =========================================
    # Eye Features
    # =========================================
    ear: float
    average_ear: float
    blink_rate: float
    perclos: float
    closed_duration: float
    is_eye_closed: float
    # =========================================
    # Mouth Features
    # =========================================
    mar: float
    average_mar: float
    yawn_duration: float
    yawn_count: int
    # =========================================
    # Head Pose Features
    # =========================================
    pitch: float
    yaw: float
    roll: float
    is_head_down: float
    head_down_frames: int
    is_head_left: float
    is_head_right: float
    # =========================================
    # Convert to numpy
    # =========================================
    def to_numpy(self) -> np.ndarray:
        return np.array([
            # Eye
            self.ear,
            self.average_ear,
            self.blink_rate,
            self.perclos,
            self.closed_duration,
            self.is_eye_closed,
            # Mouth
            self.mar,
            self.average_mar,
            self.yawn_duration,
            self.yawn_count,
            # Head Pose
            self.pitch,
            self.yaw,
            self.roll,
            self.is_head_down,
            self.head_down_frames,
            self.is_head_left,
            self.is_head_right,

        ], dtype=np.float32)
def build_feature_vector(
        eye_data: EyeData,
        mouth_data: MouthData,
        head_pose_data: HeadPoseData
) -> FeatureVector:
    """
    ساخت feature vector از DTO ها
    """
    return FeatureVector(
        # =====================
        # Eye
        # =====================
        ear=eye_data.ear,
        average_ear=eye_data.average_ear,
        blink_rate=eye_data.blink_rate,
        perclos=eye_data.perclos,
        closed_duration=eye_data.closed_duration,
        is_eye_closed=float(eye_data.is_closed),
        # =====================
        # Mouth
        # =====================
        mar=mouth_data.mar,
        average_mar=mouth_data.average_mar,
        yawn_duration=mouth_data.yawn_duration,
        yawn_count=mouth_data.yawn_count,
        # =====================
        # Head Pose
        # =====================
        pitch=head_pose_data.pitch,
        yaw=head_pose_data.yaw,
        roll=head_pose_data.roll,

        is_head_down=float(head_pose_data.is_head_down),
        head_down_frames=head_pose_data.head_down_frames,

        is_head_left=float(head_pose_data.is_head_left),
        is_head_right=float(head_pose_data.is_head_right),
    )