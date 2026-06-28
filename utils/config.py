# utils/config.py
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """Configuration for DMS system"""
    
    # Mode
    MODE: str = "production"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "dms.log"
    
    # Camera
    CAMERA_ID: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    CAMERA_FPS: int = 30
    
    # Detection thresholds
    MEDIAPIPE_CONFIDENCE: float = 0.5
    YOLO_CONFIDENCE: float = 0.5
    FALLBACK_THRESHOLD: int = 5
    
    # Drowsiness thresholds
    EAR_THRESHOLD: float = 0.25
    MAR_THRESHOLD: float = 0.6
    EAR_HISTORY_SIZE: int = 30
    
    # Driver selection weights
    DISTANCE_WEIGHT: float = 0.5
    POSITION_WEIGHT: float = 0.3
    TRACKING_WEIGHT: float = 0.2
    STEERING_SIDE: str = "left"
    
    # Calibration
    CALIBRATION_DURATION: float = 5.0
    REFERENCE_DISTANCE: float = 50.0
    
    # Depth estimation
    ENABLE_DEPTH_ESTIMATION: bool = False
    DEPTH_MODEL_TYPE: str = "MiDaS_small"
    
    # Alerts
    AUDIO_ALERTS: bool = True
    VISUAL_ALERTS: bool = True
    ALERT_COOLDOWN: float = 2.0
    
    # Performance
    DROWSY_FRAME_THRESHOLD: int = 10


# Default configuration instance
default_config = Config()