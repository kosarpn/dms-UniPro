
from enum import Enum
from dataclasses import dataclass
# ============================================================================
# Enums
# ============================================================================
class HeadPoseStatus(Enum):
    """وضعیت سر"""
    NORMAL = "normal"          # وضعیت عادی
    NODDING = "nodding"        # سر تکان دادن (خواب‌آلودگی)
    TILTED = "tilted"          # کج شدن سر
    DROOPING = "drooping"      # افتادن سر (خواب‌آلودگی شدید)
    TURNED = "turned"          # چرخیدن سر به طرفین
    LOOKING_DOWN = "looking_down"  # نگاه به پایین
class BlinkPattern(Enum):
    """الگوی پلک زدن"""
    NORMAL = "normal"          # پلک زدن عادی (15-20 بار در دقیقه)
    RAPID = "rapid"            # پلک زدن سریع (استرس، خستگی)
    SLOW = "slow"              # پلک زدن آهسته (خواب‌آلودگی)
    PROLONGED = "prolonged"    # بسته شدن طولانی چشم (خواب‌آلودگی شدید)
    MICRO_SLEEP = "micro_sleep" # میکروخواب (بسیار خطرناک)
class DrowsinessLevel(Enum):
    """سطح خواب‌آلودگی"""
    ALERT = "alert"            # هوشیار (0-30%)
    LIGHT = "light_drowsy"     # خواب‌آلودگی خفیف (30-50%)
    MODERATE = "moderate"      # خواب‌آلودگی متوسط (50-70%)
    SEVERE = "severe"          # خواب‌آلودگی شدید (70-90%)
    DROWSY = "drowsy"      # بحرانی (90-100%)
# ============================================================================
# Data Classes (DTOs)
# ============================================================================
@dataclass
class HeadPoseData:
    yaw: float
    pitch: float
    roll: float

    is_head_down: bool = False
    head_down_frames: int = 0

    is_head_left: bool = False
    is_head_right: bool = False

    confidence: float = 1.0

@dataclass
class EyeData:
    ear:float
    average_ear:float
    is_closed:bool=False
    blink_count:int=0
    blink_rate:float=0.0
    closed_duration:float=0.0
    perclos:float=0.0
@dataclass
class MouthData:
    mar: float
    average_mar: float
    max_mar: float
    mouth_width: float
    mouth_height: float
    is_yawning: bool
    yawn_detected: bool
    yawn_duration: float
    yawn_count: int
    speaking_detected: bool
    smile_detected: bool
@dataclass
class Prediction:
    """
    خروجی مشترک تمام Classifierها
    """
    is_drowsy:bool
    confidence:float
    level:DrowsinessLevel
    classifier_name: str
    reason:str=""
    inference_time_ms: float = 0.0



