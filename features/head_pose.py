# features/head_pose.py
"""
تشخیص وضعیت سر (Head Pose) شامل زوایای Yaw, Pitch, Roll
برای تشخیص خم شدن سر (خواب‌آلودگی) و چرخش سر (حواس‌پرتی)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


def calculate_head_pose(face_landmarks: np.ndarray, 
                        frame_width: int = 640,
                        frame_height: int = 480) -> Dict[str, float]:
    """
    محاسبه زوایای سر از روی نقاط کلیدی صورت
    
    Args:
        face_landmarks: نقاط کلیدی صورت (حداقل نقاط بینی و چشم‌ها)
        frame_width: عرض تصویر
        frame_height: ارتفاع تصویر
        
    Returns:
        دیکشنری شامل زوایای yaw, pitch, roll
    """
    if face_landmarks is None or len(face_landmarks) < 68:
        return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.0}
    
    # نقاط کلیدی مورد نیاز برای تخمین Head Pose
    # بینی: نقطه 33 (برای تخمین)
    # چشم چپ: نقطه 36، چشم راست: نقطه 45
    try:
        # این ایندکس‌ها بسته به مدل تشخیص باید تنظیم شوند
        # برای MediaPipe 468 نقطه‌ای، ایندکس‌ها متفاوت است
        nose_tip = face_landmarks[1] if len(face_landmarks) > 1 else [frame_width//2, frame_height//2]
        left_eye = face_landmarks[33] if len(face_landmarks) > 33 else [frame_width//3, frame_height//3]
        right_eye = face_landmarks[263] if len(face_landmarks) > 263 else [2*frame_width//3, frame_height//3]
        
        # تبدیل به آرایه numpy
        nose_tip = np.array(nose_tip[:2])
        left_eye = np.array(left_eye[:2])
        right_eye = np.array(right_eye[:2])
        
        # محاسبه Yaw (چرخش چپ/راست)
        eye_center = (left_eye + right_eye) / 2
        yaw = (nose_tip[0] - eye_center[0]) / (frame_width / 2) * 30  # حداکثر ±30 درجه
        yaw = max(-45, min(45, yaw))  # محدود کردن
        
        # محاسبه Roll (کج کردن سر)
        eye_delta = right_eye - left_eye
        roll = np.arctan2(eye_delta[1], eye_delta[0]) * 180 / np.pi
        roll = max(-30, min(30, roll))  # محدود کردن
        
        # محاسبه Pitch (بالا/پایین)
        nose_to_eye_center = nose_tip - eye_center
        pitch = nose_to_eye_center[1] / (frame_height / 2) * 20
        pitch = max(-30, min(30, pitch))  # محدود کردن
        
        # اعتماد تخمین
        confidence = 0.8
        
    except Exception as e:
        yaw, pitch, roll = 0.0, 0.0, 0.0
        confidence = 0.0
    
    return {
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
        'confidence': confidence
    }


class HeadPoseEstimator:
    """
    تخمین و تحلیل وضعیت سر (Head Pose)
    
    کاربردها:
    - تشخیص خم شدن سر (خواب‌آلودگی)
    - تشخیص چرخش سر (حواس‌پرتی)
    - تشخیص نگاه به موبایل یا بیرون
    """
    
    def __init__(self, 
                 history_size: int = 30,
                 drowsy_pitch_threshold: float = 15.0,   # درجه خم شدن برای تشخیص خواب
                 distracted_yaw_threshold: float = 25.0): # درجه چرخش برای تشخیص حواس‌پرتی
        """
        Args:
            history_size: تعداد فریم‌هایی که نگهداری می‌شوند
            drowsy_pitch_threshold: آستانه خم شدن سر (درجه) برای تشخیص خواب
            distracted_yaw_threshold: آستانه چرخش سر (درجه) برای تشخیص حواس‌پرتی
        """
        self.history_size = history_size
        self.drowsy_pitch_threshold = drowsy_pitch_threshold
        self.distracted_yaw_threshold = distracted_yaw_threshold
        
        # تاریخچه زوایا
        self.yaw_history: List[float] = []
        self.pitch_history: List[float] = []
        self.roll_history: List[float] = []
        
        # آمار
        self.total_frames = 0
        self.drowsy_frames = 0
        self.distracted_frames = 0
        
        # وضعیت فعلی
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
    
    def estimate(self, face_landmarks: np.ndarray,
                 frame_width: int = 640,
                 frame_height: int = 480) -> Dict[str, float]:
        """
        تخمین وضعیت سر از روی نقاط کلیدی
        
        Args:
            face_landmarks: نقاط کلیدی صورت
            frame_width: عرض تصویر
            frame_height: ارتفاع تصویر
            
        Returns:
            دیکشنری شامل زوایا و وضعیت‌ها
        """
        # محاسبه زوایا
        pose = calculate_head_pose(face_landmarks, frame_width, frame_height)
        
        self.current_yaw = pose['yaw']
        self.current_pitch = pose['pitch']
        self.current_roll = pose['roll']
        
        # اضافه کردن به تاریخچه
        self.yaw_history.append(self.current_yaw)
        self.pitch_history.append(self.current_pitch)
        self.roll_history.append(self.current_roll)
        
        if len(self.yaw_history) > self.history_size:
            self.yaw_history.pop(0)
            self.pitch_history.pop(0)
            self.roll_history.pop(0)
        
        self.total_frames += 1
        
        # تشخیص خواب‌آلودگی (سر خم شده به پایین)
        is_head_drooped = abs(self.current_pitch) > self.drowsy_pitch_threshold
        if is_head_drooped:
            self.drowsy_frames += 1
        else:
            self.drowsy_frames = 0
        
        # تشخیص حواس‌پرتی (سر چرخیده به چپ/راست)
        is_distracted = abs(self.current_yaw) > self.distracted_yaw_threshold
        if is_distracted:
            self.distracted_frames += 1
        else:
            self.distracted_frames = 0
        
        # میانگین زوایای اخیر (برای صاف کردن نویز)
        avg_yaw = np.mean(self.yaw_history[-10:]) if len(self.yaw_history) >= 10 else self.current_yaw
        avg_pitch = np.mean(self.pitch_history[-10:]) if len(self.pitch_history) >= 10 else self.current_pitch
        
        # تشخیص خمیازه از روی Head Pose (باز شدن دهان در MAR انجام می‌شود)
        # اینجا فقط وضعیت سر را برمی‌گردانیم
        
        return {
            'yaw': self.current_yaw,
            'pitch': self.current_pitch,
            'roll': self.current_roll,
            'avg_yaw': avg_yaw,
            'avg_pitch': avg_pitch,
            'is_head_drooped': is_head_drooped,
            'is_distracted': is_distracted,
            'drowsy_frames': self.drowsy_frames,
            'confidence': pose['confidence']
        }
    
    def get_current_pose(self) -> Dict[str, float]:
        """دریافت وضعیت سر فعلی"""
        return {
            'yaw': self.current_yaw,
            'pitch': self.current_pitch,
            'roll': self.current_roll
        }
    
    def is_drowsy_from_head_pose(self, consecutive_frames: int = 10) -> bool:
        """
        تشخیص خواب‌آلودگی از روی وضعیت سر
        
        Args:
            consecutive_frames: تعداد فریم‌های متوالی مورد نیاز
            
        Returns:
            True اگر سر به مدت کافی خم شده باشد
        """
        return self.drowsy_frames >= consecutive_frames
    
    def is_distracted_from_head_pose(self, consecutive_frames: int = 5) -> bool:
        """
        تشخیص حواس‌پرتی از روی وضعیت سر
        
        Returns:
            True اگر سر به مدت کافی چرخیده باشد
        """
        return self.distracted_frames >= consecutive_frames
    
    def reset(self):
        """بازنشانی تمام آمار"""
        self.yaw_history = []
        self.pitch_history = []
        self.roll_history = []
        self.total_frames = 0
        self.drowsy_frames = 0
        self.distracted_frames = 0
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
    
    def get_statistics(self) -> Dict:
        """دریافت آمار کلی"""
        return {
            'total_frames': self.total_frames,
            'avg_yaw': np.mean(self.yaw_history) if self.yaw_history else 0,
            'avg_pitch': np.mean(self.pitch_history) if self.pitch_history else 0,
            'max_yaw': max(self.yaw_history) if self.yaw_history else 0,
            'max_pitch': max(self.pitch_history) if self.pitch_history else 0,
            'drowsy_frames_total': self.drowsy_frames,
            'distracted_frames_total': self.distracted_frames
        }


def is_head_drooping(pitch_angle: float, threshold: float = 15.0) -> bool:
    """
    بررسی آیا سر خم شده است (نشانه خواب‌آلودگی)
    
    Args:
        pitch_angle: زاویه pitch (درجه، مثبت = پایین)
        threshold: آستانه خمیدگی
        
    Returns:
        True اگر سر خم شده باشد
    """
    return pitch_angle > threshold


def is_head_turned(yaw_angle: float, threshold: float = 25.0) -> bool:
    """
    بررسی آیا سر چرخیده است (نشانه حواس‌پرتی)
    
    Args:
        yaw_angle: زاویه yaw (درجه، مثبت = راست، منفی = چپ)
        threshold: آستانه چرخش
        
    Returns:
        True اگر سر چرخیده باشد
    """
    return abs(yaw_angle) > threshold