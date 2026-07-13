# features/ear.py
"""
محاسبه Eye Aspect Ratio (EAR) برای تشخیص بسته شدن چشم
و تحلیل تاریخچه برای تشخیص خواب‌آلودگی
"""
import numpy as np
import time
from typing import List, Optional, Tuple,Dict
def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    محاسبه Eye Aspect Ratio برای یک چشم
    فرمول: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Args:
        eye_landmarks: 6 نقطه کلیدی چشم (به ترتیب دور چشم)
    Returns:
        مقدار EAR (0 = بسته, ~0.25-0.30 = باز معمولی)
    """
    if eye_landmarks is None or len(eye_landmarks) < 6:
        return 0.28  # مقدار پیش‌فرض
    # نقاط کلیدی
    # p1: گوشه بیرونی چشم, p2: بالا, p3: بالا-داخلی
    # p4: گوشه داخلی, p5: پایین-داخلی, p6: پایین
    p1 = eye_landmarks[0]
    p2 = eye_landmarks[1]
    p3 = eye_landmarks[2]
    p4 = eye_landmarks[3]
    p5 = eye_landmarks[4]
    p6 = eye_landmarks[5]
    # محاسبه فواصل عمودی
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    # محاسبه فاصله افقی
    horizontal = np.linalg.norm(p1 - p4)
    
    if horizontal == 0:
        return 0.28
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    # محدود کردن به محدوده منطقی
    return max(0.05, min(ear, 0.40))


def calculate_average_ear(left_eye_landmarks: np.ndarray, 
                          right_eye_landmarks: np.ndarray) -> float:
    """
    محاسبه میانگین EAR هر دو چشم
    
    Args:
        left_eye_landmarks: 6 نقطه چشم چپ
        right_eye_landmarks: 6 نقطه چشم راست
        
    Returns:
        میانگین EAR دو چشم
    """
    left_ear = calculate_ear(left_eye_landmarks)
    right_ear = calculate_ear(right_eye_landmarks)
    return (left_ear + right_ear) / 2.0


class EyeAspectRatioAnalyzer:
    """
    تحلیل‌گر EAR با حفظ تاریخچه و تشخیص الگوهای پلک زدن و خواب‌آلودگی
    """
    def __init__(self, 
                 history_size: int = 30, 
                 ear_threshold: float = 0.25,
                 fps:int=30,
                #  drowsy_frames_threshold: int = 300,
                 drowsy_duration_sec:float=5.0,
):
        """
        Args:
            history_size: تعداد فریم‌هایی که نگهداری می‌شوند
            ear_threshold: آستانه تشخیص چشم بسته
            drowsy_frames_threshold: تعداد فریم‌های متوالی چشم بسته برای تشخیص خواب‌آلودگی
        """
        self.history_size = history_size
        self.ear_threshold = ear_threshold
        self.fps=fps
        #مدت زمان لازم برای خواب آلودگی
        self.drowsy_duration_sec=drowsy_duration_sec
        # hysteresis thresholds
        self.close_threshold = self.ear_threshold-0.01
        self.open_threshold = self.ear_threshold+0.01
        #state
        self.eye_closed = False
        #time-based drowsiness
        self.closed_start_time=None

        # تاریخچه
        self.ear_history: List[float] = []
        self.closed_history=[]
        self.closed_frames = 0
        self.blink_count = 0
        self.total_frames = 0
        # آمار
        self.last_blink_frame = 0
        self.blink_rate = 0.0
    def update(self, ear_value: float) -> dict:
        """
        به‌روزرسانی تحلیل‌گر با مقدار جدید EAR
        Args:
            ear_value: مقدار EAR اندازه‌گیری شده
        Returns:
            دیکشنری شامل وضعیت‌های مختلف
        """
        current_time=time.time()
        # اضافه کردن به تاریخچه
        self.ear_history.append(ear_value)
        if len(self.ear_history) > self.history_size:
            self.ear_history.pop(0)
        self.total_frames += 1
        # تشخیص بسته بودن چشم
        # is_closed = ear_value < self.ear_threshold
        # hysteresis eye state detection
        # ------------------------- # hysteresis eye detection # -------------------------
        if not self.eye_closed and ear_value < self.close_threshold:
            self.eye_closed = True
        elif self.eye_closed and ear_value > self.open_threshold:
            self.eye_closed = False
        is_closed = self.eye_closed
        self.closed_history.append(1 if is_closed else 0)
        #نگهداری فقط 60 ثانیه اخیر 
        max_history=self.fps*60
        if len(self.closed_history)>max_history:
            self.closed_history.pop(0)
        #محاسبه PERCLOS
        perclos=np.mean(self.closed_history)*100
        # ------------------------- # blink detection # -------------------------
        if is_closed:
            self.closed_frames += 1
            if self.closed_start_time is None:
                self.closed_start_time=current_time
        else:
            if self.closed_frames > 0:
                if 2 <= self.closed_frames <= 8:
                    self.blink_count += 1
                    self.last_blink_frame = self.total_frames
            self.closed_frames = 0
            self.closed_start_time=None
        closed_duration=0.0
        if self.closed_start_time is not None:
            closed_duration=current_time-self.closed_start_time
        is_drowsy=closed_duration>=self.drowsy_duration_sec
        # تشخیص خواب‌آلودگی (چشم بسته برای مدت طولانی)
        # is_drowsy = self.closed_frames > self.drowsy_frames_threshold
        # محاسبه نرخ پلک زدن (پلک در دقیقه)
        if self.total_frames > 30:
            # تخمین بر اساس 30 فریم اخیر (≈ 1 ثانیه)
            self.blink_rate = self.blink_count / (self.total_frames / self.fps) * 60
        # میانگین EAR اخیر
        avg_ear = np.mean(self.ear_history[-10:]) if len(self.ear_history) >= 10 else ear_value
        return {
            'ear': ear_value,
            'average_ear': avg_ear,
            'is_closed': is_closed,
            'is_drowsy': is_drowsy,
            'closed_frames': self.closed_frames,
            'closed_duration':closed_duration,
            'blink_count': self.blink_count,
            'blink_rate': self.blink_rate,
            'perclos':perclos,
            
        }
    def get_ear_history(self) -> List[float]:
        """دریافت تاریخچه EAR"""
        return self.ear_history.copy()
    def reset(self):
        """بازنشانی تمام آمار"""
        self.ear_history = []
        self.closed_frames = 0
        self.blink_count = 0
        self.total_frames = 0
        self.blink_rate = 0.0
    def set_threshold(self, new_threshold: float):
        """تنظیم آستانه جدید (برای کالیبراسیون شخصی)"""
        self.ear_threshold = new_threshold
    def get_statistics(self) -> dict:
        """دریافت آمار کلی"""
        return {
            'total_frames': self.total_frames,
            'blink_count': self.blink_count,
            'blink_rate': self.blink_rate,
            'average_ear': np.mean(self.ear_history) if self.ear_history else 0,
            'min_ear': min(self.ear_history) if self.ear_history else 0,
            'max_ear': max(self.ear_history) if self.ear_history else 0
        }