# features/mar.py
"""
محاسبه Mouth Aspect Ratio (MAR) برای تشخیص خمیازه
و تحلیل تاریخچه برای تشخیص الگوهای خستگی
"""

import numpy as np
from typing import List, Optional, Dict


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    """
    محاسبه Mouth Aspect Ratio برای تشخیص باز بودن دهان
    فرمول: MAR = ارتفاع دهان / عرض دهان
    Args:
        mouth_landmarks: نقاط کلیدی دهان (حداقل 6 نقطه)
    Returns:
        مقدار MAR (0.1-0.2 = بسته, 0.6+ = خمیازه)
    """
    # نقاط outer lip (لب بیرونی)
    MOUTH_OUTER = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # بالا (از چپ به راست)
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146  # پایین (از راست به چپ)
    ]

    MOUTH_LANDMARKS = [
    61,   # گوشه چپ (left corner)
    39,   # بالای چپ (top-left)
    0,    # بالای مرکز (top-center)
    269,  # بالای راست (top-right)
    291,  # گوشه راست (right corner)
    17,   # پایین مرکز (bottom-center)
]

    if mouth_landmarks is None or len(mouth_landmarks) < 6:
        return 0.0      
    # جفت‌های صحیح (top ↔ bottom)
    v1 = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[5])  # چپ: top-left ↔ bottom-right
    v2 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[4])  # مرکز: top-center ↔ bottom-center
    
    # عرض (افقی)
    h = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[3])  # left ↔ right
    
    if h < 1e-6:

        return 0.0
    
    mar = (v1 + v2) / (2.0 * h)
    return max(0.05, min(mar, 0.90))



class MouthAspectRatioAnalyzer:
    """
    تحلیل‌گر MAR با حفظ تاریخچه و تشخیص خمیازه
    """
    def __init__(self, 
                 history_size: int = 30,
                 yawn_threshold: float = 0.60,
                 min_yawn_frames: int = 5):
        """
        Args:
            history_size: تعداد فریم‌هایی که نگهداری می‌شوند
            yawn_threshold: آستانه تشخیص خمیازه
            min_yawn_frames: حداقل فریم متوالی برای تأیید خمیازه
        """
        self.history_size = history_size
        self.yawn_threshold = yawn_threshold
        self.min_yawn_frames = min_yawn_frames
        
        # تاریخچه
        self.mar_history: List[float] = []
        self.yawn_frames_count = 0
        self.yawn_total_count = 0
        self.total_frames = 0
        
        # وضعیت فعلی
        self.is_currently_yawning = False
        self.last_yawn_frame = 0
    
    def update(self, mar_value: float) -> Dict:
        """
        به‌روزرسانی تحلیل‌گر با مقدار جدید MAR
        
        Args:
            mar_value: مقدار MAR اندازه‌گیری شده
            
        Returns:
            دیکشنری شامل وضعیت‌های مختلف
        """
        # اضافه کردن به تاریخچه
        self.mar_history.append(mar_value)
        if len(self.mar_history) > self.history_size:
            self.mar_history.pop(0)
        
        self.total_frames += 1
        
        # تشخیص خمیازه
        is_yawn = mar_value > self.yawn_threshold
        
        if is_yawn:
            self.yawn_frames_count += 1
        else:
            # اگر خمیازه تمام شد و تعداد فریم‌ها کافی بود → ثبت خمیازه
            if self.yawn_frames_count >= self.min_yawn_frames:
                self.yawn_total_count += 1
                self.last_yawn_frame = self.total_frames
                self.is_currently_yawning = True
            else:
                self.is_currently_yawning = False
            
            self.yawn_frames_count = 0
        
        # اگر هنوز خمیازه در حال انجام است
        if is_yawn and self.yawn_frames_count > 0:
            self.is_currently_yawning = True
        
        # تشخیص خمیازه اخیر (برای هشدار)
        yawn_detected = False
        if self.last_yawn_frame > 0 and (self.total_frames - self.last_yawn_frame) < 5:
            yawn_detected = True
        
        # میانگین MAR اخیر
        avg_mar = np.mean(self.mar_history[-10:]) if len(self.mar_history) >= 10 else mar_value
        
        # حداکثر MAR اخیر (برای تشخیص اوج خمیازه)
        max_mar = max(self.mar_history[-20:]) if self.mar_history else mar_value
        
        return {
            'mar': mar_value,
            'average_mar': avg_mar,
            'max_mar': max_mar,
            'is_yawning': is_yawn,
            'is_currently_yawning': self.is_currently_yawning,
            'yawn_detected': yawn_detected,
            'yawn_frames': self.yawn_frames_count,
            'yawn_total_count': self.yawn_total_count
        }
    
    def get_mar_history(self) -> List[float]:
        """دریافت تاریخچه MAR"""
        return self.mar_history.copy()
    
    def reset(self):
        """بازنشانی تمام آمار"""
        self.mar_history = []
        self.yawn_frames_count = 0
        self.yawn_total_count = 0
        self.total_frames = 0
        self.is_currently_yawning = False
        self.last_yawn_frame = 0
    
    def set_threshold(self, new_threshold: float):
        """تنظیم آستانه جدید برای تشخیص خمیازه"""
        self.yawn_threshold = new_threshold
    
    def get_statistics(self) -> Dict:
        """دریافت آمار کلی"""
        return {
            'total_frames': self.total_frames,
            'yawn_count': self.yawn_total_count,
            'yawn_rate': self.yawn_total_count / (self.total_frames / 3600) if self.total_frames > 0 else 0,  # خمیازه در ساعت
            'average_mar': np.mean(self.mar_history) if self.mar_history else 0,
            'max_mar': max(self.mar_history) if self.mar_history else 0
        }


class YawnDetector:
    """
    کلاس ساده تشخیص خمیازه (برای استفاده سریع)
    """
    
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self.analyzer = MouthAspectRatioAnalyzer(yawn_threshold=threshold)
    
    def detect(self, mar_value: float) -> bool:
        """
        تشخیص خمیازه
        
        Args:
            mar_value: مقدار MAR محاسبه شده
            
        Returns:
            True اگر خمیازه تشخیص داده شود
        """
        result = self.analyzer.update(mar_value)
        return result['yawn_detected']
    
    def get_yawn_count(self) -> int:
        """دریافت تعداد خمیازه‌های ثبت شده"""
        return self.analyzer.yawn_total_count