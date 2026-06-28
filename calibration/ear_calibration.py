# calibration/ear_calibration.py
"""
کالیبراسیون EAR (Eye Aspect Ratio) برای هر راننده
هدف: تنظیم آستانه خواب‌آلودگی متناسب با ویژگی‌های چشم هر شخص
"""

import numpy as np
from typing import List, Optional


class EARCalibration:
    """
    تنظیم آستانه EAR مخصوص هر راننده
    
    روش کار:
    1. در 5-10 ثانیه اول رانندگی، EAR عادی راننده را اندازه می‌گیریم
    2. آستانه خواب‌آلودگی = 85% مقدار عادی
    3. اگر EAR از آستانه پایین‌تر رفت → راننده خواب‌آلود است
    """
    
    def __init__(self, threshold_ratio: float = 0.85):
        """
        Args:
            threshold_ratio: نسبت آستانه به EAR عادی (پیش‌فرض 85%)
        """
        self.threshold_ratio = threshold_ratio
        self.ear_values: List[float] = []  # تاریخچه EAR در زمان کالیبراسیون
        self.normal_ear: Optional[float] = None
        self.drowsy_threshold: Optional[float] = None
        self.is_calibrated = False
    
    def add_sample(self, ear_value: float):
        """
        اضافه کردن یک نمونه EAR در حین کالیبراسیون
        
        Args:
            ear_value: مقدار EAR اندازه‌گیری شده
        """
        if ear_value > 0 and ear_value < 0.5:  # محدوده معتبر
            self.ear_values.append(ear_value)
    
    def calibrate(self) -> bool:
        """
        نهایی‌سازی کالیبراسیون و محاسبه آستانه
        
        Returns:
            True اگر کالیبراسیون موفق بود
        """
        if len(self.ear_values) < 10:  # حداقل 10 نمونه نیاز داریم
            print(f"[EARCalibration] Not enough samples: {len(self.ear_values)}/10")
            return False
        
        # محاسبه EAR عادی (میانگین 70% آخرین نمونه‌ها - حذف نویز اولیه)
        stable_samples = self.ear_values[-int(len(self.ear_values) * 0.7):]
        self.normal_ear = np.mean(stable_samples)
        
        # محاسبه آستانه خواب‌آلودگی
        self.drowsy_threshold = self.normal_ear * self.threshold_ratio
        
        self.is_calibrated = True
        
        print(f"[EARCalibration] ✅ Calibration complete:")
        print(f"  - Normal EAR: {self.normal_ear:.3f}")
        print(f"  - Drowsy threshold: {self.drowsy_threshold:.3f}")
        
        return True
    
    def get_threshold(self) -> float:
        """
        دریافت آستانه خواب‌آلودگی
        
        Returns:
            آستانه یا مقدار پیش‌فرض 0.25
        """
        if self.is_calibrated and self.drowsy_threshold is not None:
            return self.drowsy_threshold
        return 0.25  # مقدار پیش‌فرض جهانی
    
    def get_normal_ear(self) -> Optional[float]:
        """دریافت EAR عادی راننده"""
        return self.normal_ear
    
    def reset(self):
        """بازنشانی کالیبراسیون"""
        self.ear_values = []
        self.normal_ear = None
        self.drowsy_threshold = None
        self.is_calibrated = False
    
    def is_drowsy(self, current_ear: float) -> bool:
        """
        بررسی آیا مقدار EAR فعلی نشان‌دهنده خواب‌آلودگی است
        
        Args:
            current_ear: مقدار EAR فعلی
            
        Returns:
            True اگر راننده خواب‌آلوده باشد
        """
        if not self.is_calibrated:
            # اگر کالیبره نشده، از آستانه پیش‌فرض استفاده کن
            return current_ear < 0.25
        return current_ear < self.drowsy_threshold