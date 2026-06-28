
# utils/kalman_filter.py
"""
فیلتر کالمن ساده برای صاف کردن EAR و MAR
نسخه اولیه - قابل ارتقاء بعداً
"""

class AdaptiveKalmanFilter:
    """
    نسخه ساده فیلتر کالمن برای صاف کردن داده‌ها
    """
    
    def __init__(self):
        # مقدار اولیه
        self.last_value = 0.3
        self.smoothing_factor = 0.7  # هرچه کمتر، صاف‌تر
    
    def update_ear(self, ear_value: float) -> float:
        """صاف کردن مقدار EAR"""
        # میانگین وزنی بین مقدار جدید و قبلی
        filtered = self.smoothing_factor * ear_value + (1 - self.smoothing_factor) * self.last_value
        self.last_value = filtered
        return filtered
    
    def update_mar(self, mar_value: float) -> float:
        """صاف کردن مقدار MAR"""
        filtered = self.smoothing_factor * mar_value + (1 - self.smoothing_factor) * self.last_value
        self.last_value = filtered
        return filtered
    