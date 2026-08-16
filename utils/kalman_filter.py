
# utils/kalman_filter.py

class AdaptiveKalmanFilter:
    """ 
     _(البته اینEMA  هست در وافع نه فیلتر کالمن) نسخه ساده فیلتر کالن"""

    def __init__(self):

        self.last_ear = 0.3

        self.last_mar = 0.2

        self.smoothing_factor = 0.7

    def update_ear(self, ear_value: float) -> float:

        filtered = (
            self.smoothing_factor * ear_value
            + (1 - self.smoothing_factor) * self.last_ear
        )

        self.last_ear = filtered

        return filtered

    def update_mar(self, mar_value: float) -> float:

        filtered = (
            self.smoothing_factor * mar_value
            + (1 - self.smoothing_factor) * self.last_mar
        )

        self.last_mar = filtered

        return filtered