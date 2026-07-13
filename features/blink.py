# features/blink.py
"""
تشخیص Blink Rate و تحلیل الگوی پلک زدن
"""

from collections import deque
from typing import Dict


class BlinkRateAnalyzer:
    """
    تحلیل نرخ پلک زدن برای تشخیص خستگی
    """
    def __init__(
        self,
        fps: int = 30,
        window_seconds: int = 60,
        min_blink_frames: int = 1,
        max_blink_frames: int = 8
    ):
        """
        Args:
            fps: نرخ فریم دوربین
            window_seconds: بازه تحلیل (ثانیه)
            min_blink_frames: حداقل فریم بسته بودن برای blink
            max_blink_frames: حداکثر فریم blink عادی
        """
        self.fps = fps
        self.window_frames = fps * window_seconds
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        # وضعیت چشم
        self.eye_closed = False
        self.closed_counter = 0
        # ذخیره زمان blink ها
        self.blink_timestamps = deque()
        # شمارنده کلی
        self.total_frames = 0

    def update(self, is_eye_closed: bool) -> Dict:
        """
        آپدیت وضعیت چشم

        Args:
            is_eye_closed: آیا چشم بسته است؟

        Returns:
            اطلاعات blink
        """

        self.total_frames += 1

        # اگر چشم بسته است
        if is_eye_closed:
            self.closed_counter += 1
            self.eye_closed = True

        # اگر چشم باز شد
        else:

            # اگر قبلا بسته بود → احتمال blink
            if self.eye_closed:

                # blink عادی
                if (
                    self.min_blink_frames
                    <= self.closed_counter
                    <= self.max_blink_frames
                ):

                    current_time = self.total_frames / self.fps
                    self.blink_timestamps.append(current_time)

                self.closed_counter = 0

            self.eye_closed = False

        # حذف blink های قدیمی‌تر از window
        current_time = self.total_frames / self.fps

        while (
            len(self.blink_timestamps) > 0
            and current_time - self.blink_timestamps[0] > 60
        ):
            self.blink_timestamps.popleft()

        # محاسبه blink rate
        blink_rate = len(self.blink_timestamps)
        blink_state="normal"
        if (blink_rate)<12:
            blink_state="staring"
        elif (blink_rate)>20:
            blink_state="fatigue"
        # تحلیل وضعیت
        is_low_blink = (blink_rate) < 8
        is_high_blink = (blink_rate) > 25

        # احتمال خستگی
        fatigue_score = 0

        if is_low_blink:
            fatigue_score += 1

        if is_high_blink:
            fatigue_score += 1

        return {
            "blink_rate": blink_rate,
            "blink_state": blink_state,
            "is_low_blink": is_low_blink,
            "is_high_blink": is_high_blink,
            "fatigue_score": fatigue_score,
            "total_blinks": len(self.blink_timestamps),
        }

    def reset(self):
        """
        ریست کامل
        """

        self.eye_closed = False
        self.closed_counter = 0
        self.blink_timestamps.clear()
        self.total_frames = 0