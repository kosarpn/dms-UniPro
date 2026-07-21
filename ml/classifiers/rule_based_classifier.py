from typing import Dict, Any
from ml.classifiers.base_classifier import BaseClassifier


class RuleBasedClassifier(BaseClassifier):
    """
    Classifier مبتنی بر قانون (Rule-Based)
    متد predict را مطابق با فراخوانی main.py پیاده‌سازی می‌کند.
    """

    def __init__(self,
                 ear_threshold: float = 0.25,
                 mar_threshold: float = 0.25,
                 drowsy_frame_threshold: int = 60):
        super().__init__()
        print(
        f"[RULE INIT] ear_threshold={ear_threshold}, "
        f"mar_threshold={mar_threshold}, "
        f"drowsy_frame_threshold={drowsy_frame_threshold}"
    )
            
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.drowsy_frame_threshold = drowsy_frame_threshold

        self.closed_eye_counter = 0
        self.yawn_counter = 0

    def update(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """نسخه update (اختیاری)"""
        return self.predict(
            ear_status={'is_closed': features.get('is_eye_closed', False), 'ear': features.get('ear', 0.0)},
            mar_status={'yawn_detected': features.get('yawn_detected', False)},
            blink_status={'blink_rate': features.get('blink_rate', 0.0)}
        )

    def predict(self,
                ear_status: Dict[str, Any],
                mar_status: Dict[str, Any],
                blink_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        ورودی: سه دیکشنری ear_status, mar_status, blink_status
        خروجی: دیکشنری prediction مطابق نیاز main.py
        """
        ear = ear_status.get('ear', 0.0)
        is_eye_closed = ear_status.get('is_closed', False)
        yawn_detected = mar_status.get('yawn_detected', False)
        blink_rate = blink_status.get('blink_rate', 0.0)

        # شمارش چشم بسته
        if ear < self.ear_threshold or is_eye_closed:
            self.closed_eye_counter += 1
        else:
            self.closed_eye_counter = 0

        # شمارش خمیازه
        if yawn_detected:
            self.yawn_counter += 1
        else:
            self.yawn_counter = max(0, self.yawn_counter - 1)

        # تصمیم‌گیری نهایی
        is_drowsy = False
        confidence = 0.0
        level = "normal"

        if self.closed_eye_counter >= self.drowsy_frame_threshold:
            is_drowsy = True
            confidence = min(0.95, 0.65 + (self.closed_eye_counter - self.drowsy_frame_threshold) * 0.025)
            level = "drowsy"
        elif self.yawn_counter >= 20:
            is_drowsy = True
            confidence = 0.78
            level = "yawning"
        elif blink_rate > 28:
            is_drowsy = True
            confidence = 0.65
            level = "high_blink"

        return {
            'blink_rate': blink_rate,
            'is_drowsy': is_drowsy,
            'is_yawning': yawn_detected,
            'confidence': round(confidence, 2),
            'level': level
        }

    def reset(self) -> None:
        self.closed_eye_counter = 0
        self.yawn_counter = 0
