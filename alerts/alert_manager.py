# alerts/alert_manager.py
"""
مدیریت هشدارهای سیستم DMS
شامل هشدارهای بصری، صوتی و ثبت در لاگ
"""

import time
import threading
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum


class AlertType(Enum):
    """نوع هشدار"""
    DROWSY = "drowsy"
    YAWN = "yawn"
    DISTRACTED = "distracted"
    NO_FACE = "no_face"
    LOW_LIGHT = "low_light"


class AlertSeverity(Enum):
    """شدت هشدار"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """یک هشدار"""
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


class AlertManager:
    """
    مدیریت هشدارهای سیستم
    
    ویژگی‌ها:
    - جلوگیری از هشدارهای تکراری (cooldown)
    - هشدار بصری (تغییر رنگ صفحه)
    - هشدار صوتی (بیپ)
    - ثبت تاریخچه هشدارها
    """
    
    def __init__(self,
                 audio_enabled: bool = True,
                 visual_enabled: bool = True,
                 alert_cooldown: float = 2.0,
                 console_log: bool = True):
        """
        Args:
            audio_enabled: فعال بودن هشدار صوتی
            visual_enabled: فعال بودن هشدار بصری
            alert_cooldown: زمان خنک‌سازی بین هشدارها (ثانیه)
            console_log: چاپ هشدار در کنسول
        """
        self.audio_enabled = audio_enabled
        self.visual_enabled = visual_enabled
        self.alert_cooldown = alert_cooldown
        self.console_log = console_log
        
        # تاریخچه هشدارها
        self.alert_history: List[Alert] = []
        self.last_alert_time: Dict[AlertType, float] = {}
        
        # وضعیت فعلی
        self.current_alert: Optional[Alert] = None
        self.alert_active = False
        self.alert_start_time = 0
        
        # رنگ صفحه برای هشدار بصری
        self.alert_color = (0, 0, 255)  # قرمز
    
    def _can_alert(self, alert_type: AlertType) -> bool:
        """
        بررسی آیا می‌توان هشدار داد (جلوگیری از تکرار)
        
        Returns:
            True اگر زمان کافی از هشدار قبلی گذشته باشد
        """
        last_time = self.last_alert_time.get(alert_type, 0)
        return (time.time() - last_time) >= self.alert_cooldown
    
    def _record_alert(self, alert: Alert):
        """ثبت هشدار در تاریخچه"""
        self.alert_history.append(alert)
        self.last_alert_time[alert.type] = alert.timestamp
        
        # نگهداری فقط 100 هشدار آخر
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)
    
    def _play_beep(self):
        """پخش صدای بوق (در ترد جداگانه برای non-blocking)"""
        if not self.audio_enabled:
            return
        
        def _beep():
            try:
                # برای ویندوز
                import winsound
                winsound.Beep(1000, 300)  # 1000Hz, 300ms
                winsound.Beep(1500, 200)
                winsound.Beep(2000, 300)
            except:
                # برای سیستم‌های دیگر
                print("\a")  # ASCII bell
        
        threading.Thread(target=_beep, daemon=True).start()
    
    def trigger_drowsy_alert(self, 
                              confidence: float = 1.0,
                              ear_value: float = 0.0) -> bool:
        """
        فعال کردن هشدار خواب‌آلودگی
        
        Args:
            confidence: اطمینان از تشخیص
            ear_value: مقدار EAR فعلی
            
        Returns:
            True اگر هشدار فعال شد
        """
        if not self._can_alert(AlertType.DROWSY):
            return False
        
        alert = Alert(
            type=AlertType.DROWSY,
            severity=AlertSeverity.CRITICAL,
            message=f"DROWSY! EAR={ear_value:.3f}",
            confidence=confidence,
            metadata={'ear_value': ear_value}
        )
        
        self._record_alert(alert)
        self.current_alert = alert
        self.alert_active = True
        self.alert_start_time = time.time()
        
        # اقدامات هشدار
        if self.console_log:
            print(f"\n🚨 [ALERT] DROWSINESS DETECTED! (confidence: {confidence:.2f})")
        
        if self.audio_enabled:
            self._play_beep()
        
        return True
    
    def trigger_yawn_alert(self, confidence: float = 1.0) -> bool:
        """
        فعال کردن هشدار خمیازه
        
        Returns:
            True اگر هشدار فعال شد
        """
        if not self._can_alert(AlertType.YAWN):
            return False
        
        alert = Alert(
            type=AlertType.YAWN,
            severity=AlertSeverity.WARNING,
            message="Yawn detected - fatigue possible",
            confidence=confidence
        )
        
        self._record_alert(alert)
        
        if self.console_log:
            print(f"\n😮 [ALERT] YAWN DETECTED!")
        
        if self.audio_enabled:
            self._play_beep()
        
        return True
    
    def trigger_distracted_alert(self, confidence: float = 1.0) -> bool:
        """
        فعال کردن هشدار حواس‌پرتی
        """
        if not self._can_alert(AlertType.DISTRACTED):
            return False
        
        alert = Alert(
            type=AlertType.DISTRACTED,
            severity=AlertSeverity.WARNING,
            message="Driver distracted - pay attention!",
            confidence=confidence
        )
        
        self._record_alert(alert)
        
        if self.console_log:
            print(f"\n👀 [ALERT] DISTRACTED DETECTED!")
        
        return True
    
    def trigger_no_face_alert(self, duration: float = 0.0) -> bool:
        """
        هشدار عدم تشخیص چهره
        """
        if not self._can_alert(AlertType.NO_FACE):
            return False
        
        alert = Alert(
            type=AlertType.NO_FACE,
            severity=AlertSeverity.WARNING,
            message=f"Face not detected for {duration:.1f}s",
            metadata={'duration': duration}
        )
        
        self._record_alert(alert)
        
        if self.console_log and duration > 2.0:
            print(f"\n⚠️ [WARNING] No face detected for {duration:.1f}s")
        
        return True
    
    def update_visual(self, frame, drowsy: bool = False, yawning: bool = False) -> object:
        """
        به‌روزرسانی هشدار بصری روی فریم
        
        Args:
            frame: فریم تصویر
            drowsy: آیا هشدار خواب‌آلودگی فعال است
            yawning: آیا خمیازه تشخیص داده شده
            
        Returns:
            فریم با هشدار بصری (در صورت نیاز)
        """
        if not self.visual_enabled:
            return frame
        
        import cv2
        
        h, w = frame.shape[:2]
        
        # هشدار خواب‌آلودگی (قرمز)
        if drowsy or (self.alert_active and self.current_alert and 
                      self.current_alert.type == AlertType.DROWSY):
            # نوار قرمز در پایین صفحه
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "⚠️ DROWSY! WAKE UP! ⚠️", (w//2-180, h-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # اگر زمان هشدار تمام شد
            if self.alert_active and (time.time() - self.alert_start_time) > 3.0:
                self.alert_active = False
        
        # هشدار خمیازه (زرد)
        elif yawning:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-60), (w, h), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "😮 Yawn Detected - Take a break!", (w//2-180, h-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def get_last_alert(self) -> Optional[Alert]:
        """دریافت آخرین هشدار"""
        return self.alert_history[-1] if self.alert_history else None
    
    def get_alert_count(self, alert_type: Optional[AlertType] = None) -> int:
        """
        تعداد هشدارها
        
        Args:
            alert_type: نوع هشدار (اختیاری)
            
        Returns:
            تعداد هشدارها
        """
        if alert_type is None:
            return len(self.alert_history)
        return sum(1 for a in self.alert_history if a.type == alert_type)
    
    def clear_history(self):
        """پاک کردن تاریخچه هشدارها"""
        self.alert_history.clear()
        self.last_alert_time.clear()
        self.alert_active = False
    
    def get_statistics(self) -> Dict:
        """دریافت آمار هشدارها"""
        return {
            'total_alerts': len(self.alert_history),
            'drowsy_alerts': self.get_alert_count(AlertType.DROWSY),
            'yawn_alerts': self.get_alert_count(AlertType.YAWN),
            'distracted_alerts': self.get_alert_count(AlertType.DISTRACTED),
            'last_alert_time': self.last_alert_time,
            'alert_active': self.alert_active
        }
    
    def cleanup(self):
        """پاکسازی منابع"""
        self.clear_history()
        self.current_alert = None


class SoundAlert:
    """
    کلاس ساده برای هشدار صوتی
    """
    
    @staticmethod
    def beep(frequency: int = 1000, duration_ms: int = 200):
        """پخش یک بوق ساده"""
        try:
            import winsound
            winsound.Beep(frequency, duration_ms)
        except:
            print("\a")


class VisualAlert:
    """
    کلاس ساده برای هشدار بصری
    """
    
    @staticmethod
    def draw_warning(frame, message: str, color: tuple = (0, 0, 255)):
        """کشیدن هشدار روی صفحه"""
        import cv2
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-60), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, message, (w//2-150, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame