
import time
import numpy as np
from typing import List, Optional, Dict
def extract_mouth_points(face_landmarks:np.ndarray)->np.ndarray:
    """
    استخراج 20 نقطه outer lip از MediaPipe
    Args:
        face_landmarks: (478, 2) کل نقاط چهره 
    Returns:
        (20, 2) نقاط کامل دهان
    """
    MOUTH_OUTER = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,    # بالا
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146  # پایین
    ]
    MOUTH_LANDMARKS = [
    61,   # گوشه چپ (left corner)
    39,   # بالای چپ (top-left)
    0,    # بالای مرکز (top-center)
    269,  # بالای راست (top-right)
    291,  # گوشه راست (right corner)
    17,   # پایین مرکز (bottom-center)
]
    if face_landmarks.shape[0] < 478:
        raise ValueError(f"Need 478 landmarks, got {face_landmarks.shape[0]}")
    
    return face_landmarks[MOUTH_OUTER]


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    """
    محاسبه MAR با استفاده از کل contour دهان (20 نقطه)
    Args:
        mouth_landmarks: (20, 2) یا (478, 2) 
    Returns:
        MAR (0.1-0.2 = بسته, 0.6+ = خمیازه)
    """
    if mouth_landmarks is None or len(mouth_landmarks) < 20:
        return 0.0      
    if mouth_landmarks.shape[0]==478:
        mouth_landmarks=extract_mouth_points(mouth_landmarks)
    upper_lip=mouth_landmarks[:10]
    lower_lip=mouth_landmarks[10:]
    vertical_distances=[]
    for upper_point in upper_lip:
        distances=np.linalg.norm(lower_lip-upper_point,axis=1)
        vertical_distances.append(np.min(distances))
    avg_vertical=np.mean(vertical_distances)
    horizontal=np.linalg.norm(mouth_landmarks[0]-mouth_landmarks[10])
    if horizontal<1e-6:
        return 0.0
    mar=avg_vertical/horizontal
    return np.clip(mar,0.05,0.90)
def calculate_mouth_geometry(mouth_landmarks: np.ndarray):
    """
    محاسبه ابعاد اصلی دهان
    """
    if mouth_landmarks.shape[0] == 478:
        mouth_landmarks = extract_mouth_points(mouth_landmarks)
    left_corner = mouth_landmarks[0]
    right_corner = mouth_landmarks[10]
    top_center = mouth_landmarks[5]
    bottom_center = mouth_landmarks[15]
    mouth_width = np.linalg.norm(left_corner - right_corner)
    mouth_height = np.linalg.norm(top_center - bottom_center)
    return mouth_width, mouth_height
class MouthAspectRatioAnalyzer:
    """
    تحلیل‌گر MAR با حفظ تاریخچه و تشخیص خمیازه
    """
    def __init__(self, 
                 history_size: int = 30,
                 yawn_threshold: float = 0.25,
                 min_yawn_frames: int = 5):
        """
        Args:
            history_size: تعداد فریم‌هایی که نگهداری می‌شوند
            yawn_threshold: آستانه تشخیص خمیازه
            min_yawn_frames: حداقل فریم متوالی برای تأیید خمیازه
        """
        self.prev_mar=None
        self.history_size = history_size
        self.yawn_threshold = yawn_threshold
        self.min_yawn_frames = min_yawn_frames
        self.mouth_open_start_time=None
        # تاریخچه
        self.mar_history: List[float] = []
        self.yawn_frames_count = 0
        self.yawn_total_count = 0
        self.total_frames = 0  
        # وضعیت فعلی
        self.is_currently_yawning = False
        self.last_yawn_frame = 0
        #talking detection 
        self.mar_variance_window = []
    
    def update(self, mar_value: float, mouth_landmarks: np.ndarray) -> Dict:
        current_time = time.time()
        mar_velocity = 0.0
        yawn_detected = False

        # محاسبه واریانس MAR برای تشخیص حرف زدن
        self.mar_variance_window.append(mar_value)
        if len(self.mar_variance_window) > 15:
            self.mar_variance_window.pop(0)
        mar_variance = np.var(self.mar_variance_window)

        # محاسبه سرعت تغییر MAR
        if self.prev_mar is not None:
            mar_velocity = mar_value - self.prev_mar
        self.prev_mar = mar_value

        # اضافه کردن به تاریخچه
        self.mar_history.append(mar_value)
        if len(self.mar_history) > self.history_size:
            self.mar_history.pop(0)
        self.total_frames += 1

        # تشخیص خمیازه
        is_yawn = mar_value > self.yawn_threshold

        # تنظیم زمان شروع باز بودن دهان
        if is_yawn:
            if self.mouth_open_start_time is None:
                self.mouth_open_start_time = current_time
            mouth_open_duration = current_time - self.mouth_open_start_time
        else:
            self.mouth_open_start_time = None
            mouth_open_duration = 0.0

        # شمارش فریم‌های خمیازه
        if is_yawn:
            self.yawn_frames_count += 1
        else:
            if self.yawn_frames_count >= self.min_yawn_frames:
                self.yawn_total_count += 1
                self.last_yawn_frame = self.total_frames
                self.is_currently_yawning = True
                yawn_detected = True
            self.yawn_frames_count = 0
            self.is_currently_yawning = False

        # اگر هنوز در حال خمیازه است
        if is_yawn and self.yawn_frames_count >= self.min_yawn_frames:
            self.is_currently_yawning = True
            yawn_detected = True

        # تشخیص خمیازه اخیر
        if self.last_yawn_frame > 0 and (self.total_frames - self.last_yawn_frame) < 5:
            yawn_detected = True

        # محاسبه هندسه دهان
        mouth_width, mouth_height = calculate_mouth_geometry(mouth_landmarks)
        smile_ratio = mouth_width / (mouth_height + 1e-6)
        is_smiling = smile_ratio > 4.5

        # تشخیص حرف زدن
        is_talking = (mar_variance > 0.002) and (mouth_open_duration < 1.5)

        # میانگین و حداکثر MAR
        avg_mar = np.mean(self.mar_history[-10:]) if len(self.mar_history) >= 10 else mar_value
        max_mar = max(self.mar_history[-20:]) if self.mar_history else mar_value

        return {
            'mar': mar_value,
            'average_mar': avg_mar,
            'max_mar': max_mar,
            'is_yawning': self.is_currently_yawning,
            'is_currently_yawning': self.is_currently_yawning,
            'yawn_detected': yawn_detected,
            'yawn_frames': self.yawn_frames_count,
            'yawn_total_count': self.yawn_total_count,
            'mouth_open_duration': mouth_open_duration,
            'mouth_width': mouth_width,
            'mouth_height': mouth_height,
            'mar_velocity': mar_velocity,
            'mar_variance': mar_variance,
            'smile_ratio': smile_ratio,
            'is_smiling': is_smiling,
            'is_talking': is_talking,
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