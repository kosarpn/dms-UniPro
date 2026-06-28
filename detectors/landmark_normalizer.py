"""
MediaPipe Face Mesh Detector
خروجی: 468 نقطه کلیدی 3D
اولویت اول در سیستم Hybrid
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class MediaPipeResult:
    """نتیجه تشخیص با MediaPipe"""
    success: bool
    landmarks: Optional[np.ndarray]  # (468, 2) or (468, 3)
    confidence: float
    detection_time_ms: float
    face_bbox: Optional[Tuple[int, int, int, int]]
    method: str = "mediapipe"


class MediaPipeDetector:
    """
    تشخیص چهره و استخراج 468 نقطه کلیدی با MediaPipe
    """
    
    # نقاط کلیدی مورد نیاز برای DMS
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321]
    NOSE = [1, 2, 4]
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 refine_landmarks: bool = True):
        
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.consecutive_failures = 0
        self.total_detections = 0
        self.total_frames = 0
        
    def detect(self, frame: np.ndarray) -> MediaPipeResult:
        """
        تشخیص چهره و استخراج نقاط کلیدی
        
        Args:
            frame: تصویر BGR
            
        Returns:
            MediaPipeResult شامل نقاط کلیدی و اطلاعات تشخیص
        """
        self.total_frames += 1
        start_time = time.time()
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.face_mesh.process(rgb_frame)
            process_time = (time.time() - start_time) * 1000
            
            if results.multi_face_landmarks:
                self.consecutive_failures = 0
                self.total_detections += 1
                
                face_landmarks = results.multi_face_landmarks[0]
                
                # استخراج نقاط کلیدی
                landmarks = []
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append([x, y])
                
                landmarks = np.array(landmarks)
                
                # محاسبه bounding box از روی نقاط
                min_xy = np.min(landmarks, axis=0)
                max_xy = np.max(landmarks, axis=0)
                padding = 20
                bbox = (
                    max(0, min_xy[0] - padding),
                    max(0, min_xy[1] - padding),
                    min(w - min_xy[0], max_xy[0] - min_xy[0] + 2*padding),
                    min(h - min_xy[1], max_xy[1] - min_xy[1] + 2*padding)
                )
                
                # محاسبه اطمینان بر اساس تعداد نقاط معتبر
                confidence = min(1.0, len(landmarks) / 468)
                
                return MediaPipeResult(
                    success=True,
                    landmarks=landmarks,
                    confidence=confidence,
                    detection_time_ms=process_time,
                    face_bbox=bbox
                )
            
            else:
                self.consecutive_failures += 1
                
                # اطمینان بر اساس نرخ شکست
                confidence = max(0, 1.0 - (self.consecutive_failures / 10))
                
                return MediaPipeResult(
                    success=False,
                    landmarks=None,
                    confidence=confidence,
                    detection_time_ms=process_time,
                    face_bbox=None
                )
                
        except Exception as e:
            self.consecutive_failures += 1
            return MediaPipeResult(
                success=False,
                landmarks=None,
                confidence=0.0,
                detection_time_ms=0,
                face_bbox=None
            )
    
    def get_detection_rate(self) -> float:
        """درصد موفقیت تشخیص"""
        if self.total_frames == 0:
            return 0.0
        return (self.total_detections / self.total_frames) * 100
    
    def close(self):
        self.face_mesh.close()
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """کشیدن نقاط کلیدی روی فریم (برای دیباگ)"""
        if landmarks is not None:
            # نقاط عمومی (سبز)
            for point in landmarks:
                cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
            
            # نقاط چشم (قرمز)
            for idx in self.LEFT_EYE + self.RIGHT_EYE:
                if idx < len(landmarks):
                    x, y = landmarks[idx]
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # نقاط دهان (زرد)
            for idx in self.MOUTH:
                if idx < len(landmarks):
                    x, y = landmarks[idx]
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
        
        return frame