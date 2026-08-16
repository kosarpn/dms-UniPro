import cv2
import numpy as np
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
class DetectionMethod(Enum):
    """روش تشخیص فعلی"""
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"
    NONE = "none"
@dataclass
class HybridResult:
    """نتیجه تشخیص ترکیبی"""
    success: bool
    landmarks: Optional[np.ndarray]
    face_bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    detection_time_ms: float
    method_used: DetectionMethod
    metadata: Dict = None
class HybridDetector:
    def __init__(self,
                 mediapipe_confidence: float = 0.5,
                 yolo_confidence: float = 0.5,
                 fallback_threshold: int = 5):
        self.mediapipe_confidence = mediapipe_confidence
        self.yolo_confidence = yolo_confidence
        self.fallback_threshold = fallback_threshold
        # وضعیت
        self.current_method = DetectionMethod.MEDIAPIPE
        self.consecutive_failures = 0
        self.last_valid_landmarks = None
        self.last_valid_time = 0
        # آمار
        self.total_detections = 0
        self.total_frames = 0
        self.mediapipe_used = 0
        self.yolo_used = 0
        # مقداردهی اولیه تشخیص‌دهنده‌ها
        self._init_detectors()
    def _init_detectors(self):
        # MediaPipe
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.mediapipe_confidence,
                min_tracking_confidence=0.5
            )
            self.mediapipe_available = True
            print(" MediaPipe initialized")
        except Exception as e:
            print(f" MediaPipe not available: {e}")
            self.mediapipe_available = False
        # YOLO
        # try:
        #     from ultralytics import YOLO
        #     self.yolo_model = YOLO('yolov8n-face.pt')
        #     self.yolo_available = True
        #     print("[HybridDetector] YOLO initialized")
        # except Exception as e:
        #     print(f"[HybridDetector] YOLO not available: {e}")
        #     self.yolo_available = False
    def detect(self, frame: np.ndarray) -> HybridResult:
        self.total_frames += 1
        start_time = time.time()
        # انتخاب روش بر اساس وضعیت
        # if self.current_method == DetectionMethod.MEDIAPIPE:
        #     result = self._detect_mediapipe(frame)
            
        #     # اگر MediaPipe fail شد و شرایط Fallback فراهم است
        #     if not result.success and self._should_fallback():
        #         self.current_method = DetectionMethod.YOLO
        #         self.consecutive_failures = 0
        #         print("[HybridDetector] Switching to YOLO fallback")
        #         result = self._detect_yolo(frame)
        # elif self.current_method == DetectionMethod.YOLO:
        #     result = self._detect_yolo(frame)
        #     # گاهی برگرد به MediaPipe برای تست
        #     if self._should_try_mediapipe():
        #         mp_result = self._detect_mediapipe(frame)
        #         if mp_result.success:
        #             self.current_method = DetectionMethod.MEDIAPIPE
        #             print("[HybridDetector] Switching back to MediaPipe")
        #             result = mp_result
        if self.current_method == DetectionMethod.MEDIAPIPE:
            result = self._detect_mediapipe(frame)
        else:
            result = HybridResult(
                success=False,
                landmarks=None,
                face_bbox=None,
                confidence=0.0,
                detection_time_ms=(time.time() - start_time) * 1000,
                method_used=DetectionMethod.NONE)
        # به‌روزرسانی آمار
        if result.success:
            self.total_detections += 1
            if result.method_used == DetectionMethod.MEDIAPIPE:
                self.mediapipe_used += 1
            else:
                self.yolo_used += 1
            
            self.consecutive_failures = 0
            self.last_valid_landmarks = result.landmarks
            self.last_valid_time = time.time()
        else:
            self.consecutive_failures += 1
        return result
    def _detect_mediapipe(self, frame: np.ndarray) -> HybridResult:
        """تشخیص با MediaPipe"""
        if not self.mediapipe_available:
            return HybridResult(
                success=False, landmarks=None, face_bbox=None,
                confidence=0.0, detection_time_ms=0,
                method_used=DetectionMethod.MEDIAPIPE
            )
        start_time = time.time()
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        try:
            results = self.face_mesh.process(rgb_frame)
            process_time = (time.time() - start_time) * 1000
            if results.multi_face_landmarks:
                # استخراج نقاط کلیدی
                landmarks = []
                for lm in results.multi_face_landmarks[0].landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append([x, y])
                landmarks_np = np.array(landmarks)
                # محاسبه bounding box
                min_xy = np.min(landmarks_np, axis=0)
                max_xy = np.max(landmarks_np, axis=0)
                padding = 20
                bbox = (
                    max(0, min_xy[0] - padding),
                    max(0, min_xy[1] - padding),
                    min(w - min_xy[0], max_xy[0] - min_xy[0] + 2*padding),
                    min(h - min_xy[1], max_xy[1] - min_xy[1] + 2*padding)
                )
                return HybridResult(
                    success=True,
                    landmarks=landmarks_np,
                    face_bbox=bbox,
                    confidence=0.9,
                    detection_time_ms=process_time,
                    method_used=DetectionMethod.MEDIAPIPE
                )
            return HybridResult(
                success=False, landmarks=None, face_bbox=None,
                confidence=0.0, detection_time_ms=process_time,
                method_used=DetectionMethod.MEDIAPIPE
            )     
        except Exception as e:
            return HybridResult(
                success=False, landmarks=None, face_bbox=None,
                confidence=0.0, detection_time_ms=0,
                method_used=DetectionMethod.MEDIAPIPE )
    # def _detect_yolo(self, frame: np.ndarray) -> HybridResult:
    #     """تشخیص با YOLO"""
    #     if not self.yolo_available:
    #         return HybridResult(
    #             success=False, landmarks=None, face_bbox=None,
    #             confidence=0.0, detection_time_ms=0,
    #             method_used=DetectionMethod.YOLO
    #         )
    #     start_time = time.time()
    #     h, w = frame.shape[:2]
    #     try:
    #         results = self.yolo_model(frame, conf=self.yolo_confidence, verbose=False)
    #         process_time = (time.time() - start_time) * 1000
            
    #         if len(results) > 0 and results[0].boxes is not None:
    #             box = results[0].boxes[0]
    #             x1, y1, x2, y2 = box.xyxy[0].tolist()
    #             bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    #             confidence = float(box.conf[0]) if box.conf is not None else 0.7
    #             # YOLO نقاط کلیدی کامل ندارد
    #             # ایجاد نقاط تقریبی از روی bounding box
    #             # landmarks = self._create_approximate_landmarks(bbox, w, h)
    #             landmarks = None
    #             return HybridResult(
    #                 success=True,
    #                 landmarks=landmarks,
    #                 face_bbox=bbox,
    #                 confidence=confidence ,
    #                 detection_time_ms=process_time,
    #                 method_used=DetectionMethod.YOLO)
    #         return HybridResult(
    #             success=False, landmarks=None, face_bbox=None,
    #             confidence=0.0, detection_time_ms=process_time,
    #             method_used=DetectionMethod.YOLO
    #         )   
    #     except Exception as e:
    #         return HybridResult(
    #             success=False, landmarks=None, face_bbox=None,
    #             confidence=0.0, detection_time_ms=0,
    #             method_used=DetectionMethod.YOLO
    #         )
    # def _create_approximate_landmarks(self, bbox: Tuple, frame_w: int, frame_h: int) -> np.ndarray:
    #     """ایجاد نقاط کلیدی تقریبی از روی bounding box (برای Fallback)"""
    #     x, y, w, h = bbox
    #     # نقاط کلیدی تقریبی (468 نقطه - پر کردن با مقادیر پیش‌فرض)
    #     landmarks = []
    #     for i in range(468):
    #         # نقاط اصلی: چشم‌ها، بینی، دهان
    #         if i < 6:  # چشم چپ
    #             lx = x + w * 0.3 + (i % 3) * w * 0.05
    #             ly = y + h * 0.4 + (i // 3) * h * 0.05
    #         elif i < 12:  # چشم راست
    #             lx = x + w * 0.7 + ((i-6) % 3) * w * 0.05
    #             ly = y + h * 0.4 + ((i-6) // 3) * h * 0.05
    #         elif i < 24:  # دهان
    #             lx = x + w * 0.5 + ((i-12) % 4 - 2) * w * 0.1
    #             ly = y + h * 0.7 + ((i-12) // 4) * h * 0.05
    #         else:
    #             lx = x + w * 0.5 + (i % 20 - 10) * w * 0.03
    #             ly = y + h * 0.5 + (i // 20 - 5) * h * 0.02
            
    #         landmarks.append([max(0, min(frame_w, int(lx))), 
    #                         max(0, min(frame_h, int(ly)))])
    #     return np.array(landmarks)
    def _should_fallback(self) -> bool:
        return self.consecutive_failures >= self.fallback_threshold
    def _should_try_mediapipe(self) -> bool:
        """آیا باید برگردیم به MediaPipe؟"""
        return (self.consecutive_failures == 0 and 
                self.total_frames % 30 == 0)  # هر 30 فریم یک بار تست کن
    def get_statistics(self) -> Dict:
        """دریافت آمار"""
        return {
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'detection_rate': (self.total_detections / self.total_frames * 100) if self.total_frames > 0 else 0,
            'mediapipe_used': self.mediapipe_used,
            'yolo_used': self.yolo_used,
            'current_method': self.current_method.value}
    def close(self):
        """بستن اتصال‌ها"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()