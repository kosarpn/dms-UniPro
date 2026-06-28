import cv2
import numpy as np
from .mediapipe_detector import DetectionResult

class YOLODetector:
    def __init__(self, model_path: str = "yolov8n-face.pt"):
        self.model_path = model_path
        self.model = None
        self.available = False
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
        except Exception as e:
            print(f"YOLO load failed: {e}")
            
    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self.available:
            return DetectionResult(score=0.0)
            
        results = self.model(frame, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return DetectionResult(score=0.0)
            
        # بهترین تشخیص را بگیر
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # اگر keypoints داشت (بعضی مدل‌های YOLO face)
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            # تبدیل ۵ نقطه YOLO به فرمت MediaPipe (بعداً interpolate می‌کنیم)
            landmarks = self._map_yolo_to_full(kpts, x1, y1, x2, y2)
        else:
            landmarks = None
            
        return DetectionResult(
            landmarks=landmarks,
            bbox=(x1, y1, x2-x1, y2-y1),
            score=conf,
            method="yolo"
        )
        
    def _map_yolo_to_full(self, keypoints, x1, y1, x2, y2):
        """
        ۵ نقطه YOLO (چشم‌ها، بینی، گوش‌ها) را به فرمت اولیه مپ می‌کند
        بعداً کالمن یا نرمالایزر آن را کامل می‌کند
        """
        # برای شروع فقط همین ۵ نقطه را برمی‌گرداند
        # نقاط: [چپ چشم چپ، راست چشم چپ، بینی، چپ چشم راست، راست چشم راست]
        if len(keypoints) >= 5:
            return keypoints[:5]
        return None
