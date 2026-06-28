
""" this is extra file
"""
import cv2 
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from typing import Optional,List,dict,Any,Tuple
from pathlib import Path

class YOLOFaceDetector:
    """
    تشخیص چهره با استفاده از YOLOv8-Face
    
    مزایا نسبت به MediaPipe:
    - دقت بالاتر در زوایای مختلف
    - تشخیص چهره‌های کوچک (دور)
    - سرعت بالا (42+ FPS)
    - خروجی bounding box با کیفیت

    """
    def __init__(self,
            model_path:str='yolov8n-face.pt',          
            conf_threshold: float = 0.5,
             iou_threshold: float = 0.45
                 ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # بارگذاری مدل YOLO
        print(f"[INFO] Loading YOLOv8-Face model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            print(f"[INFO] Model loaded successfully on {self.model.device}")
        except Exception as e:
            print(f"[ERROR] Failed to load the model! Error details: {e}")
        # اطلاعات مدل
        self.input_size = 640  # اندازه ورودی استاندارد YOLO
        self.classes = ['face']  # فقط چهره
        def detect(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
            """
            تشخیص چهره‌ها در فریم
            
            Args:
                frame: تصویر BGR از دوربین
                
            Returns:
                (bounding_boxes, landmarks)
                - bounding_boxes: لیستی از (x, y, width, height)
                - landmarks: لیستی از نقاط کلیدی (برای هر چهره)
            """

            h, w = frame.shape[:2]
            results = self.model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False  # جلوگیری از چاپ لاگ اضافی
            )
            boxes = []
            all_landmarks = []
            if len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        for i, box in enumerate(result.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x, y, width, height = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                            # اطمینان از محدوده معتبر
                            x = max(0, x)
                            y = max(0, y)
                            width = min(w - x, width)
                            height = min(h - y, height)
                            if width > 10 and height > 10:  # حداقل سایز برای تشخیص معتبر
                                boxes.append((x, y, width, height))
                                # استخراج نقاط کلیدی صورت (اگه مدل داره)
                                landmarks = self._extract_landmarks(result, i, w, h)
                                all_landmarks.append(landmarks)
                    return boxes,all_landmarks       
    def _extract_landmarks(self, result, idx: int, w: int, h: int) -> Optional[np.ndarray]:
            """
            استخراج نقاط کلیدی صورت (برای EAR/MAR)
            
            YOLOv8-Face می‌تواند نقاط کلیدی چشم و دهان را هم برگرداند
            """
            try:
                # YOLOv8-Face می‌تواند نقاط کلیدی را برگرداند
                # این بستگی به نسخه مدل دارد
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kpts = result.keypoints[idx].data[0].cpu().numpy()
                    # تبدیل به مختصات پیکسلی
                    landmarks = []
                    for i in range(0, len(kpts), 3):
                        x, y, conf = kpts[i], kpts[i+1], kpts[i+2]
                        if conf > 0.5:
                            landmarks.append([int(x), int(y)])
                    return np.array(landmarks) if landmarks else None
            except Exception:
                pass
            
            return None
  
    def get_eye_landmarks(self, landmarks: np.ndarray, side: str = 'left') -> Optional[np.ndarray]:
        """
        استخراج نقاط کلیدی چشم برای محاسبه EAR
            
        بسته به خروجی مدل YOLOv8-Face
        """
        if landmarks is None or len(landmarks) < 6:
            return None
        if side == 'left':
                indices = [0, 1, 2, 3, 4, 5]
        else:
                indices = [6, 7, 8, 9, 10, 11]
        eye_points = []
        for idx in indices:
            if idx < len(landmarks):
                eye_points.append(landmarks[idx])
        return np.array(eye_points) if len(eye_points) == 6 else None
    def get_mouth_landmarks(self, landmarks: np.ndarray) -> Optional[np.ndarray]:

