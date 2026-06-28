import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional,List
@dataclass
class DetectionResult:
    landmarks:Optional[np.ndarray]=None
    bbox:Optional[tuple]=None
    score:float=0.0
    method:str="mediapipe"
class MediaPipeDetector:
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_face=mp.solutions.face_mesh
            self.face_mesh=self.mp_face.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.available=True
        except ImportError:
            print("Warning: MediaPipe not installed")
            self.available=False
    def detect(self,frame:np.ndarray)->DetectionResult:
        if not self.available:
            return DetectionResult(score=0.0)
        # تبدیل رنگ برای MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return DetectionResult(score=0.0)
        # استخراج 468 نقطه
        h, w = frame.shape[:2]
        landmarks = []
        face_landmarks = results.multi_face_landmarks[0]
            
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append([x, y])
        # محاسبه خودکار BBox
        landmarks_arr = np.array(landmarks)
        x_min, y_min = landmarks_arr.min(axis=0)
        x_max, y_max = landmarks_arr.max(axis=0)
            
        return DetectionResult(
                landmarks=landmarks_arr,
                bbox=(x_min, y_min, x_max-x_min, y_max-y_min),
                score=0.95,
                method="mediapipe"
            )
    
    def release(self):
        if self.available:
            self.face_mesh.close()


