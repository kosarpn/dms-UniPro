import cv2
import mediapipe as mp
import numpy as np 
from ml.dataset_features import DatasetFeatureVector
class FeaturExtractor:
    def __init__(self):
        #را اندازی mediapipe face mesh
        self.mp_face_mesh=mp.solutions.face_mesh
        self.face_mesh=self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5             )
        #اندیس لندمارک های مهم 
        self.LEFT_EYE=[33,160,158,133,153,144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
    def _calculate_ear(self, landmarks, eye_indices):
        """محاسبه Eye Aspect Ratio (EAR) برای تشخیص بسته بودن چشم"""
        points = [landmarks[i] for i in eye_indices]
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        C = np.linalg.norm(points[0] - points[3])
        ear = (A + B) / (2.0 * C)
        return ear
    def _calculate_mar(self, landmarks):
        """محاسبه Mouth Aspect Ratio (MAR) برای تشخیص خمیازه"""
        A = np.linalg.norm(landmarks[81] - landmarks[311])
        B = np.linalg.norm(landmarks[13] - landmarks[14])
        C = np.linalg.norm(landmarks[78] - landmarks[308])
        mar = (A + B) / (2.0 * C)
        return mar
    def extract(self,image:np.ndarray):
        """
        دریافت یک تصویر و استخراج ویژگی‌ها
        خروجی: DatasetFeatureVector یا None (اگر صورت تشخیص داده نشود)
        """
        if image is None:
            return None
        h,w=image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
        # محاسبه EAR (میانگین چشم چپ و راست)
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        # محاسبه MAR
        mar = self._calculate_mar(landmarks)
        # محاسبه ابعاد دهان
        mouth_width = np.linalg.norm(landmarks[78] - landmarks[308])
        mouth_height = np.linalg.norm(landmarks[13] - landmarks[14])
        # محاسبه فاصله بین دو چشم
        eye_distance = np.linalg.norm(landmarks[33] - landmarks[263])

        # محاسبه ابعاد صورت
        face_width = np.linalg.norm(landmarks[234] - landmarks[454])
        face_height = np.linalg.norm(landmarks[10] - landmarks[152])

        # ساخت و بازگشت آبجکت DatasetFeatureVector
        return DatasetFeatureVector(
            ear=float(ear),
            mar=float(mar),
            mouth_width=float(mouth_width),
            mouth_height=float(mouth_height),
            eye_distance=float(eye_distance),
            face_width=float(face_width),
            face_height=float(face_height)
        )

