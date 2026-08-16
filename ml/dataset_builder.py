import os
import cv2
import pandas as pd
from pathlib import Path
from detectors.hybrid_detector import HybridDetector,DetectionMethod
from features.extractor import FeatureExtractor
class DatasetBuilder:
    def __init__(self):
        self.detector = HybridDetector()
        self.extractor = FeatureExtractor()
        self.samples = []
        self.failed_detection=0
        self.failed_landmarks=0
    def process_image(
            self,
            image_path,
            label):
        image = cv2.imread(
            str(image_path) )
        if image is None:
            return
        detection = self.detector.detect(
            image)
        if detection.method_used != DetectionMethod.MEDIAPIPE:
            return
        # فقط MediaPipe
        if not detection.success:
            self.failed_detection += 1
            return
        if detection.landmarks is None:
            self.failed_landmarks += 1
            return
        (
            feature_vector,
            eye_data,
            mouth_data,
            head_pose_data

        ) = self.extractor.extract(
            detection.landmarks,
            image.shape
        )
        bbox = detection.face_bbox
        sample = {
            # Eye features
            "ear":
                eye_data.average_ear,
            # Mouth features
            "mar":
                mouth_data.mar,
            # Head pose
            "pitch":
                head_pose_data.pitch,
            "yaw":
                head_pose_data.yaw,
            "roll":
                head_pose_data.roll,
            "is_head_down": float(head_pose_data.is_head_down),
            "is_head_left": float(head_pose_data.is_head_left),
            "is_head_right": float(head_pose_data.is_head_right),
            "label":
                label
        }
        print(
            "METHOD:",
            detection.method_used,
            "SUCCESS:",
            detection.success,
            "LANDMARKS:",
            detection.landmarks is not None
        )
        self.samples.append(sample)
    def process_folder(
            self,
            folder_path,
            label
    ):
        for file in os.listdir(folder_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = Path(folder_path) / file
            self.process_image(
                image_path,
                label)
    def save_csv(
            self,
            output_path
    ):
        df = pd.DataFrame(
            self.samples
        )
        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True )
        df.to_csv(
            output_path,
            index=False)
        print(
            "Dataset saved:",
            output_path)
        print(
            df.head())
        print(
            df["label"].value_counts())
        print("FAILED DETECTION:", self.failed_detection)
        print("FAILED LANDMARKS:", self.failed_landmarks)
        print("SAVED SAMPLES:", len(self.samples))
