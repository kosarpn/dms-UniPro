import os
import cv2
from pathlib import Path
import pandas as pd
from ml.feature_extractor import FeaturExtractor
from ml.dataset_features import DatasetFeatureVector


class DatasetBuilder:
    def __init__(self):
        self.extractor = FeaturExtractor()
        self.samples = []

    def process_image(self, image_path: str) -> DatasetFeatureVector | None:
        image_path = Path(image_path)
        if not image_path.exists():
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            return None

        features = self.extractor.extract(image)
        return features

    def process_folder(self, folder_path: str, label: int):
        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith((".jpg")):
                continue

            image_path = os.path.join(folder_path, file_name)
            features = self.process_image(image_path)

            if features is None:
                continue

            features.label = label
            self.samples.append(features)

    def process_dataset(self, dataset_root: str):
        train_path = os.path.join(dataset_root, "train")

        # پوشه notdrowsy
        notdrowsy_path = os.path.join(train_path, "notdrowsy")
        if os.path.exists(notdrowsy_path):
            print("Processing notdrowsy...")
            self.process_folder(notdrowsy_path, label=0)

        # پوشه drowsy
        drowsy_path = os.path.join(train_path, "drowsy")
        if os.path.exists(drowsy_path):
            print("Processing drowsy...")
            for folder_name in os.listdir(drowsy_path):
                folder_path = os.path.join(drowsy_path, folder_name)
                if os.path.isdir(folder_path):
                    self.process_folder(folder_path, label=1)

    def save_csv(self, output_path: str = "data/processed/dataset.csv") -> None:
        if len(self.samples) == 0:
            raise ValueError("No samples found. Dataset is empty!")

        rows = []
        for sample in self.samples:
            rows.append({
                "ear": sample.ear,
                "mar": sample.mar,
                "mouth_width": sample.mouth_width,
                "mouth_height": sample.mouth_height,
                "eye_distance": sample.eye_distance,
                "face_width": sample.face_width,
                "face_height": sample.face_height,
                "label": sample.label
            })

        df = pd.DataFrame(rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"✅ Dataset saved to: {output_path}")
        print(f"Total samples: {len(df)}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
