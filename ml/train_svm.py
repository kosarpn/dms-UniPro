import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
class SVMTrainer:
    def __init__(self):
        self.model=SVC(kernel="rbf",C=1.0,
                       gamma="scale",probability=True,random_state=42)
        self.scaler=StandardScaler()
    def load_dataset(self,csv_path:str):
        csv_path=Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        self.df = pd.read_csv(csv_path)
    def prepare_data(self):

        self.X = self.df[
            [
                "ear",
                "mar",
                "mouth_width",
                "mouth_height",
                "eye_distance",
                "face_width",
                "face_height",
            ]
        ]

        self.y = self.df["label"]

    def split_data(self):

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y,
        )

    def scale_features(self):

        self.X_train = self.scaler.fit_transform(
            self.X_train
        )

        self.X_test = self.scaler.transform(
            self.X_test
        )

    def train(self):

        self.model.fit(
            self.X_train,
            self.y_train,
        )

    def evaluate(self):

        predictions = self.model.predict(
            self.X_test
        )

        accuracy = accuracy_score(
            self.y_test,
            predictions,
        )

        print(f"\nAccuracy : {accuracy:.4f}\n")

        print("Classification Report\n")

        print(
            classification_report(
                self.y_test,
                predictions,
            )
        )

        print("Confusion Matrix\n")

        print(
            confusion_matrix(
                self.y_test,
                predictions,
            )
        )

    def save(self, model_dir="models"):

        model_dir = Path(model_dir)

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.model,
            model_dir / "svm_model.pkl",
        )

        joblib.dump(
            self.scaler,
            model_dir / "svm_scaler.pkl",
        )
        print("\nModel Saved Successfully")

        print(model_dir / "svm_model.pkl")

        print(model_dir / "svm_scaler.pkl")
def main():

    trainer = SVMTrainer()

    trainer.load_dataset(
        "data/processed/dataset.csv"
    )

    trainer.prepare_data()

    trainer.split_data()

    trainer.scale_features()

    trainer.train()

    trainer.evaluate()

    trainer.save()


if __name__ == "__main__":
    main()

        