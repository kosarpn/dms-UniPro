import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
class RFTrainer:
    FEATURES = [
            "ear",
            "mar",
            "pitch",
            "yaw",
            "roll",
            "is_head_down",
            "is_head_left",
            "is_head_right",
        ]
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1)
    #لود کردن دیتاست
    def load_dataset(self,csv_path:str):
        csv_path=Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        self.df=pd.read_csv(csv_path)
        print(
                    "Dataset loaded:"
                    ,
                    self.df.shape
                )
    #برای آماده کردن x,y 
    def prepare_data(self):
        missing=[
            f for f in self.FEATURES
            if f not in self.df.columns
        ]
        if missing:
            raise ValueError(
                f"Missing features:{missing}"
            )
        self.X=self.df[self.FEATURES]
        self.y=self.df["label"]
        print("\nFeatures:")
        print(self.FEATURES)

    #split
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
    #ترین دیتا 
    def train(self):
        print("\nTraining Random Forest...")
        self.model.fit(
            self.X_train,
            self.y_train,
        )
        print("Traing finished")
    #ارزیابی
    def evaluate(self):
        predictions=self.model.predict(self.X_test)
        accuracy=accuracy_score(
            self.y_test,
            predictions,
        )
    
        f1=f1_score(
            self.y_test,
            predictions,
            average="binary"

        )
        print("\n====================")
        print("Random Forest Evaluation")
        print("====================")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                self.y_test,
                predictions,
            ))
        print("Confusion Matrix:")
        print(
            confusion_matrix(
                self.y_test,
                predictions,))
        #ذخیره سازی

    def save(self, model_dir="models"):

        model_dir = Path(model_dir)

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.model,
            model_dir / "rf_model.pkl",
        )

        print("\nModel saved:")
        print(model_dir / "rf_model.pkl")
def main():
        trainer=RFTrainer()
        trainer.load_dataset(
        "data/processed/dataset.csv"
            )

        trainer.prepare_data()
        trainer.split_data()
        trainer.train()
        trainer.evaluate()
        trainer.save()

if __name__ == "__main__":
    main()