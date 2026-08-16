import joblib
from pathlib import Path
import numpy as np

from sklearn.svm import SVC

from ml.classifiers.base_classifier import BaseClassifier
from core.data_types import DrowsinessLevel

from core.features_vector import FeatureVector
from core.data_types import Prediction

class SVMClassifier(BaseClassifier):
    def __init__(
        self,
        model_path="models/svm_model.pkl",
        scaler_path="models/svm_scaler.pkl"
    ):
        self.model = None
        self.scaler = None

        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        if self.model_path.exists() and self.scaler_path.exists():
            self.load(
                self.model_path,
                self.scaler_path
            )
    def _extract_features(
        self,
        features: FeatureVector
    ):
        return np.array([
            features.ear,
            features.mar,
            features.pitch,
            features.yaw,
            features.roll,
            features.is_head_down,
            features.is_head_left,
            features.is_head_right,
        ], dtype=np.float32)
    # -----------------------------
    # Training
    # -----------------------------
    def fit(self, X, y):

        self.model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )

        self.model.fit(
            X,
            y
        )
    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(
        self,
        features: FeatureVector
    ) -> Prediction:
        if self.model is None:
            raise RuntimeError(
                "SVM model is not loaded"
            )
        x = np.array(
            self._extract_features(features)
        )
        x = x.reshape(
            1,
            -1
        )
        if self.scaler:
            x = self.scaler.transform(x)
        prediction = int(
            self.model.predict(x)[0]
        )
        # probability = self.model.predict_proba(x)[0]
        # confidence = float(
        #     np.max(probability)
        # )
        score = self.model.decision_function(x)[0]
        confidence = abs(score)
        return Prediction(
            is_drowsy=bool(prediction),
            confidence=confidence,
            level=(
                DrowsinessLevel.DROWSY
                if prediction
                else
                DrowsinessLevel.ALERT
            ),
            reason=
               "svm_classifier"
            ,
            classifier_name=self.name
        )
    # -----------------------------
    # Persistence
    # -----------------------------

    def save(
        self,
        model_path=None,
        scaler_path=None
    ):

        model_path = model_path or self.model_path
        scaler_path = scaler_path or self.scaler_path


        joblib.dump(
            self.model,
            model_path
        )


        if self.scaler:
            joblib.dump(
                self.scaler,
                scaler_path
            )
    def load(
        self,
        model_path,
        scaler_path
    ):

        self.model = joblib.load(
            model_path
        )

        self.scaler = joblib.load(
            scaler_path
        )
    def reset(self):

        pass
    def predict_batch(self,dataframe):
        x=dataframe[
            ["ear",
             "mar",
             "pitch",
             "yaw",
             "roll",
             "is_head_down",
             "is_head_left",
             "is_head_right",]
        ]
        if self.scaler:
            x=self.scaler.transform(x)
        return self.model.predict(x)