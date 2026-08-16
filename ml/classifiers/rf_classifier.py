import joblib
from pathlib import Path
import numpy as np
from ml.classifiers.base_classifier import BaseClassifier
from core.features_vector import FeatureVector
from core.data_types import Prediction, DrowsinessLevel
class RFClassifier(BaseClassifier):
    def __init__(
        self,
        model_path="models/rf_model.pkl"
    ):
        self.model = None
        self.model_path = Path(model_path)
        if self.model_path.exists():
            self.load(self.model_path)
    # ============================
    # Feature Extraction
    # ============================
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
    # ============================
    # Prediction
    # ============================
    def predict(
        self,
        features: FeatureVector
    ) -> Prediction:
        if self.model is None:
            raise RuntimeError(
                "RF model is not loaded"
            )
        x = self._extract_features(features)
        x = x.reshape(
            1,
            -1
        )
        prediction = int(
            self.model.predict(x)[0])

        if hasattr(self.model, "predict_proba"):

            probability = self.model.predict_proba(x)[0]

            confidence = float(
                np.max(probability))

        else:

            confidence = 1.0



        return Prediction(

            is_drowsy=bool(prediction),

            confidence=confidence,

            level=(
                DrowsinessLevel.DROWSY
                if prediction
                else DrowsinessLevel.ALERT
            ),

            reason="rf_classifier",

            classifier_name=self.name
        )



    # ============================
    # Save / Load
    # ============================

    def save(
        self,
        path=None
    ):

        path = path or self.model_path


        joblib.dump(
            self.model,
            path
        )


    def load(
        self,
        path
    ):

        self.model = joblib.load(
            path
        )
    # ============================
    # Reset
    # ============================
    def reset(self):
        pass
    # ============================
    # Offline Evaluation Helper
    # ============================

    def predict_batch(
        self,
        dataframe
    ):

        x = dataframe[

            [
                "ear",
                "mar",
                "pitch",
                "yaw",
                "roll",
                "is_head_down",
                "is_head_left",
                "is_head_right",
            ]

        ]


        return self.model.predict(x)