import joblib
from typing import Dict,Any
from pathlib import Path
import numpy as np
from ml.classifiers.base_classifier import BaseClassifier
from ml.dataset_features import DatasetFeatureVector
class SVMClassifier(BaseClassifier):
    def __init__(self,model_path:str="models/svm_model.pkl",scaler_path:str="models/svm_scaler.pkl"):
        self.model=None
        self.scaler=None
        self.load(model_path, scaler_path)
    def  load(self, model_path:str,scaler_path:str):
        self.model=joblib.load(Path(model_path))
        self.scaler=joblib.load(
            Path(scaler_path)
        )
    def predict(self, features:DatasetFeatureVector,)->dict:
        x=features.to_numpy()
        x=x.reshape(1,-1)
        x=self.scaler.transform(x)
        prediction=self.model.predict(x)[0]
        probability = self.model.predict_proba(x)[0]
        confidence = float(
                np.max(probability))
        return {
                "is_drowsy": bool(prediction),
                "confidence": confidence,
            }
    def reset(self):
        pass
      