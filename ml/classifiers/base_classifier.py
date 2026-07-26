"""
Base interface for all Drowsiness Classifiers.

تمام مدل‌های تصمیم‌گیری پروژه (Rule-Based, SVM, RF, LSTM, CNN-LSTM)
باید از این کلاس ارث‌بری کنند.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from core.features_vector import FeatureVector
from core.data_types import Prediction


class BaseClassifier(ABC):
    """
    Base interface for every drowsiness classifier.
    """

    @abstractmethod
    def predict(
        self,
        features: FeatureVector
    ) -> Prediction:
        """
        Predict driver state.
        Parameters
        ----------
        features : FeatureVector
            Feature vector extracted from current frame.
        Returns
        -------
        dict
        Example
        -------
        {
            "is_drowsy": False,
            "confidence": 0.94,
            "level": "alert",
            "reason": [],
            "classifier": "rule_based"
        }
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal states.

        هنگام تغییر راننده یا شروع مجدد سیستم فراخوانی می‌شود.
        """
        pass

    # ----------------------------------------------------
    # Training API
    # ----------------------------------------------------

    def fit(self, X, y):
        """
        Train classifier.

        فقط برای مدل‌های ML/DL استفاده خواهد شد.
        Rule-Based از این متد استفاده نمی‌کند.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training."
        )

    # ----------------------------------------------------
    # Model Persistence
    # ----------------------------------------------------

    def save(self, path: str):
        """
        Save trained model.

        فقط در SVM / RF / LSTM پیاده‌سازی خواهد شد.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support save()."
        )

    def load(self, path: str):
        """
        Load trained model.

        فقط در SVM / RF / LSTM پیاده‌سازی خواهد شد.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support load()."
        )
    # ----------------------------------------------------
    # Information
    # ----------------------------------------------------
    @property
    def name(self) -> str:
        """
        Name of classifier.
        """
        return self.__class__.__name__
    @property
    def is_trainable(self) -> bool:
        """
        آیا این مدل نیاز به آموزش دارد؟
        """
        return False