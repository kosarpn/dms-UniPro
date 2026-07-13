""" Feature Vector Builder تبدیل DTO ها به بردار عددی برای ML models """
from dataclasses import dataclass
from typing import Optional
import numpy as np
from core.data_types import (
    EyeData,
    MouthData
)
@dataclass 
class FeatureVector:
    """ Feature vector قابل استفاده برای ML """
    # ========================================= # Eye Features # =========================================
    ear:float
    average_ear:float
    blink_rate:float
    perclos:float
    closed_duration:float
    # ========================================= # Mouth Features # =========================================
    mar:float
    average_mar:float
    yawn_duration:float
    yawn_count:int
    speaking_detected:float
    smile_detected:float
    # ========================================= # Utility # =========================================
    def to_numpy(self)->np.ndarray:
        """ تبدیل به numpy vector """
        return np.array([
            #Eye
            self.ear,
            self.average_ear,
            self.blink_rate,
            self.perclos,
            self.closed_duration,
            #Mouth
            self.mar,
            self.average_mar,
            self.yawn_duration,
            self.yawn_count,
            self.speaking_detected,
            self.smile_detected,
        ])

def build_feature_vector(
            eye_data:EyeData,
            mouth_data:MouthData,
    )->FeatureVector:
        """ ساخت feature vector از DTO ها """
        return FeatureVector(
            # ===================================== # Eye # =====================================
            ear=eye_data.ear,
            average_ear=eye_data.average_ear,
            blink_rate=eye_data.blink_rate,
            perclos=eye_data.perclos,
            closed_duration=eye_data.closed_duration,

            # ===================================== # Mouth # ===================================== #
            mar=mouth_data.mar,
            average_mar=mouth_data.average_mar, 
            yawn_duration=mouth_data.yawn_duration,
            yawn_count=mouth_data.yawn_count,
            speaking_detected=float( mouth_data.speaking_detected ), 
            smile_detected=float( mouth_data.smile_detected ),
        )

        