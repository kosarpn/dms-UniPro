from ml.classifiers.base_classifier import BaseClassifier
from core.features_vector import FeatureVector
from calibration.calibration_manager import DriverProfile
from core.features_vector import FeatureVector
from core.data_types import Prediction,DrowsinessLevel
class RuleBasedClassifier(BaseClassifier):
    
    def __init__(self, profile: DriverProfile | None = None):
        

        self.score_threshold = 60
        self.profile = profile
        self.perclos_threshold = 0.35
        self.closed_duration_threshold = 2.0
        self.yawn_duration_threshold = 1.5
        self.head_down_frames_threshold = 20
        self.blink_rate_threshold = 30
    def predict(self, features:FeatureVector):
        score = 0
        reasons = []
        if features.average_ear<self.profile.ear_threshold:
            score+=25
            reasons.append("low_ear")
        if features.perclos > self.perclos_threshold:
            score += 25
            reasons.append("high_perclos")
        if features.closed_duration > self.closed_duration_threshold:
            score += 20
            reasons.append("eyes_closed")
        if features.blink_rate > self.blink_rate_threshold:
            score += 10
            reasons.append("high_blink_rate")
        if features.yawn_duration > self.yawn_duration_threshold:
            score += 10
            reasons.append("yawning")
        if features.head_down_frames > self.head_down_frames_threshold:
            score += 10
            reasons.append("head_down")
        confidence = min(score / 100.0, 1.0)
        # تعیین level
        if features.closed_duration >= 3:

            level = DrowsinessLevel.DROWSY


        elif score >= 60:

            level = DrowsinessLevel.SEVERE


        elif score >= 40:

            level = DrowsinessLevel.MODERATE


        elif score >= 20:

            level = DrowsinessLevel.LIGHT


        else:

            level = DrowsinessLevel.ALERT
 

        return Prediction(

            is_drowsy = score >= self.score_threshold,

            confidence = confidence,

            level = level,

            classifier_name="RuleBased",

            reason=", ".join(reasons)

        )
    def reset(self):
            """
            Reset classifier internal states
            """
            pass

