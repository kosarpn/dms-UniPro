"""
Feature Extractor

MediaPipe FaceMesh landmarks
            |
            v
    EAR Analyzer
    MAR Analyzer
    Head Pose Estimator
            |
            v
      EyeData
      MouthData
      HeadPoseData
            |
            v
      Feature Vector
"""

from features.ear import EyeAspectRatioAnalyzer
from features.mar import MouthAspectRatioAnalyzer
from features.head_pose import HeadPoseEstimator

import numpy as np


from core.data_types import (
    EyeData,
    MouthData,
    HeadPoseData,
)

from core.features_vector import build_feature_vector


from features.ear import (
    calculate_average_ear,
)

from features.mar import (
    calculate_mar,
    extract_mouth_points,
)


# MediaPipe FaceMesh indices

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


class FeatureExtractor:

    def __init__(
        self
    ):

        self.ear_analyzer = EyeAspectRatioAnalyzer()
        self.mar_analyzer = MouthAspectRatioAnalyzer()
        self.head_pose_estimator = HeadPoseEstimator()


    # =========================================================

    def extract(
        self,
        landmarks: np.ndarray,
        image_shape,
    ):

        """
        landmarks:
            MediaPipe FaceMesh landmarks
            shape = (468,2) 

        return:
            feature_vector,
            eye_data,
            mouth_data,
            head_pose_data
        """


        # =====================================================
        # 1) EAR
        # =====================================================

        left_eye = landmarks[LEFT_EYE]

        right_eye = landmarks[RIGHT_EYE]


        ear_value = calculate_average_ear(
            left_eye,
            right_eye
        )


        ear_result = self.ear_analyzer.update(
            ear_value
        )


        eye_data = EyeData(

            ear=ear_result["ear"],

            average_ear=
                ear_result["average_ear"],

            is_closed=
                ear_result["is_closed"],


            blink_count=
                ear_result["blink_count"],


            blink_rate=
                ear_result["blink_rate"],


            closed_duration=
                ear_result["closed_duration"],


            perclos=
                ear_result["perclos"]

        )



        # =====================================================
        # 2) MAR
        # =====================================================


        mouth_points = extract_mouth_points(
            landmarks
        )


        mar_value = calculate_mar(
            mouth_points
        )


        mar_result = self.mar_analyzer.update(
            mar_value,
            mouth_points
        )


        mouth_data = MouthData(

            mar=
                mar_result["mar"],


            average_mar=
                mar_result["average_mar"],


            max_mar=
                mar_result["max_mar"],


            mouth_width=
                mar_result["mouth_width"],


            mouth_height=
                mar_result["mouth_height"],


            is_yawning=
                mar_result["is_yawning"],


            yawn_detected=
                mar_result["yawn_detected"],


            yawn_duration=
                mar_result["mouth_open_duration"],


            yawn_count=
                mar_result["yawn_total_count"],


            speaking_detected=
                mar_result["is_talking"],


            smile_detected=
                mar_result["is_smiling"],

        )



        # =====================================================
        # 3) Head Pose
        # =====================================================


        hp = self.head_pose_estimator.estimate(
            landmarks,
            image_shape
        )


        if hp is None:

            head_pose_data = HeadPoseData(

                yaw=0.0,
                pitch=0.0,
                roll=0.0,

                is_head_down=False,
                head_down_frames=0,

                is_head_left=False,
                is_head_right=False
            )


        else:

            head_pose_data = HeadPoseData(

                yaw=hp["yaw"],

                pitch=hp["pitch"],

                roll=hp["roll"],


                is_head_down=
                    hp.get(
                        "head_down",
                        False
                    ),


                head_down_frames=
                    hp.get(
                        "head_down_frames",
                        0
                    ),


                is_head_left=
                    hp.get(
                        "head_left",
                        False
                    ),


                is_head_right=
                    hp.get(
                        "head_right",
                        False
                    )
            )



        # =====================================================
        # 4) Feature Vector
        # =====================================================


        feature_vector = build_feature_vector(

            eye_data,

            mouth_data,

            head_pose_data
        )


        return (

            feature_vector,

            eye_data,

            mouth_data,

            head_pose_data

        )