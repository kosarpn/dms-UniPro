import numpy as np
import numpy as np


class FeatureExtractor:

    def __init__(
        self,
        ear_analyzer,
        mar_analyzer,
        head_pose_estimator,
        kalman_filter
    ):

        self.ear_analyzer = ear_analyzer
        self.mar_analyzer = mar_analyzer
        self.head_pose_estimator = head_pose_estimator
        self.kalman_filter = kalman_filter



    def extract(self, landmarks):

        features = {
            "ear":0.0,
            "mar":0.0,
            "head_pose":None
        }


        ear = self._extract_ear(landmarks)

        ear_filtered = self.kalman_filter.update_ear(
            ear
        )


        mar = self._extract_mar(landmarks)

        mar_filtered = self.kalman_filter.update_mar(
            mar
        )


        head_pose = self.head_pose_estimator.process(
            landmarks
        )


        features["ear"] = ear_filtered
        features["mar"] = mar_filtered
        features["head_pose"] = head_pose


        return features



    def _extract_ear(self, landmarks):

        LEFT_EYE=[
            33,160,158,133,153,144
        ]

        RIGHT_EYE=[
            362,385,387,263,373,380
        ]


        left=landmarks[LEFT_EYE]
        right=landmarks[RIGHT_EYE]


        from features.ear import calculate_average_ear

        return calculate_average_ear(
            left,
            right
        )



    def _extract_mar(self, landmarks):

        MOUTH=[
            61,185,40,39,37,
            0,267,269,270,409,
            291,375,321,405,314,
            17,84,181,91,146
        ]


        from features.mar import calculate_mar


        return calculate_mar(
            landmarks[MOUTH]
        )