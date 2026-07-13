from dataclasses import dataclass
import numpy as np 
@dataclass
class DatasetFeatureVector:
    ear:float
    mar:float
    mouth_width:float
    mouth_height:float
    eye_distance:float
    face_width:float
    face_height:float
    def to_numpy(self):
        return np.array([
            self.ear,
            self.mar,
            self.mouth_width,
            self.mouth_height,
            self.eye_distance,
            self.face_width,
            self.face_height
        ]
        ,dtype=np.float32)