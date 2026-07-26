from dataclasses import dataclass
import numpy as np 
@dataclass
class DatasetFeatureVector:
    ear:float
    mar:float
    pitch:float
    yaw:float
    roll:float
    face_width:float
    face_height:float
    mouth_width:float
    mouth_height:float
    def to_numpy(self):
        return np.array([
            self.ear,
            self.mar,
            self.mouth_width,
            self.mouth_height,
            self.pitch,
            self.yaw,
            self.roll,
            self.face_width,
            self.face_height
        ]
        ,dtype=np.float32)