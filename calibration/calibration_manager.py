# calibration/calibration_manager.py
"""
مدیریت کالیبراسیون کامل سیستم DMS
"""
import time
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
class CalibrationPhase(Enum):
    """مراحل کالیبراسیون"""
    IDLE = "idle"
    CALIBRATING_EAR = "calibrating_ear"
    CALIBRATING_DISTANCE = "calibrating_distance"
    CALIBRATING_POSITION = "calibrating_position"
    COMPLETE = "complete"
@dataclass
class DriverProfile:
    driver_id: str
    ear_normal: float = 0.27
    ear_threshold: float = 0.23
    ear_history: List[float] = field(default_factory=list)
    face_area_at_50cm: float = 25000
    eye_distance_at_50cm: float = 60
    driver_x_center: float = 320
    driver_x_range: Tuple[float, float] = (200, 440)
    driver_y_range: Tuple[float, float] = (100, 380)
    calibration_time: float = field(default_factory=time.time)
class CalibrationManager:
    """مدیر کالیبراسیون جامع"""
    def __init__(self, 
                 calibration_duration_sec: float = 5.0,
                 reference_distance_cm: float = 50.0):
        self.calibration_duration = calibration_duration_sec
        self.reference_distance = reference_distance_cm
        self.current_phase = CalibrationPhase.IDLE
        self.current_driver_id: Optional[str] = None
        self.current_profile: Optional[DriverProfile] = None
        self._temp_data = {
            'ear': [],
            'face_bboxes': [],
            'eye_distances': [],
            'timestamps': []
            }
        self.calibration_start_time = 0
        self.is_calibrating = False
        self.saved_profiles: Dict[str, DriverProfile] = {} 
    def start_calibration(self, driver_id: str) -> bool:
        """شروع فرآیند کالیبراسیون"""
        if self.is_calibrating:
            return False
        self.current_driver_id = driver_id
        self.current_phase = CalibrationPhase.CALIBRATING_EAR
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        
        self._temp_data = {
            'ear': [],
            'face_bboxes': [],
            'eye_distances': [],
            'timestamps': []
        }
        print("=" * 50)
        print(f"CALIBRATION STARTED for driver: {driver_id}")
        print(f"Phase 1: Please look straight at camera for {self.calibration_duration} seconds...")
        print("=" * 50)
        
        return True
    
    def add_frame(self, ear: float, face_bbox: Tuple, eye_distance: Optional[float] = None):
        """اضافه کردن داده هر فریم"""
        if not self.is_calibrating:
            return
        current_time = time.time()
        elapsed = current_time - self.calibration_start_time
        self._temp_data['ear'].append(ear)
        self._temp_data['face_bboxes'].append(face_bbox)
        if eye_distance:
            self._temp_data['eye_distances'].append(eye_distance)
        self._temp_data['timestamps'].append(current_time)
        if elapsed >= self.calibration_duration:
            if self.current_phase == CalibrationPhase.CALIBRATING_EAR:
                self._finalize_ear_calibration()
                self._start_next_phase()
            elif self.current_phase == CalibrationPhase.CALIBRATING_DISTANCE:
                self._finalize_distance_calibration()
                self._start_next_phase()
            elif self.current_phase == CalibrationPhase.CALIBRATING_POSITION:
                self._finalize_position_calibration()
                self._complete_calibration()
    def _finalize_ear_calibration(self):
        ear_values = self._temp_data['ear']
        if len(ear_values) > 10:
            stable_ears = ear_values[-int(len(ear_values) * 0.7):]
            normal_ear = np.mean(stable_ears)
            ear_threshold = normal_ear * 0.85
            self.current_profile = DriverProfile(
                driver_id=self.current_driver_id,
                ear_normal=normal_ear,
                ear_threshold=ear_threshold
            )  
            print(f"[CALIBRATION] EAR calibration complete:")
            print(f"  - Normal EAR: {normal_ear:.3f}")
            print(f"  - Drowsy threshold: {ear_threshold:.3f}")
        else:
            print("[CALIBRATION] ERROR: Not enough EAR data!")
            self._abort_calibration()
    
    def _finalize_distance_calibration(self):
        """نهایی‌سازی کالیبراسیون فاصله"""
        face_bboxes = self._temp_data['face_bboxes']
        
        if face_bboxes:
            areas = [b[2] * b[3] for b in face_bboxes[-int(len(face_bboxes) * 0.7):]]
            avg_area = np.mean(areas)
            self.current_profile.face_area_at_50cm = avg_area
            print(f"[CALIBRATION] Distance calibration complete:")
            print(f"  - Face area at {self.reference_distance}cm: {avg_area:.0f}px")
            # فاصله دو چشم (اگر داریم)
            eye_distances = self._temp_data['eye_distances']
            if eye_distances:
                avg_eye_dist = np.mean(eye_distances[-int(len(eye_distances) * 0.7):])
                self.current_profile.eye_distance_at_50cm = avg_eye_dist
                print(f"  - Eye distance: {avg_eye_dist:.1f}px")
            else:
                print(f"  - Eye distance: N/A")
    
    def _finalize_position_calibration(self):
        """نهایی‌سازی کالیبراسیون موقعیت"""
        face_bboxes = self._temp_data['face_bboxes']
        if face_bboxes:
            centers_x = [b[0] + b[2]/2 for b in face_bboxes]
            centers_y = [b[1] + b[3]/2 for b in face_bboxes]
            center_x = np.mean(centers_x)
            center_y = np.mean(centers_y)
            std_x = np.std(centers_x)
            std_y = np.std(centers_y) 
            self.current_profile.driver_x_center = center_x
            self.current_profile.driver_x_range = (
                max(0, center_x - 2.5 * std_x),
                min(640, center_x + 2.5 * std_x)
            )
            self.current_profile.driver_y_range = (
                max(0, center_y - 2.5 * std_y),
                min(480, center_y + 2.5 * std_y)
            )
            print(f"[CALIBRATION] Position calibration complete:")
            print(f"  - X range: {self.current_profile.driver_x_range[0]:.0f}-{self.current_profile.driver_x_range[1]:.0f}px")
            print(f"  - Y range: {self.current_profile.driver_y_range[0]:.0f}-{self.current_profile.driver_y_range[1]:.0f}px")
    def _start_next_phase(self):
        """رفتن به مرحله بعدی"""
        if self.current_phase == CalibrationPhase.CALIBRATING_EAR:
            self.current_phase = CalibrationPhase.CALIBRATING_DISTANCE
            self.calibration_start_time = time.time()
            self._temp_data = {'ear': [], 'face_bboxes': [], 'eye_distances': [], 'timestamps': []}
            print(f"\n[CALIBRATION] Phase 2: Please stay at {self.reference_distance}cm from camera...")
        elif self.current_phase == CalibrationPhase.CALIBRATING_DISTANCE:
            self.current_phase = CalibrationPhase.CALIBRATING_POSITION
            self.calibration_start_time = time.time()
            self._temp_data = {'ear': [], 'face_bboxes': [], 'eye_distances': [], 'timestamps': []}
            print(f"\n[CALIBRATION] Phase 3: Look left, right, up, down (for position range)...")
    def _complete_calibration(self):
        """اتمام کالیبراسیون"""
        if self.current_profile:
            self.saved_profiles[self.current_driver_id] = self.current_profile
            print("\n" + "=" * 50)
            print(f"CALIBRATION COMPLETE for driver: {self.current_driver_id}")
            print("=" * 50)
        
        self.is_calibrating = False
        self.current_phase = CalibrationPhase.COMPLETE
    
    def _abort_calibration(self):
        """لغو کالیبراسیون"""
        self.is_calibrating = False
        self.current_phase = CalibrationPhase.IDLE
        print("[CALIBRATION] Aborted due to insufficient data!")
    
    def get_current_threshold(self) -> float:
        """دریافت آستانه EAR شخصی‌سازی شده"""
        if self.current_profile:
            return self.current_profile.ear_threshold
        return 0.25
    
    def finalize_calibration(self):
        """دریافت پروفایل نهایی"""
        return self.current_profile