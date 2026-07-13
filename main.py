"""
Driver Drowsiness Detection System (DMS)
A real-time driver monitoring system using hybrid face detection,
depth estimation, and personalized calibration.

Author: Your Name
Date: 2025
Version: 2.0.0

Features:
- Hybrid face detection (MediaPipe + YOLO fallback)
- Monocular depth estimation (MiDAS)
- Driver selection with distance + position calibration
- Real-time drowsiness detection (EAR + MAR + Head Pose)
- Kalman filtering for noise reduction
- Personalized calibration for each driver
"""
import cv2
import time
import signal
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from features.ear import calculate_ear
from features.ear import calculate_average_ear
from features.mar import calculate_mar
from features.blink import BlinkRateAnalyzer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# Imports - Organized by layer
# ============================================================================

# Configuration & Utils
from utils.config import Config
from utils.logger import setup_logger, get_logger
from utils.kalman_filter import AdaptiveKalmanFilter

# Calibration Layer
from calibration.calibration_manager import CalibrationManager
from calibration.ear_calibration import EARCalibration
from calibration.distance_calibration import DistanceCalibration
from calibration.position_calibration import PositionCalibration

# Detection Layer
from detectors.hybrid_detector import HybridDetector

# Depth Estimation Layer
from depth.midas_depth import MiDASDepthEstimator
# Driver Selection Layer
from selection.hybrid_selector import HybridDriverSelector
# Feature Extraction Layer
from features.ear import EyeAspectRatioAnalyzer
from features.mar import MouthAspectRatioAnalyzer
from features.head_pose import HeadPoseEstimator

# Alert Layer
from alerts.alert_manager import AlertManager

#import کردن classifier ها
# from ml.classifiers.rule_based_classifier import RuleBasedClassifier
from ml.classifiers.svm_classifier import SVMClassifier
from ml.feature_extractor import FeaturExtractor
# ============================================================================
# Main Application Class
# ============================================================================

class DriverMonitoringSystem:
    """
    Main DMS Application Class
    
    Responsibilities:
    - Orchestrate all system components
    - Handle camera input and video processing
    - Manage calibration workflow
    - Coordinate driver selection
    - Process drowsiness detection
    - Handle graceful shutdown
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Driver Monitoring System
        Args:
            config_path: Path to custom configuration file (optional)
        """
        # ====================================================================
        # Layer 1: Configuration & Logging
        # ====================================================================
        self.config = self._load_configuration(config_path)
        self.logger = setup_logger(
            name="DMS",
            log_level=self.config.LOG_LEVEL,
            log_file=self.config.LOG_FILE
        ) 
        self.logger.info("=" * 60)
        self.logger.info("Driver Drowsiness Detection System Initializing...")
        self.logger.info(f"Version: 2.0.0 | Mode: {self.config.MODE}")
        self.logger.info("=" * 60)
        
        # ====================================================================
        # Layer 2: Detection Components
        # ====================================================================
        self.logger.info("Initializing detection components...")
        self.face_detector = HybridDetector(
            mediapipe_confidence=self.config.MEDIAPIPE_CONFIDENCE,
            yolo_confidence=self.config.YOLO_CONFIDENCE,
            fallback_threshold=self.config.FALLBACK_THRESHOLD
        )
        
        self.depth_estimator = MiDASDepthEstimator(
            model_type=self.config.DEPTH_MODEL_TYPE
        ) if self.config.ENABLE_DEPTH_ESTIMATION else None
        
        # ====================================================================
        # Layer 3: Calibration Components
        # ====================================================================
        self.logger.info("Initializing calibration components...")
        
        self.calibration_manager = CalibrationManager(
            calibration_duration_sec=self.config.CALIBRATION_DURATION,
            reference_distance_cm=self.config.REFERENCE_DISTANCE
        )
        
        self.ear_calibration = EARCalibration()
        self.distance_calibration = DistanceCalibration(
            reference_distance_cm=self.config.REFERENCE_DISTANCE
        )
        self.position_calibration = PositionCalibration(
            frame_width=self.config.FRAME_WIDTH,
            frame_height=self.config.FRAME_HEIGHT
        )
        # ====================================================================
        # Layer 4: Driver Selection
        # ====================================================================
        self.logger.info("Initializing driver selection...")
        self.driver_selector = HybridDriverSelector(
            steering_side=self.config.STEERING_SIDE,
            frame_width=self.config.FRAME_WIDTH,
            frame_height=self.config.FRAME_HEIGHT,
            distance_weight=self.config.DISTANCE_WEIGHT,
            position_weight=self.config.POSITION_WEIGHT,
            tracking_weight=self.config.TRACKING_WEIGHT
        )
        # ====================================================================
        # Layer 5: Feature Analysis
        # ====================================================================
        self.logger.info("Initializing feature analyzers...")
        self.ear_analyzer = EyeAspectRatioAnalyzer(
            history_size=self.config.EAR_HISTORY_SIZE,
            fps=self.config.CAMERA_FPS,          
            drowsy_duration_sec=5.0
        )
        self.mar_analyzer = MouthAspectRatioAnalyzer(
            yawn_threshold=self.config.MAR_THRESHOLD,
        )
        self.blink_analyzer=BlinkRateAnalyzer()
        # self.classifier=RuleBasedClassifier(
        #     ear_threshold=self.config.EAR_THRESHOLD,
        #     mar_threshold=self.config.MAR_THRESHOLD,
        #     drowsy_frame_threshold=self.config.DROWSY_FRAME_THRESHOLD
        # )
        self.classifier=SVMClassifier()
        self.feature_extractor=FeaturExtractor()
        self.head_pose_estimator = HeadPoseEstimator()
        # ====================================================================
        # Layer 6: Signal Processing
        # ====================================================================
        self.logger.info("Initializing signal processing...")
        self.kalman_filter = AdaptiveKalmanFilter()
        # ====================================================================
        # Layer 7: Alert System
        # ====================================================================
        self.logger.info("Initializing alert system...")
        
        self.alert_manager = AlertManager(
            audio_enabled=self.config.AUDIO_ALERTS,
            visual_enabled=self.config.VISUAL_ALERTS,
            alert_cooldown=self.config.ALERT_COOLDOWN
        )
        # ====================================================================
        # Layer 8: State Management
        # ====================================================================
        self.is_running = False
        self.is_calibrated = False
        self.current_driver_id: Optional[str] = None
        self.current_driver_profile = None
        
        # Statistics
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        self.processing_times = []
        # ====================================================================
        # Layer 9: Signal Handlers
        # ====================================================================
        self._setup_signal_handlers()
        
        self.logger.info("✅ All components initialized successfully!")
    
    # ========================================================================
    # Configuration Methods
    # ========================================================================
    
    def _load_configuration(self, config_path: Optional[str]) -> Config:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            # Load custom config
            return Config.from_file(config_path)
        else:
            # Use default config
            return Config()
    
    # ========================================================================
    # Signal Handling
    # ========================================================================
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    # ========================================================================
    # Camera Management
    # ========================================================================
    
    def _initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """Initialize and configure camera"""
        self.logger.info("Initializing camera...")
        
        cap = cv2.VideoCapture(self.config.CAMERA_ID)
        
        if not cap.isOpened():
            self.logger.error("Cannot open camera!")
            return None
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        # Disable auto features for stability
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.logger.info(f"Camera ready: {self.config.FRAME_WIDTH}x{self.config.FRAME_HEIGHT} @ {self.config.CAMERA_FPS}fps")
        return cap
    # ========================================================================
    # Calibration Workflow
    # ========================================================================
    
    def _run_calibration(self, cap: cv2.VideoCapture) -> bool:
        """
        Run the calibration process
        Steps:
        1. EAR calibration (5 seconds)
        2. Distance calibration (5 seconds)
        3. Position calibration (5 seconds)
        """
        self.logger.info("Starting calibration process...")
        driver_id = input("Enter driver ID: ").strip()
        if not driver_id:
            driver_id = f"driver_{int(time.time())}"
        self.calibration_manager.start_calibration(driver_id)
        calibration_frames = []
        start_time = time.time()
        while time.time() - start_time < self.config.CALIBRATION_DURATION * 3:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Display calibration progress
            remaining = self.config.CALIBRATION_DURATION * 3 - (time.time() - start_time)
            cv2.putText(frame, f"CALIBRATION - Please look straight", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time remaining: {remaining:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Calibration", frame)
            
            # Process frame for calibration
            result = self.face_detector.detect(frame)
            if result.landmarks is not None:
                print("LANDMARKS:", result.landmarks.shape)
            if result.success and result.landmarks is not None:
                # Extract EAR from landmarks
                ear_value = self._extract_ear_from_landmarks(result.landmarks)
                
                self.calibration_manager.add_frame(
                    ear=ear_value,
                    face_bbox=result.face_bbox,
                    eye_distance=None
                )
                calibration_frames.append(result.face_bbox)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        
        # Finalize calibration
        profile = self.calibration_manager.finalize_calibration()
        if profile:
            self.current_driver_id = driver_id
            self.current_driver_profile = profile
            self.is_calibrated = True
            
            # Update driver selector with calibrated range
            if hasattr(profile, 'driver_x_range'):
                self.driver_selector.set_driver_range(
                    profile.driver_x_range,
                    profile.driver_y_range
                )
            
            cv2.destroyWindow("Calibration")
            self.logger.info(f"✅ Calibration complete for driver: {driver_id}")
            return True
        else:
            self.logger.error("Calibration failed!")
            return False
    # ========================================================================
    # Frame Processing Pipeline
    # ========================================================================
    def _process_frame(self, frame: np.ndarray) -> dict:
        """
        Complete frame processing pipeline
        Pipeline steps:
        1. Face detection (Hybrid: MediaPipe → YOLO)
        2. Depth estimation (MiDAS)
        3. Driver selection
        4. Feature extraction (EAR, MAR, Head Pose)
        5. Kalman filtering
        6. Drowsiness classification
        7. Alert triggering
        Returns:
            Dictionary with all processing results
        """
        start_time = time.time()
        result = {
            'success': False,
            'face_detected': False,
            'driver_selected': False,
            'ear': 0.0,
            'mar': 0.0,
            'blink_rate':0.0,
            'blink_state':"normal",
            'is_drowsy': False,
            'is_yawning': False,
            'processing_time_ms': 0.0
        }
        # ====================================================================
        # Step 1: Face Detection (Hybrid)
        # ====================================================================
        detection_result = self.face_detector.detect(frame)
        print("METHOD:",detection_result.method_used)
        if not detection_result.success or detection_result.landmarks is None:
            return result
        result['face_detected'] = True
        landmarks = detection_result.landmarks
        face_bbox = detection_result.face_bbox
        # ====================================================================
        # Step 2: Depth Estimation (if enabled)
        # ====================================================================
        depth_map = None
        distance_score = 0.5
        if self.depth_estimator and self.depth_estimator.is_available:
            depth_map = self.depth_estimator.estimate_depth_map(frame)
            if depth_map is not None and face_bbox:
                face_depth = self.depth_estimator.get_face_distance(frame, face_bbox)
                distance_score = 1 - (face_depth / 255.0) if face_depth else 0.5
        # ====================================================================
        # Step 3: Driver Selection
        # ====================================================================
        driver_idx = self.driver_selector.select_driver(
            faces=[face_bbox] if face_bbox else [],
            distances=[distance_score],
            depth_map=depth_map)
        if driver_idx is None:
            return result
        result['driver_selected'] = True
        # ====================================================================
        # Step 4: Feature Extraction
        # ====================================================================
        result['method']=detection_result.method_used.value
        # ------------------------------------------------
        # MEDIAPIPE MODE
        # ------------------------------------------------
        if detection_result.method_used==detection_result.method_used.MEDIAPIPE:
            #EAR
            ear_value=self._extract_ear_from_landmarks(landmarks)
            filtered_ear=self.kalman_filter.update_ear(ear_value)
            result['ear']=filtered_ear
            #MAR
            mar_value=self._extract_mar_from_landmarks(landmarks)
            filtered_mar=self.kalman_filter.update_mar(mar_value)
            result['mar']=filtered_mar
            MOUTH_OUTER = [
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 375, 321, 405, 314, 17, 84, 181, 91, 146
            ]
            mouth_landmarks = landmarks[MOUTH_OUTER]
            #Head pose
            head_pose = self._extract_head_pose_from_landmarks(landmarks)
            # ====================================================================
            # Step 5: Drowsiness Detection
            # ====================================================================
            ear_threshold = (
                self.calibration_manager.get_current_threshold()
                if self.is_calibrated
                else 0.25
            )
            self.ear_analyzer.set_threshold(ear_threshold)
            ear_status = self.ear_analyzer.update(filtered_ear)
            mar_status = self.mar_analyzer.update(filtered_mar, mouth_landmarks)
            blink_status = self.blink_analyzer.update(
                ear_status["is_closed"]
            )

            # prediction = self.classifier.predict(
            #     ear_status=ear_status,
            #     mar_status=mar_status,
            #     blink_status=blink_status,
            # )
            dataset_features = self.feature_extractor.extract(frame)
            if dataset_features is None:
                return result
            prediction = self.classifier.predict(
                dataset_features)
            # result["blink_rate"] = prediction["blink_rate"]
            result["is_drowsy"] = prediction["is_drowsy"]
            # result["is_yawning"] = prediction["is_yawning"]
            result["confidence"] = prediction["confidence"]
            # result["level"] = prediction["level"]
            # ear_drowsy=(
            #     ear_status['is_drowsy']
            # )
            # is_drowsy=ear_drowsy 
            print(f"EAR : {filtered_ear:.3f}")
            print(f"MAR : {filtered_mar:.3f}")
            # print(f"Blink Rate : {prediction['blink_rate']:.1f}")
            print(f"Drowsy : {prediction['is_drowsy']}")
            # print(f"Yawn : {prediction['is_yawning']}")
            print(f"Confidence : {prediction['confidence']:.2f}")
            # print(f"Level : {prediction['level']}")
            print("----------------------------")
        # ------------------------------------------------
        # YOLO FALLBACK MODE
        # ------------------------------------------------
        else:
            result['ear']=0.0
            result['mar']=0.0
            result['is_drowsy']=False
            result['is_yawning']=False
        # ====================================================================
        # Step 6: Alert Management
        # ====================================================================
        if result["is_drowsy"]:
            self.alert_manager.trigger_drowsy_alert(
                confidence=result.get("confidence", 0.8),
                ear_value=result.get("ear", 0.0)
            )

        elif result["is_yawning"]:
            self.alert_manager.trigger_yawn_alert()
        # ====================================================================
        # Step 7: Update Statistics
        # ====================================================================
        result['processing_time_ms'] = (time.time() - start_time) * 1000
        result['success'] = True
        return result
    # ========================================================================
    # Feature Extraction Helpers
    # ========================================================================
    def _extract_ear_from_landmarks(self, landmarks: np.ndarray) -> float:
        """Extract EAR value from face landmarks"""
        """MediaPipe 468-point indices"""
        LEFT_EYE  = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        try:
            left_eye=landmarks[LEFT_EYE]
            right_eye=landmarks[RIGHT_EYE]
            return calculate_average_ear(left_eye,right_eye)
        except Exception as e:
            self.logger.warning(f"EAR extraction failed: {e}")
            return 0.0 
    def _extract_mar_from_landmarks(self, landmarks: np.ndarray) -> float:
        """Extract MAR value from face landmarks"""
        if landmarks is None or len(landmarks)<468:
            return None
#         MOUTH_INDICES = [
#         61,   # left corner
#         13,   # upper inner lip
#         14,   # lower inner lip
#         291   # right cornerّ
# ]
          
        # MediaPipe MOUTH_OUTER indices (20 points)
        MOUTH_OUTER = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # Upper lip
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146  # Lower lip
        ]
        try:
            mouth_landmarks=landmarks[MOUTH_OUTER]
            return calculate_mar(mouth_landmarks)
        except Exception as e:
            self.logger.error(f"MAR extraction failed:{e}")
            return 0.0
    def _extract_head_pose_from_landmarks(self, landmarks: np.ndarray) -> dict:
        """Extract head pose (yaw, pitch, roll) from landmarks"""
        try:
            # Placeholder
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        except:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def _visualize(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw results on frame
        
        Displays:
        - Face bounding box (green for driver, yellow for others)
        - EAR and MAR values
        - Drowsiness/yawn alerts
        - Calibration status
        - FPS counter
        """
        display = frame.copy()
        h, w = display.shape[:2]
        
        # ====================================================================
        # Status Panel (Top-left)
        # ====================================================================
        panel_y = 0
        cv2.rectangle(display, (0, panel_y), (250, 120), (0, 0, 0), -1)
        cv2.rectangle(display, (0, panel_y), (250, 120), (0, 255, 0), 1)
        
        # Driver status
        driver_status = f"Driver: {self.current_driver_id or 'Unknown'}"
        cv2.putText(display, driver_status, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calibration status
        calib_status = "Calibrated" if self.is_calibrated else "Not Calibrated"
        calib_color = (0, 255, 0) if self.is_calibrated else (0, 0, 255)
        cv2.putText(display, f"Calibration: {calib_status}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, calib_color, 1)
        
        # EAR value
        cv2.putText(display, f"EAR: {result['ear']:.3f}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # MAR value
        cv2.putText(display, f"MAR: {result['mar']:.3f}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(
            display,f"Blink Rate:{result['blink_rate']:.1f}/min ({result['blink_state']})",
            (20,180),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255,255,0),
            2
        )
        # Processing time
        cv2.putText(display, f"Time: {result['processing_time_ms']:.1f}ms", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # ====================================================================
        # Alert Display (Center-top)
        # ====================================================================
        if result['is_drowsy']:
            cv2.rectangle(display, (w//2-200, 0), (w//2+200, 50), (0, 0, 255), -1)
            cv2.putText(display, "⚠️ DROWSY! WAKE UP! ⚠️", (w//2-180, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif result['is_yawning']:
            cv2.rectangle(display, (w//2-150, 0), (w//2+150, 50), (0, 255, 255), -1)
            cv2.putText(display, "😮 YAWN DETECTED", (w//2-130, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # ====================================================================
        # Info Panel (Top-right)
        # ====================================================================
        fps = self.fps_counter if time.time() - self.fps_time < 1.0 else 0
        cv2.putText(display, f"FPS: {fps}", (w-100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        method = "MediaPipe" if result.get('method') == 'mediapipe' else "YOLO"
        cv2.putText(display, f"Method: {method}", (w-100, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # ====================================================================
        # Confidence Bar (Bottom)
        # ====================================================================
        confidence = result.get('confidence', 0.5)
        bar_width = int(w * confidence)
        cv2.rectangle(display, (0, h-10), (bar_width, h), (0, 255, 0), -1)
        cv2.rectangle(display, (0, h-10), (w, h), (255, 255, 255), 1)
        
        return display
    
    # ========================================================================
    # Statistics & Logging
    # ========================================================================
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()
            
            if self.frame_count % 100 == 0:
                avg_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
                self.logger.info(f"Stats - Frame: {self.frame_count}, FPS: {fps}, Avg Time: {avg_time:.1f}ms")
    
    # ========================================================================
    # Main Loop
    # ========================================================================
    
    def run(self):
        """Main application loop"""
        # Initialize camera
        cap = self._initialize_camera()
        if cap is None:
            self.logger.error("Failed to initialize camera!")
            return
        # Run calibration if needed
        if not self.is_calibrated:
            self.logger.info("System not calibrated. Starting calibration...")
            if not self._run_calibration(cap):
                self.logger.warning("Calibration skipped or failed. Using default settings.")
        self.is_running = True
        self.logger.info("🚀 System running. Press 'q' to quit, 'c' to recalibrate")
        # Main processing loop
        while self.is_running:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("Failed to read frame")
                break
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            # Process frame
            result = self._process_frame(frame)
            # Visualize results
            display = self._visualize(frame, result)
            # Show frame
            cv2.imshow("Driver Monitoring System", display)
            # Update FPS
            self._update_fps()
            # Store processing time
            if result['processing_time_ms'] > 0:
                self.processing_times.append(result['processing_time_ms'])
                if len(self.processing_times) > 1000:
                    self.processing_times.pop(0)
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.logger.info("User requested shutdown")
                break
            elif key == ord('c'):
                self.logger.info("Recalibration requested")
                self.is_calibrated = False
                self._run_calibration(cap)
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.shutdown()
    # ========================================================================
    # Shutdown
    # ========================================================================
    def shutdown(self):
        """Graceful shutdown of all components"""
        self.logger.info("Shutting down system...")
        self.is_running = False
        
        # Close detectors
        if hasattr(self.face_detector, 'close'):
            self.face_detector.close()
        
        # Save calibration data
        if self.current_driver_profile:
            self.logger.info(f"Saving profile for driver: {self.current_driver_id}")
        
        # Log final statistics
        self.logger.info("=" * 60)
        self.logger.info(f"System Statistics:")
        self.logger.info(f"  - Total frames processed: {self.frame_count}")
        self.logger.info(f"  - Average processing time: {np.mean(self.processing_times):.1f}ms" if self.processing_times else "  - N/A")
        self.logger.info(f"  - Calibrated: {self.is_calibrated}")
        self.logger.info("=" * 60)
        self.logger.info("✅ System shutdown complete")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Application entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Run application
    try:
        app = DriverMonitoringSystem(config_path=args.config)
        
        if args.debug:
            app.config.LOG_LEVEL = "DEBUG"
        
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()