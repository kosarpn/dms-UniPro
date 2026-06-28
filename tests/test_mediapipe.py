# tests/test_mediapipe.py
"""
تست MediaPipe Face Mesh - نسخه اصلاح شده
"""

import cv2
import mediapipe as mp  # ✅ اینطوری درست است
import time
import numpy as np


def test_mediapipe():
    print("=" * 60)
    print("TEST 1: MediaPipe Face Mesh")
    print("=" * 60)
    
    # ✅ راه‌اندازی MediaPipe به روش درست
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # راه‌اندازی دوربین
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return
    
    print("[INFO] Camera opened successfully")
    print("[INFO] Press 'q' to quit, 's' to save screenshot")
    print("-" * 60)
    
    # آمار
    fps_counter = 0
    fps_time = time.time()
    frame_count = 0
    detection_count = 0
    
    # نقاط کلیدی مهم برای DMS
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        results = face_mesh.process(rgb_frame)
        process_time = (time.time() - start_time) * 1000
        
        detected = False
        
        if results.multi_face_landmarks:
            detected = True
            detection_count += 1
            h, w = frame.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                # کشیدن تمام نقاط
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # برجسته کردن نقاط چشم (رنگ قرمز)
                for idx in LEFT_EYE + RIGHT_EYE:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # برجسته کردن نقاط دهان (رنگ زرد)
                for idx in MOUTH:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # نمایش اطلاعات روی صفحه
        cv2.putText(frame, f"Method: MediaPipe", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Time: {process_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Detected: {detected}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if detected else (0, 0, 255), 2)
        
        if detected:
            cv2.putText(frame, f"Landmarks: 468 points", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
            print(f"[INFO] MediaPipe FPS: {fps}, Detection rate: {detection_count/frame_count*100:.1f}%")
        
        cv2.putText(frame, f"FPS: {fps if 'fps' in locals() else 0}", 
                   (frame.shape[1]-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("MediaPipe Test - 468 Landmarks", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"mediapipe_{frame_count}.png", frame)
            print(f"[INFO] Screenshot saved: mediapipe_{frame_count}.png")
    
    # گزارش نهایی
    print("-" * 60)
    print(f"[RESULT] MediaPipe Test Complete")
    print(f"  - Total frames: {frame_count}")
    print(f"  - Detection rate: {detection_count/frame_count*100:.1f}%")
    if frame_count > 0:
        print(f"  - Average time: ~{(process_time):.1f}ms/frame")
    print("=" * 60)
    
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_mediapipe()