# tests/test_dlib.py
"""
تست Dlib 68-point - روش Fallback اول
نیاز به دانلود مدل دارد
"""

import cv2
import dlib
import time
import numpy as np
import urllib.request
import bz2
import os
from pathlib import Path


def download_dlib_model():
    """دانلود خودکار مدل Dlib اگر وجود نداشت"""
    model_path = Path("shape_predictor_68_face_landmarks.dat")
    model_path.parent.mkdir(exist_ok=True)
    
    if not model_path.exists():
        print("[INFO] Downloading Dlib model (95MB)...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_path = model_path.parent / "shape_predictor_68_face_landmarks.dat.bz2"
        
        urllib.request.urlretrieve(url, bz2_path)
        
        # Decompress
        with bz2.BZ2File(bz2_path, 'rb') as source:
            with open(model_path, 'wb') as dest:
                dest.write(source.read())
        
        bz2_path.unlink()
        print("[INFO] Model downloaded successfully")
    
    return str(model_path)


def test_dlib():
    print("=" * 60)
    print("TEST 2: Dlib 68-point Landmark Detector")
    print("=" * 60)
    
    # دانلود مدل
    model_path = download_dlib_model()
    
    # راه‌اندازی Dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    
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
    
    # نقاط کلیدی چشم برای Dlib (68-point)
    # چشم چپ: 36-41, چشم راست: 42-47
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))
    JAW = list(range(0, 17))
    MOUTH = list(range(48, 68))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        start_time = time.time()
        faces = detector(gray)
        process_time = (time.time() - start_time) * 1000
        
        detected = False
        
        if len(faces) > 0:
            detected = True
            detection_count += 1
            
            for face in faces:
                # کشیدن bounding box
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # استخراج نقاط کلیدی
                landmarks = predictor(gray, face)
                
                # کشیدن تمام 68 نقطه
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # برجسته کردن نقاط چشم (قرمز)
                for i in LEFT_EYE + RIGHT_EYE:
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # برجسته کردن نقاط دهان (زرد)
                for i in MOUTH:
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # نمایش اطلاعات
        cv2.putText(frame, f"Method: Dlib 68-point", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Time: {process_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Detected: {detected} ({len(faces)} faces)", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0) if detected else (0, 0, 255), 2)
        
        if detected:
            cv2.putText(frame, f"Landmarks: 68 points", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
            print(f"[INFO] Dlib FPS: {fps}, Detection rate: {detection_count/frame_count*100:.1f}%")
        
        cv2.putText(frame, f"FPS: {fps if 'fps' in locals() else 0}", 
                   (frame.shape[1]-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Dlib Test - 68 Landmarks", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"dlib_{frame_count}.png", frame)
            print(f"[INFO] Screenshot saved: dlib_{frame_count}.png")
    
    print("-" * 60)
    print(f"[RESULT] Dlib Test Complete")
    print(f"  - Total frames: {frame_count}")
    print(f"  - Detection rate: {detection_count/frame_count*100:.1f}%")
    print(f"  - Average time: ~{process_time:.1f}ms/frame")
    print("=" * 60)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_dlib()