# test_ear_simple.py
"""
تست ساده EAR با MediaPipe - بدون YOLO
"""

import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_ear(eye_points):
    """محاسبه EAR از 6 نقطه چشم"""
    if len(eye_points) < 6:
        return 0.28
    
    # نقاط به ترتیب: گوشه بیرونی، بالا، بالا-داخلی، گوشه داخلی، پایین-داخلی، پایین
    p1 = eye_points[0]
    p2 = eye_points[1]
    p3 = eye_points[2]
    p4 = eye_points[3]
    p5 = eye_points[4]
    p6 = eye_points[5]
    
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    
    if horizontal == 0:
        return 0.28
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# راه‌اندازی MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# نقاط چشم در MediaPipe (468 نقطه)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# راه‌اندازی دوربین
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=" * 60)
print("SIMPLE EAR TEST - MediaPipe Only")
print("=" * 60)
print("Instructions:")
print("  - Look straight at camera")
print("  - CLOSE your eyes for 2-3 seconds")
print("  - Watch the EAR value drop")
print("  - Press 'q' to quit")
print("=" * 60)

closed_frames = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb)
    
    ear_left = 0.28
    ear_right = 0.28
    ear_avg = 0.28
    
    if results.multi_face_landmarks:
        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0]
        
        # استخراج نقاط چشم چپ
        left_eye_points = []
        for idx in LEFT_EYE_INDICES:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            left_eye_points.append([x, y])
        
        # استخراج نقاط چشم راست
        right_eye_points = []
        for idx in RIGHT_EYE_INDICES:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            right_eye_points.append([x, y])
        
        # محاسبه EAR
        ear_left = calculate_ear(np.array(left_eye_points))
        ear_right = calculate_ear(np.array(right_eye_points))
        ear_avg = (ear_left + ear_right) / 2
        
        # کشیدن نقاط چشم روی صفحه
        for point in left_eye_points + right_eye_points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        
        # تشخیص چشم بسته
        is_closed = ear_avg < 0.22
        
        if is_closed:
            closed_frames += 1
        else:
            closed_frames = max(0, closed_frames - 1)
        
        is_drowsy = closed_frames > 10
        
        # هشدار
        if is_drowsy:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 255), -1)
            cv2.putText(frame, "⚠️ DROWSY! WAKE UP! ⚠️", (w//2-200, h-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            
            if frame_count % 30 == 0:
                print(f"[ALERT] Drowsy! EAR={ear_avg:.3f}")
    
    # نمایش اطلاعات
    cv2.putText(frame, f"LEFT EAR: {ear_left:.3f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"RIGHT EAR: {ear_right:.3f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"AVG EAR: {ear_avg:.3f}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Closed frames: {closed_frames}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # راهنما
    cv2.putText(frame, "Close your eyes to test | Press 'q' to quit", (10, frame.shape[0]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    cv2.imshow("EAR Test - Close Your Eyes", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("\n[INFO] Test completed")