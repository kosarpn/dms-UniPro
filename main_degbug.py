# main_debug.py - نسخه دیباگ برای تست
import cv2
import time
from detectors.hybrid_detector import HybridDetector
from features.ear import EyeAspectRatioAnalyzer

cap = cv2.VideoCapture(0)
detector = HybridDetector()
ear_analyzer = EyeAspectRatioAnalyzer()

print("=" * 50)
print("DEBUG MODE - Close your eyes to see alert")
print("Press 'q' to quit")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # تشخیص چهره
    result = detector.detect(frame)
    
    ear_value = 0.28  # مقدار پیش‌فرض
    
    if result.success and result.landmarks is not None:
        # محاسبه EAR (ساده شده برای تست)
        try:
            # نقاط چشم چپ (برای MediaPipe)
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            left_eye_points = []
            for idx in left_eye_indices:
                if idx < len(result.landmarks):
                    left_eye_points.append(result.landmarks[idx])
            
            if len(left_eye_points) >= 6:
                left_eye = np.array(left_eye_points)
                # محاسبه EAR ساده
                vertical1 = np.linalg.norm(left_eye[1] - left_eye[5])
                vertical2 = np.linalg.norm(left_eye[2] - left_eye[4])
                horizontal = np.linalg.norm(left_eye[0] - left_eye[3])
                if horizontal > 0:
                    ear_value = (vertical1 + vertical2) / (2.0 * horizontal)
        except:
            pass
        
        # به‌روزرسانی تحلیل‌گر
        ear_status = ear_analyzer.update(ear_value)
        
        # تشخیص هشدار
        is_drowsy = ear_status['is_drowsy']
        
        # نمایش روی صفحه
        cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Closed frames: {ear_status['closed_frames']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if is_drowsy:
            # هشدار قرمز
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 255), -1)
            cv2.putText(frame, "⚠️ DROWSY! WAKE UP! ⚠️", (w//2-180, h-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            print(f"[ALERT] Drowsy detected! EAR={ear_value:.3f}")
    
    cv2.imshow("Debug - Close your eyes", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()