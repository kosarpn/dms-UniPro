import cv2
from ultralytics import YOLO
import torch

# --- مدل خود را اینجا بارگذاری کنید ---
try:
    model = YOLO('yolov8n-face.pt') # مطمئن شوید نام فایل و مسیر آن درست است
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# تنظیم دستگاه (GPU یا CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# باز کردن وب‌کم
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # اجرای مدل روی فریم
    results = model(frame, verbose=False)

    # --- بخش عیب‌یابی ---
    # این بخش به ما نشان می‌دهد داخل متغیر results چه خبر است
    print("-" * 40) # جداکننده برای هر فریم
    if results and results[0].boxes:
        num_detections = len(results[0].boxes)
        print(f"Found {num_detections} object(s).")

        # بررسی وجود و محتوای Keypoints
        if results[0].keypoints is not None:
            print("Keypoints object is NOT None.")
            kpts_data = results[0].keypoints.data
            print(f"  - Keypoints data shape: {kpts_data.shape}") # شکل تنسور را چاپ می‌کند
            if kpts_data.numel() > 0: # بررسی اینکه تنسور خالی نباشد
                 print(f"  - Keypoints data for first object:\n{kpts_data[0]}")
            else:
                 print("  - Keypoints data is empty.")
        else:
            print("!!! Keypoints object IS None. No landmarks detected. !!!")

    else:
        print("No objects detected in this frame.")
    # --- پایان بخش عیب‌یابی ---


    # --- بخش رسم (فعلاً دست نخورده باقی می‌ماند) ---
    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if results[0].keypoints and results[0].keypoints.data is not None and len(results[0].keypoints.data) > i:
                keypoints = results[0].keypoints.data[i]
                for kp_idx, kp in enumerate(keypoints):
                    x, y, conf = kp
                    if conf > 0.5: # فقط نقاط با اطمینان بالا را رسم کن
                        px, py = int(x), int(y)
                        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(kp_idx), (px + 5, py),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow('YOLOv8 Debugging', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
