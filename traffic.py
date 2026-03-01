import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Load YOLO Model
model = YOLO("yolov8n.pt")   # lightweight model

video = cv2.VideoCapture("traffic.mp4")     # Open Video

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

def classify_density(count):        # Traffic Classification Logic
    if count < 10:
        return "Low", 30
    elif count < 25:
        return "Medium", 60
    else:
        return "High", 90

traffic_data = []       # Data Storage

# Vehicle classes (YOLO COCO IDs)
vehicle_classes = [2, 3, 5, 7]  
# 2=car, 3=motorcycle, 5=bus, 7=truck

while True:                 # Main Loop
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)

    vehicle_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            if cls in vehicle_classes:
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                cv2.putText(frame,
                            f"Vehicle {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2)

    # Traffic Logic
    density, signal_time = classify_density(vehicle_count)

    # Display Info
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Traffic: {density}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Signal Time: {signal_time}s", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Save Log
    traffic_data.append([
        datetime.now().strftime("%H:%M:%S"),
        vehicle_count,
        density,
        signal_time
    ])

    cv2.imshow("Smart Traffic Management (YOLO)", frame)

    if cv2.waitKey(1) == 27:
        break

# Save CSV Log
df = pd.DataFrame(traffic_data,
                  columns=["Time", "Vehicle Count",
                           "Traffic Density", "Signal Time"])

df.to_csv("traffic_log.csv", index=False)

print("Traffic data saved successfully")

video.release()
cv2.destroyAllWindows()