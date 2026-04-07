import cv2
from ultralytics import YOLO
import time

model = YOLO("yolo26m.pt")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

previous_positions = {}
attendance_count = 0

previous_time = time.time()

while True:
    success, frame = cap.read()

    if success:
        results = model.track(source=frame, conf=0.7, persist=True)
        annotated_frame = results[0].plot()

        frame_mid = frame.shape[1] // 2
        cv2.line(annotated_frame, (frame_mid, 0), (frame_mid, frame.shape[0]), (0, 0, 255), 2)

        person_boxes = results[0].boxes[results[0].boxes.cls == 0]

        for box in person_boxes:
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2

            if box.id is None:
                continue

            track_id = int(box.id)

            if track_id in previous_positions:
                prev_x = previous_positions[track_id]

                if prev_x < frame_mid and x_center >= frame_mid:
                    attendance_count += 1
                elif prev_x > frame_mid and x_center <= frame_mid:
                    attendance_count -= 1

            previous_positions[track_id] = x_center

        cv2.putText(annotated_frame, f"Attendance: {attendance_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("Tracking enabled", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
