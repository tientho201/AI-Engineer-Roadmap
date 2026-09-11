import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolo11s.pt")
tracker = sv.ByteTrack()                 # gán ID ổn định cho từng đối tượng qua các frame
box_annotator   = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()    # vẽ đường di chuyển

# Đếm người đi qua một đường kẻ
line = sv.LineZone(start=sv.Point(0, 400), end=sv.Point(1280, 400))
line_annotator = sv.LineZoneAnnotator()

cap = cv2.VideoCapture("traffic.mp4")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, conf=0.3, iou=0.5, classes=[0], verbose=False)[0]  # 0 = person
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)

    labels = [f"#{tid} {conf:.2f}"
              for tid, conf in zip(detections.tracker_id, detections.confidence)]

    frame = trace_annotator.annotate(frame, detections)
    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels)
    line.trigger(detections)
    frame = line_annotator.annotate(frame, line)

    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

print(f"Vào: {line.in_count} | Ra: {line.out_count}")
cap.release(); cv2.destroyAllWindows()