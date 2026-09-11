"""
video_analytics.py — Dự án tổng hợp Phase 5
Detect -> Track -> Đếm theo vùng -> Cảnh báo -> Stream
"""
import cv2, time
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv
import numpy as np


class VideoAnalytics:
    def __init__(self, model_path="yolo11s.pt", zones: dict | None = None):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(track_activation_threshold=0.3,
                                    lost_track_buffer=30)
        self.zones = zones or {}
        self.history = defaultdict(lambda: deque(maxlen=30))   # lưu quỹ đạo
        self.fps_buf = deque(maxlen=30)

    def compute_speed(self, tid, px_per_meter=50, fps=30):
        """Ước lượng tốc độ từ quỹ đạo pixel."""
        pts = self.history[tid]
        if len(pts) < 10:
            return 0.0
        d = np.linalg.norm(np.array(pts[-1]) - np.array(pts[-10]))
        return (d / px_per_meter) / (10 / fps) * 3.6      # km/h

    def process(self, frame):
        t0 = time.perf_counter()
        res = self.model(frame, conf=0.3, iou=0.5, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
        det = self.tracker.update_with_detections(det)

        alerts = []
        for box, tid, cls in zip(det.xyxy, det.tracker_id, det.class_id):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            self.history[tid].append((cx, cy))

            speed = self.compute_speed(tid)
            if speed > 60:
                alerts.append(f"ID#{tid} vượt tốc độ: {speed:.0f} km/h")

            # Kiểm tra xâm nhập vùng cấm
            for name, poly in self.zones.items():
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    alerts.append(f"ID#{tid} xâm nhập vùng '{name}'")

        self.fps_buf.append(1 / (time.perf_counter() - t0))
        return det, alerts, np.mean(self.fps_buf)


if __name__ == "__main__":
    zones = {"khu_cấm": np.array([[100,300],[400,300],[400,600],[100,600]])}
    va = VideoAnalytics(zones=zones)

    cap = cv2.VideoCapture(0)
    box_ann = sv.BoxAnnotator()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        det, alerts, fps = va.process(frame)
        frame = box_ann.annotate(frame, det)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        for i, a in enumerate(alerts[:3]):
            cv2.putText(frame, a, (10, 70 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("analytics", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()