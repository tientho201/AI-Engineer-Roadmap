import numpy as np

def iou(box1, box2):
    """Intersection over Union. Format: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def nms(boxes, scores, iou_thresh=0.45):
    """Non-Maximum Suppression: bỏ các box trùng nhau, giữ box điểm cao nhất."""
    order = np.argsort(-np.asarray(scores))
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        order = np.array([j for j in order[1:]
                          if iou(boxes[i], boxes[j]) < iou_thresh])
    return keep


boxes  = [[10,10,50,50], [12,12,52,52], [100,100,150,150]]
scores = [0.9, 0.85, 0.95]
print("IoU 2 box đầu:", round(iou(boxes[0], boxes[1]), 3))   # 0.78 -> trùng nhiều
print("Giữ lại index:", nms(boxes, scores))                  # [2, 0]