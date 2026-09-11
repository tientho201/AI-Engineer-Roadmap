import numpy as np
from Demo4 import iou

def average_precision(preds, gts, iou_thresh=0.5):
    """
    preds: list của (score, box), đã sắp giảm dần theo score
    gts:   list các box ground truth
    """
    preds = sorted(preds, key=lambda x: -x[0])
    matched = set()
    tp, fp = [], []

    for score, box in preds:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = iou(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh:
            tp.append(1); fp.append(0); matched.add(best_j)
        else:
            tp.append(0); fp.append(1)

    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall    = tp / max(len(gts), 1)
    precision = tp / np.maximum(tp + fp, 1e-9)

    # AP = diện tích dưới đường precision-recall
    ap = 0.0
    for t in np.arange(0, 1.01, 0.1):
        p = precision[recall >= t].max() if (recall >= t).any() else 0
        ap += p / 11
    return ap


preds = [(0.95, [10,10,50,50]), (0.80, [100,100,140,140]), (0.60, [200,200,220,220])]
gts   = [[12,12,52,52], [100,100,150,150]]
print("AP@0.5 =", round(average_precision(preds, gts), 4))

# mAP@0.5      = AP trung bình trên mọi lớp, tại IoU 0.5
# mAP@0.5:0.95 = trung bình thêm qua IoU từ 0.5 đến 0.95 (khắt khe hơn, chuẩn COCO)