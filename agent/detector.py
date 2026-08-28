import numpy as np
import time


class Detector:
    def __init__(self, cfg: dict, budget):
        self.cfg = cfg["models"]["detector"]
        self.budget = budget
        self.model = None
        self.available = False
        self.load_error = None
        self._attempted = False

    @property
    def name(self):
        return self.cfg["weights"]

    def ensure_loaded(self) -> bool:
        if self._attempted or self.available:
            return self.available
        self._attempted = True
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.cfg["weights"])
            self.available = True
        except Exception as e:
            self.load_error = str(e)
        return self.available

    def detect_persons(self, img: np.ndarray):
        if not self.ensure_loaded():
            return []
        if self.budget is None or self.budget.exhausted:
            return []
        started = time.perf_counter()
        try:
            res = self.model.predict(
                img,
                classes=[0],
                conf=float(self.cfg["conf"]),
                iou=float(self.cfg["iou"]),
                imgsz=int(self.cfg["imgsz"]),
                verbose=False,
            )[0]
            self.budget.log_call(
                self.name,
                "detector_person",
                started_at=started,
                duration_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            self.budget.log_call(
                self.name,
                "detector_person",
                started_at=started,
                duration_seconds=time.perf_counter() - started,
                success=False,
                error=exc,
            )
            return []
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out


def boxes_in_zone(boxes, zone_norm, img_w, img_h):
    if not zone_norm:
        return boxes
    zx1, zy1, zx2, zy2 = zone_norm
    keep = []
    for x1, y1, x2, y2, c in boxes:
        cx = (x1 + x2) / 2 / max(img_w, 1)
        cy = (y1 + y2) / 2 / max(img_h, 1)
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            keep.append((x1, y1, x2, y2, c))
    return keep


def head_box(box, img_w, img_h):
    x1, y1, x2, y2, c = box
    w = x2 - x1
    h = y2 - y1
    hy2 = min(y2, y1 + 0.45 * h)
    pad_x = 0.15 * w
    return (
        max(0, x1 - pad_x) / img_w,
        max(0, y1 - 0.05 * h) / img_h,
        min(img_w, x2 + pad_x) / img_w,
        hy2 / img_h,
    )
