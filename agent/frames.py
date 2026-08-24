import os
import hashlib
from collections import OrderedDict

import numpy as np


THUMB_W, THUMB_H = 160, 90


def _imwrite_safe(path, img):
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        buf.tofile(path)


def _imread_safe(path):
    import cv2

    if not os.path.exists(path):
        return None
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class FrameStore:
    def __init__(self, video_path: str, budget, cfg: dict):
        import cv2

        self.cv2 = cv2
        self.path = video_path
        self.budget = budget
        self.cfg = cfg
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.n_frames <= 0:
            self.n_frames = int(self.fps * 3600 * 4)
        self.duration = max(self.n_frames / self.fps, 1e-3)
        base = os.path.splitext(os.path.basename(video_path))[0]
        tag = hashlib.md5(video_path.encode()).hexdigest()[:8]
        self.cache_dir = os.path.join("cache", f"{base}_{tag}")
        os.makedirs(self.cache_dir, exist_ok=True)
        stride = float(cfg.get("coarse_stride_seconds", 3.0))
        times = np.arange(0.0, self.duration, stride)
        limit = max(1, budget.coarse_budget)
        if len(times) > limit:
            idx = np.linspace(0, len(times) - 1, limit).astype(int)
            times = times[idx]
        self.coarse_times = [float(t) for t in times]
        self.thumbs: dict[float, np.ndarray] = {}
        self.motion: dict[float, float] = {}
        self._lru: OrderedDict[float, np.ndarray] = OrderedDict()
        self._lru_max = 24

    def _disk_path(self, t: float) -> str:
        return os.path.join(self.cache_dir, f"f_{round(float(t), 2):.2f}.jpg")

    def _seek(self, t: float):
        self.cap.set(self.cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0 - 5.0))
        ok, frame = self.cap.read()
        return frame if ok else None

    def _remember(self, t: float, img: np.ndarray) -> np.ndarray:
        key = round(float(t), 2)
        self._lru[key] = img
        self._lru.move_to_end(key)
        while len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)
        return img

    def _load_cached(self, t: float):
        img = _imread_safe(self._disk_path(t))
        if img is not None:
            return self._remember(t, img)
        return None

    def _decode(self, t: float, fine: bool):
        allowed = self.budget.spend_fine(1) if fine else self.budget.spend_coarse(1)
        if not allowed:
            return None
        img = self._seek(t)
        if img is None and t > 0:
            img = self._seek(max(0.0, t - 0.25))
        if img is None:
            return None
        _imwrite_safe(self._disk_path(t), img)
        return self._remember(t, img)

    def build_coarse(self) -> None:
        prev_thumb = None
        for t in self.coarse_times:
            if self.budget.out_of_time:
                break
            img = self._decode(t, fine=False)
            if img is None:
                continue
            small = self.cv2.resize(img, (THUMB_W, THUMB_H))
            gray = small.mean(axis=2).astype(np.uint8)
            self.thumbs[round(float(t), 2)] = gray
            if prev_thumb is not None:
                self.motion[round(float(t), 2)] = float(
                    np.abs(gray.astype(int) - prev_thumb.astype(int)).mean()
                )
            else:
                self.motion[round(float(t), 2)] = 0.0
            prev_thumb = gray

    def get_coarse(self, t: float):
        key = round(float(t), 2)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        img = self._load_cached(key)
        if img is not None:
            return img
        if key in set(round(x, 2) for x in self.coarse_times):
            return self._decode(key, fine=False)
        return None

    def get_fine(self, t: float):
        key = round(float(t), 2)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        img = self._load_cached(key)
        if img is not None:
            return img
        return self._decode(key, fine=True)

    def nearest_coarse_time(self, t: float):
        if not self.coarse_times:
            return None
        arr = np.array(self.coarse_times)
        return float(arr[int(np.argmin(np.abs(arr - float(t))))])

    def zone_motion(self, zone_norm):
        if not zone_norm or len(self.thumbs) < 2:
            return dict(self.motion)
        zx1 = int(max(0.0, min(1.0, zone_norm[0])) * THUMB_W)
        zy1 = int(max(0.0, min(1.0, zone_norm[1])) * THUMB_H)
        zx2 = int(max(1.0, min(1.0, zone_norm[2])) * THUMB_W)
        zy2 = int(max(1.0, min(1.0, zone_norm[3])) * THUMB_H)
        times = sorted(self.thumbs.keys())
        out = {}
        prev_t = None
        for t in times:
            if prev_t is None:
                out[t] = 0.0
            else:
                d = np.abs(
                    self.thumbs[t][:, zx1:zx2].astype(int)
                    - self.thumbs[prev_t][:, zx1:zx2].astype(int)
                )
                out[t] = float(d.mean())
            prev_t = t
        return out

    def rank_candidates(self, zone_norm=None, top_k=None):
        top_k = top_k or 12
        series = self.zone_motion(zone_norm)
        scored = sorted(series.items(), key=lambda kv: kv[1], reverse=True)
        return [float(t) for t, _s in scored[:top_k]]

    def close(self):
        self.cap.release()
