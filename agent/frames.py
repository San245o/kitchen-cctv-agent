import os
import hashlib

import numpy as np


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
        limit = max(1, self.budget.coarse_budget)
        if len(times) > limit:
            idx = np.linspace(0, len(times) - 1, limit).astype(int)
            times = times[idx]
        self.coarse_times = [float(t) for t in times]
        self.coarse_frames: dict[float, np.ndarray] = {}
        self.motion: dict[float, float] = {}

    def _seek(self, t: float) -> np.ndarray:
        self.cap.set(self.cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0 - 5.0))
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def build_coarse(self) -> None:
        prev_small = None
        for t in self.coarse_times:
            if self.budget.out_of_time:
                break
            img = self._decode_for(t, fine=False)
            if img is None:
                continue
            small = self.cv2.resize(img, (160, 90))
            gray = small.mean(axis=2)
            if prev_small is not None:
                self.motion[t] = float(np.abs(gray.astype(int) - prev_small.astype(int)).mean())
            else:
                self.motion[t] = 0.0
            prev_small = gray

    def _decode_for(self, t: float, fine: bool):
        key = round(float(t), 2)
        if key in self.coarse_frames:
            return self.coarse_frames[key]
        allowed = self.budget.spend_fine(1) if fine else self.budget.spend_coarse(1)
        if not allowed:
            return None
        img = self._seek(t)
        if img is None and t > 0:
            img = self._seek(max(0.0, t - 0.25))
        if img is None:
            return None
        self.coarse_frames[key] = img
        return img

    def get_coarse(self, t: float) -> np.ndarray:
        return self.coarse_frames.get(round(float(t), 2))

    def get_fine(self, t: float) -> np.ndarray:
        return self._decode_for(t, fine=True)

    def nearest_coarse_time(self, t: float) -> float | None:
        if not self.coarse_times:
            return None
        arr = np.array(self.coarse_times)
        return float(arr[int(np.argmin(np.abs(arr - float(t))))])

    def motion_sorted(self):
        return sorted(self.motion.items(), key=lambda kv: kv[1], reverse=True)

    def close(self):
        self.cap.release()
