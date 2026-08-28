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
        cache_root = cfg.get("cache_dir", "cache")
        self.cache_dir = os.path.join(cache_root, f"{base}_{tag}")
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
        t = max(0.0, min(float(t), self.duration))
        return os.path.join(self.cache_dir, f"f_{round(t, 2):.2f}.jpg")

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
        t = max(0.0, min(float(t), self.duration))
        allowed = (
            self.budget.spend_fine(1, timestamp=t)
            if fine
            else self.budget.spend_coarse(1, timestamp=t)
        )
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
        key = round(max(0.0, min(float(t), self.duration)), 2)
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
        key = round(max(0.0, min(float(t), self.duration)), 2)
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
        zx2 = max(zx1 + 1, int(max(0.0, min(1.0, zone_norm[2])) * THUMB_W))
        zy2 = max(zy1 + 1, int(max(0.0, min(1.0, zone_norm[3])) * THUMB_H))
        times = sorted(self.thumbs.keys())
        out = {}
        prev_t = None
        for t in times:
            if prev_t is None:
                out[t] = 0.0
            else:
                d = np.abs(
                    self.thumbs[t][zy1:zy2, zx1:zx2].astype(int)
                    - self.thumbs[prev_t][zy1:zy2, zx1:zx2].astype(int)
                )
                out[t] = float(d.mean())
            prev_t = t
        return out

    def rank_candidates(
        self,
        zone_norm=None,
        top_k=None,
        semantic_scores=None,
        semantic_weight=0.75,
        min_gap_seconds=None,
    ):
        top_k = top_k or 12
        series = self.zone_motion(zone_norm)
        times = sorted(series)
        if not times:
            return []

        motion = np.asarray([series[t] for t in times], dtype=np.float32)
        lo, hi = np.percentile(motion, [5, 95]) if len(motion) > 1 else (0.0, 1.0)
        motion = np.clip((motion - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)

        if semantic_scores:
            semantic = np.asarray([semantic_scores.get(round(float(t), 2), 0.0) for t in times], dtype=np.float32)
            slo, shi = np.percentile(semantic, [5, 95]) if len(semantic) > 1 else (0.0, 1.0)
            semantic = np.clip((semantic - slo) / max(float(shi - slo), 1e-6), 0.0, 1.0)
            fused = semantic_weight * semantic + (1.0 - semantic_weight) * motion
        else:
            fused = motion

        ranked = sorted(zip(times, fused.tolist()), key=lambda kv: kv[1], reverse=True)
        gap = (
            float(min_gap_seconds)
            if min_gap_seconds is not None
            else max(1.0, float(self.cfg.get("coarse_stride_seconds", 3.0)) * 1.5)
        )
        selected = []
        for t, _score in ranked:
            if all(abs(float(t) - prior) >= gap for prior in selected):
                selected.append(float(t))
            if len(selected) >= top_k:
                break
        return selected

    def close(self):
        self.cap.release()
