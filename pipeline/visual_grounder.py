"""
Frame extraction with high-contrast timestamp burn-in and ROI cropping.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def burn_timestamp_overlay(image: np.ndarray, timestamp_sec: float) -> np.ndarray:
    """
    Burns an OCR-legible, high-contrast timestamp overlay into the top-left corner.
    Uses a solid black background banner with bright neon text '[T=XX.XXs]'
    to guarantee Gemini's OCR reads it perfectly regardless of scene brightness.
    """
    img = image.copy()
    h, w = img.shape[:2]

    text = f"[T={timestamp_sec:.2f}s]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.0, w / 800.0))
    thickness = 2 if scale >= 0.7 else 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Draw solid black background rectangle for 100% contrast
    box_pad = 6
    cv2.rectangle(
        img,
        (10, 10),
        (10 + text_w + box_pad * 2, 10 + text_h + box_pad * 2),
        (0, 0, 0),
        -1
    )

    # Draw bright yellow text (BGR: 0, 255, 255)
    text_org = (10 + box_pad, 10 + text_h + box_pad)
    cv2.putText(img, text, text_org, font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return img


def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    roi: Optional[Tuple[float, float, float, float]] = None,
    max_width: int = 768,
) -> List[Tuple[float, np.ndarray]]:
    """
    Extracts frames precisely at given timestamps (seconds) using millisecond seeking.
    roi: optional (ymin, xmin, ymax, xmax) normalized 0.0-1.0
    Returns list of (timestamp_sec, processed_frame_bgr).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / video_fps if video_fps > 0 else 0.0

    frames_out = []
    # Deduplicate and sort timestamps
    sorted_ts = sorted(list(set(timestamps)))

    for ts in sorted_ts:
        if ts < 0 or (duration_sec > 0 and ts > duration_sec):
            continue

        # Seek to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]

        # Apply ROI crop if requested
        if roi:
            ymin, xmin, ymax, xmax = roi
            y1 = max(0, int(ymin * h))
            y2 = min(h, int(ymax * h))
            x1 = max(0, int(xmin * w))
            x2 = min(w, int(xmax * w))
            if y2 > y1 and x2 > x1:
                frame = frame[y1:y2, x1:x2]
                h, w = frame.shape[:2]

        # Resize width if exceeding max_width to keep token cost optimal
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (max_width, int(h * scale)))

        # Burn-in high-contrast timestamp
        burned_frame = burn_timestamp_overlay(frame, ts)
        frames_out.append((ts, burned_frame))

    cap.release()
    return frames_out


def extract_window_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = 1.0,
    max_frames: int = 15,
) -> List[Tuple[float, np.ndarray]]:
    """
    Extracts frames over a temporal window [start_sec, end_sec] at target fps.
    """
    if end_sec <= start_sec:
        end_sec = start_sec + 2.0

    duration = end_sec - start_sec
    needed_frames = int(duration * fps) + 1
    if needed_frames > max_frames:
        fps = max_frames / duration

    step = 1.0 / max(0.1, fps)
    ts_list = []
    curr = start_sec
    while curr <= end_sec and len(ts_list) < max_frames:
        ts_list.append(round(curr, 2))
        curr += step

    return extract_frames_at_timestamps(video_path, ts_list)
