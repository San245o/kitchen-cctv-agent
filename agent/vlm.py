import json
import re

import numpy as np


def crop_upscale(img: np.ndarray, zone_norm=None, scale: int = 3) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    if zone_norm:
        x1, y1, x2, y2 = zone_norm
        x1, x2 = int(max(0, x1) * w), int(min(1, x2) * w)
        y1, y2 = int(max(0, y1) * h), int(min(1, y2) * h)
        crop = img[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
    else:
        crop = img
    if scale != 1:
        crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    return crop


def to_pil(img: np.ndarray):
    from PIL import Image

    return Image.fromarray(img[:, :, ::-1])


class VLM:
    YES_NO_PROMPT = (
        "{q}\nAnswer with exactly one word: YES, NO, or UNSURE. "
        "Use UNSURE only if the image does not show enough detail to decide."
    )
    VERIFY_PROMPT = (
        "These are consecutive frames from a fixed kitchen camera. "
        "Question: {q}\nAnswer with exactly one word: YES, NO, or UNSURE."
    )
    READ_PROMPT = (
        "Read all visible text in this image. "
        "If no text is clearly readable, reply exactly: NOT_READABLE.\n"
        "Reply with the text only."
    )
    LOCATE_PROMPT = (
        "Locate the {phrase} in this image. "
        'Reply with JSON only: {{"box_2d": [ymin, xmin, ymax, xmax]}} '
        "using pixel coordinates of this exact image. If not visible, reply {{}}"
    )

    def __init__(self, cfg: dict, budget):
        self.cfg = cfg["models"]["vlm"]
        self.budget = budget
        self.model_id = self.cfg["primary"]
        self.model = None
        self.processor = None
        self.device = None
        self.available = False
        self.load_error = None
        self._attempted = False

    def use_alternate(self):
        self.cfg["primary"], self.cfg["alternate"] = self.cfg["alternate"], self.cfg["primary"]
        self.model_id = self.cfg["primary"]
        self._attempted = False
        self.available = False

    def ensure_loaded(self) -> bool:
        if self._attempted or self.available:
            return self.available
        self._attempted = True
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            mid = self.model_id
            self.processor = AutoProcessor.from_pretrained(mid)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception as e:
            err = f"{self.model_id}: {e}"
            self.load_error = err if not self.load_error else f"{self.load_error} | {err}"
            alt = self.cfg.get("alternate")
            if alt and alt != self.model_id:
                self.model_id = alt
                self._attempted = False
                return self.ensure_loaded()
        return self.available

    def _generate(self, images, prompt: str):
        if not images:
            return None
        if not self.ensure_loaded():
            return None
        try:
            import torch

            content = [{"type": "image", "image": to_pil(im)} for im in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[to_pil(im) for im in images], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=int(self.cfg["max_new_tokens"]),
                    do_sample=False,
                )
            trimmed = out[:, inputs["input_ids"].shape[1]:]
            resp = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            self.budget.log_call(self.model_id, "vlm_generate")
            return resp
        except Exception:
            return None

    def yes_no(self, img, question: str):
        resp = self._generate([img], self.YES_NO_PROMPT.format(q=question))
        return parse_yes_no(resp)

    def verify_window(self, frames, question: str):
        frames = frames[: int(self.cfg.get("max_frames_per_call", 4))]
        resp = self._generate(frames, self.VERIFY_PROMPT.format(q=question))
        return parse_yes_no(resp)

    def read_text(self, img):
        resp = self._generate([img], self.READ_PROMPT)
        if not resp:
            return None, 0.0
        if "NOT_READABLE" in resp.upper() or len(resp.strip()) < 2:
            return None, 0.3
        clean = resp.strip().strip('"')
        return clean, 0.75

    def locate(self, img, phrase: str):
        h, w = img.shape[:2]
        resp = self._generate([img], self.LOCATE_PROMPT.format(phrase=phrase))
        if not resp:
            return None
        m = re.search(r"\{.*\}", resp, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            box = data.get("box_2d")
            if not box or len(box) != 4:
                return None
            ymin, xmin, ymax, xmax = [float(v) for v in box]
            scale = 1000.0 if max(abs(v) for v in box) > 2.0 else 1.0
            if scale > 1:
                ymin, xmin, ymax, xmax = [v / scale for v in (ymin, xmin, ymax, xmax)]
            return (
                max(0.0, min(xmin / w if scale == 1 else xmin, 1.0)),
                max(0.0, ymin),
                min(1.0, xmax / w if scale == 1 else xmax),
                min(1.0, ymax),
            )
        except Exception:
            return None


def parse_yes_no(resp):
    if not resp:
        return None, 0.0
    head = re.split(r"[\s.,!]", resp.strip().upper(), 1)[0]
    mapping = {"YES": ("yes", 0.85), "NO": ("no", 0.85), "UNSURE": (None, 0.4)}
    if head in mapping:
        return mapping[head]
    if "UNSURE" in resp.upper():
        return None, 0.4
    if "YES" in resp.upper():
        return "yes", 0.6
    if "NO" in resp.upper():
        return "no", 0.6
    return None, 0.3


def consensus(votes, min_agree=2):
    valid = [(v, c) for v, c in votes if v is not None]
    if len(valid) < min_agree:
        return None, 0.0
    counts = {}
    conf_sum = {}
    for v, c in valid:
        counts[v] = counts.get(v, 0) + 1
        conf_sum[v] = conf_sum.get(v, 0.0) + c
    best = max(counts, key=lambda k: counts[k])
    if counts[best] < min_agree:
        return None, 0.0
    n = len(valid)
    agreement = counts[best] / n
    conf = (conf_sum[best] / counts[best]) * (0.6 + 0.4 * agreement)
    return best, round(min(conf, 0.95), 2)


def motion_scores(gray_small_list):
    if len(gray_small_list) < 2:
        return []
    prev = gray_small_list[0]
    out = []
    for g in gray_small_list[1:]:
        out.append(float(np.abs(g.astype(int) - prev.astype(int)).mean()))
        prev = g
    return out
