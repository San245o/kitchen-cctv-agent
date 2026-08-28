import json
import gc
import re
import time

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
        "These frames are ordered from oldest to newest and come from a fixed kitchen camera. "
        "Use visible changes across the sequence to judge actions or state transitions; do not infer "
        "an action from an unchanged single-frame state. Question: {q}\n"
        "Answer with exactly one word: YES, NO, or UNSURE."
    )
    READ_PROMPT = (
        "Read all visible text in this image. "
        "If no text is clearly readable, reply exactly: NOT_READABLE.\n"
        "Reply with the text only."
    )
    READ_TARGETED_PROMPT = (
        "Look at this kitchen camera frame. Question about on-screen text: {q}\n"
        "If the requested text is clearly readable, reply with just the text.\n"
        "If it is not visible or not readable, reply exactly: NOT_READABLE."
    )
    LOCATE_PROMPT = (
        "Locate the {phrase} in this image. "
        'Reply with JSON only: {{"box_2d": [ymin, xmin, ymax, xmax]}} '
        "using normalized integer coordinates from 0 to 1000. If not visible, reply {{}}"
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

    def set_budget(self, budget):
        self.budget = budget

    def _try_load(self, mid):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(mid)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        self.model = model
        self.device = device

    def ensure_loaded(self) -> bool:
        if self._attempted or self.available:
            return self.available
        self._attempted = True
        mid = self.cfg["primary"]
        try:
            self._try_load(mid)
            self.model_id = mid
            self.available = True
            return True
        except Exception as e:
            self.load_error = f"{mid}: {e}"
            self.model = None
            self.processor = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return False

    def _generate(self, images, prompt: str):
        if not images:
            return None
        if self.budget is None or self.budget.exhausted:
            return None
        if not self.ensure_loaded():
            return None
        timeout_s = float(self.cfg.get("call_timeout_seconds", 180))
        started = time.perf_counter()
        response = None
        error = None
        try:
            response = self._generate_inner(images, prompt, timeout_s)
            return response
        except Exception as exc:
            error = exc
            return None
        finally:
            self.budget.log_call(
                self.model_id,
                "vlm_generate",
                started_at=started,
                duration_seconds=time.perf_counter() - started,
                success=response is not None,
                error=error,
            )

    def _generate_inner(self, images, prompt: str, timeout_s: float):
        import torch

        content = [{"type": "image", "image": to_pil(im)} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[to_pil(im) for im in images], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg["max_new_tokens"]),
                max_time=timeout_s,
                do_sample=False,
            )
        trimmed = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def yes_no(self, img, question: str):
        resp = self._generate([img], self.YES_NO_PROMPT.format(q=question))
        return parse_yes_no(resp)

    def verify_window(self, frames, question: str):
        frames = frames[: int(self.cfg.get("max_frames_per_call", 4))]
        resp = self._generate(frames, self.VERIFY_PROMPT.format(q=question))
        return parse_yes_no(resp)

    def read_text(self, img):
        resp = self._generate([img], self.READ_PROMPT)
        return parse_text_response(resp)

    def read_targeted(self, img, question: str):
        resp = self._generate([img], self.READ_TARGETED_PROMPT.format(q=question))
        text, conf = parse_text_response(resp)
        if text is None:
            return None, None, 0.3
        visible, vconf = self.yes_no(img, f"Is this the requested information clearly readable in the image: '{text}'?")
        if visible == "yes":
            return True, text, max(conf, 0.7) * 0.9 + vconf * 0.1
        return None, None, 0.4

    def locate(self, img, phrase: str):
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
            scale = 1.0 if max(abs(v) for v in box) <= 1.0 else 1000.0
            ymin, xmin, ymax, xmax = [v / scale for v in (ymin, xmin, ymax, xmax)]
            x1, y1 = max(0.0, min(xmin, 1.0)), max(0.0, min(ymin, 1.0))
            x2, y2 = max(0.0, min(xmax, 1.0)), max(0.0, min(ymax, 1.0))
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)
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


def parse_text_response(resp):
    if not resp:
        return None, 0.0
    if "NOT_READABLE" in resp.upper() or len(resp.strip()) < 2:
        return None, 0.3
    clean = resp.strip().strip('"')
    return clean, 0.75


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
