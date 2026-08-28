import gc
import time

import numpy as np


class SemanticIndex:
    """Small in-process frame/text index used for question-conditioned retrieval."""

    def __init__(self, cfg: dict, budget):
        self.cfg = cfg["models"].get("retriever", {})
        self.budget = budget
        self.model_id = self.cfg.get("model", "google/siglip2-base-patch16-224")
        self.model = None
        self.processor = None
        self.device = None
        self.available = False
        self.load_error = None
        self.times = []
        self.image_features = None
        self.text_features = {}

    def ensure_loaded(self):
        if self.available:
            return True
        if not self.cfg.get("enabled", True):
            return False
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id, dtype=dtype).to(self.device).eval()
            self.available = True
        except Exception as exc:
            self.load_error = str(exc)
        return self.available

    def build(self, store, query_texts):
        if not self.ensure_loaded():
            return False
        import torch
        from PIL import Image

        batch_size = int(self.cfg.get("batch_size", 24))
        times = []
        features = []
        batch_times = []
        batch_images = []

        def flush_images():
            if not batch_images or self.budget.exhausted:
                return
            started = time.perf_counter()
            success = False
            error = None
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    emb = self.model.get_image_features(**inputs)
                    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                features.append(emb.float().cpu().numpy())
                times.extend(batch_times)
                success = True
            except Exception as exc:
                error = exc
            finally:
                self.budget.log_call(
                    self.model_id,
                    "retriever_image_batch",
                    started_at=started,
                    duration_seconds=time.perf_counter() - started,
                    success=success,
                    error=error,
                )
                batch_images.clear()
                batch_times.clear()

        for t in store.coarse_times:
            if self.budget.exhausted:
                break
            img = store.get_coarse(t)
            if img is None:
                continue
            batch_images.append(Image.fromarray(img[:, :, ::-1]))
            batch_times.append(round(float(t), 2))
            if len(batch_images) >= batch_size:
                flush_images()
        flush_images()

        if not features:
            return False
        self.times = times
        self.image_features = np.concatenate(features, axis=0)

        texts = sorted({str(t).strip() for t in query_texts if str(t).strip()})
        for start in range(0, len(texts), 32):
            if self.budget.exhausted:
                break
            chunk = texts[start : start + 32]
            called = time.perf_counter()
            success = False
            error = None
            try:
                inputs = self.processor(text=chunk, padding=True, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    emb = self.model.get_text_features(**inputs)
                    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                for text, vector in zip(chunk, emb.float().cpu().numpy()):
                    self.text_features[text] = vector
                success = True
            except Exception as exc:
                error = exc
            finally:
                self.budget.log_call(
                    self.model_id,
                    "retriever_text_batch",
                    started_at=called,
                    duration_seconds=time.perf_counter() - called,
                    success=success,
                    error=error,
                )
        return bool(self.text_features)

    def scores(self, text):
        vector = self.text_features.get((text or "").strip())
        if vector is None or self.image_features is None:
            return None
        values = self.image_features @ vector
        return {t: float(v) for t, v in zip(self.times, values)}

    def release_model(self):
        self.model = None
        self.processor = None
        self.available = bool(self.image_features is not None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
