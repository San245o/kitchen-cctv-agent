"""
Real-time token and budget ledger enforcing the $0.30 / 60min hard cap.
Accurately accounts for Gemini 3.8 Flash multimodal token pricing,
including Context Caching ($0.075 / 1M input discount + storage).
"""

import time
from typing import Dict, Any


class CostGovernor:
    # Gemini 3.8 Flash official rates
    RATE_INPUT_PER_1M = 0.75             # $0.75 per 1,000,000 standard input tokens
    RATE_CACHED_INPUT_PER_1M = 0.075     # $0.075 per 1,000,000 cached input tokens (90% off)
    RATE_CACHE_STORAGE_PER_1M_HOUR = 0.50 # $0.50 per 1,000,000 cached tokens per hour
    RATE_OUTPUT_PER_1M = 3.75            # $3.75 per 1,000,000 output tokens
    TOKENS_PER_IMAGE = 259               # Standard Google GenAI image tokenization weight

    # Competition hard caps
    MAX_COST_PER_60MIN = 0.30            # Hard budget ceiling ($)
    MAX_RUNTIME_SEC = 1500.0             # 25 minutes wall clock

    def __init__(self):
        self.start_time = time.time()
        self.frames_processed = 0
        self.model_calls = 0
        self.total_input_tokens = 0
        self.total_cached_input_tokens = 0
        self.total_output_tokens = 0
        self.max_cached_tokens = 0
        self.total_cache_storage_sec = 0.0
        self.total_source_seconds = 0.0

    def record_frames(self, count: int):
        self.frames_processed += count

    def record_source_duration(self, duration_sec: float):
        self.total_source_seconds += duration_sec

    def record_call(self, num_images: int, prompt_text: str = "", response_text: str = ""):
        """
        Record a standard (uncached) VLM call.
        """
        self.model_calls += 1
        self.record_frames(num_images)

        image_tokens = num_images * self.TOKENS_PER_IMAGE
        text_input_tokens = max(10, len(prompt_text) // 4)
        output_tokens = max(20, len(response_text) // 4)

        self.total_input_tokens += (image_tokens + text_input_tokens)
        self.total_output_tokens += output_tokens

    def record_cached_call(
        self,
        cached_tokens: int,
        prompt_text: str = "",
        response_text: str = "",
        cache_lifetime_sec: float = 300.0,
    ):
        """
        Record a VLM call hitting a Google Context Cache.
        Billed at the 90% discounted $0.075 / 1M rate.
        """
        self.model_calls += 1

        text_input_tokens = max(10, len(prompt_text) // 4)
        output_tokens = max(20, len(response_text) // 4)

        self.total_cached_input_tokens += cached_tokens
        self.total_input_tokens += text_input_tokens
        self.total_output_tokens += output_tokens

        if cached_tokens > self.max_cached_tokens:
            self.max_cached_tokens = cached_tokens
        self.total_cache_storage_sec = max(self.total_cache_storage_sec, cache_lifetime_sec)

    def get_estimated_cost_usd(self) -> float:
        # 1. Standard input cost
        standard_input_cost = (self.total_input_tokens / 1_000_000.0) * self.RATE_INPUT_PER_1M
        # 2. Cached input cost ($0.075 / 1M)
        cached_input_cost = (self.total_cached_input_tokens / 1_000_000.0) * self.RATE_CACHED_INPUT_PER_1M
        # 3. Cache storage cost ($0.50 / 1M / hour)
        storage_hours = self.total_cache_storage_sec / 3600.0
        cache_storage_cost = (self.max_cached_tokens / 1_000_000.0) * self.RATE_CACHE_STORAGE_PER_1M_HOUR * storage_hours
        # 4. Output cost ($3.75 / 1M)
        output_cost = (self.total_output_tokens / 1_000_000.0) * self.RATE_OUTPUT_PER_1M

        total = standard_input_cost + cached_input_cost + cache_storage_cost + output_cost
        return round(total, 4)

    def get_normalized_cost_per_60min(self) -> float:
        cost = self.get_estimated_cost_usd()
        source_minutes = self.total_source_seconds / 60.0
        if source_minutes >= 1.0:
            return round((cost / source_minutes) * 60.0, 4)
        return round(cost, 4)

    def get_run_log(self) -> Dict[str, Any]:
        """
        Returns official run_log dictionary matching challenge specification.
        """
        elapsed = round(time.time() - self.start_time, 2)
        est_cost = self.get_estimated_cost_usd()
        norm_cost = self.get_normalized_cost_per_60min()

        is_budget_pass = (norm_cost <= self.MAX_COST_PER_60MIN) and (elapsed <= self.MAX_RUNTIME_SEC)

        return {
            "runtime_seconds": elapsed,
            "frames_processed": self.frames_processed,
            "model_calls": self.model_calls,
            "estimated_model_api_cost_usd": est_cost,
            "normalized_model_api_cost_per_60min_usd": norm_cost,
            "source_video_minutes": round(self.total_source_seconds / 60.0, 2),
            "budget_status": "PASS" if is_budget_pass else "FAIL",
            "model_primary": "gemini-3.8-flash",
            "context_caching_used": self.total_cached_input_tokens > 0,
        }
