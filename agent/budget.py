from dataclasses import dataclass, field
import time


@dataclass
class CallRecord:
    model: str
    kind: str
    cost_usd: float
    seconds: float


class Budget:
    def __init__(self, duration_seconds: float, cfg: dict):
        hour = 3600.0
        self.duration_seconds = max(duration_seconds, 1e-6)
        video_hours = self.duration_seconds / hour
        self.frame_cap = int(round(cfg["frames_per_video_hour"] * video_hours))
        self.coarse_budget = int(self.frame_cap * cfg["coarse_fraction"])
        self.fine_budget = self.frame_cap - self.coarse_budget
        self.frames_decoded = 0
        self.coarse_used = 0
        self.fine_used = 0
        self.calls: list[CallRecord] = []
        self.t_start = time.perf_counter()
        self.wall_clock_limit_s = cfg.get("wall_clock_minutes", 24) * 60

    @property
    def frames_processed(self) -> int:
        return self.frames_decoded

    @property
    def frames_remaining(self) -> int:
        return max(0, self.frame_cap - self.frames_decoded)

    def spend_coarse(self, n: int = 1) -> bool:
        if self.coarse_used + n > self.coarse_budget or self.frames_remaining < n:
            return False
        self.coarse_used += n
        self.frames_decoded += n
        return True

    def spend_fine(self, n: int = 1) -> bool:
        if self.fine_used + n > self.fine_budget or self.frames_remaining < n:
            return False
        self.fine_used += n
        self.frames_decoded += n
        return True

    def log_call(self, model: str, kind: str, cost_usd: float = 0.0) -> None:
        self.calls.append(CallRecord(model, kind, float(cost_usd), time.perf_counter()))

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.t_start

    @property
    def out_of_time(self) -> bool:
        return self.elapsed_seconds >= self.wall_clock_limit_s

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def normalized_cost_per_60min(self) -> float:
        return self.total_cost_usd * (hour / self.duration_seconds)
