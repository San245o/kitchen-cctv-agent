from dataclasses import asdict, dataclass
import time
from typing import Optional


@dataclass
class CallRecord:
    model: str
    kind: str
    cost_usd: float
    started_at_seconds: float
    duration_seconds: float
    success: bool
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class FrameRecord:
    timestamp: float
    sample_kind: str

    def to_dict(self):
        return asdict(self)


HOUR = 3600.0


class Budget:
    def __init__(self, duration_seconds: float, cfg: dict, clock0: float = None):
        self.duration_seconds = max(duration_seconds, 1e-6)
        video_hours = self.duration_seconds / HOUR
        self.frame_cap = int(round(cfg["frames_per_video_hour"] * video_hours))
        self.coarse_budget = int(self.frame_cap * cfg["coarse_fraction"])
        self.fine_budget = self.frame_cap - self.coarse_budget
        self.frames_decoded = 0
        self.coarse_used = 0
        self.fine_used = 0
        self.calls: list[CallRecord] = []
        self.frame_records: list[FrameRecord] = []
        self.video_id = None
        self.clock0 = clock0 if clock0 is not None else time.perf_counter()
        self.wall_clock_limit_s = cfg.get("wall_clock_minutes", 24) * 60
        self.calls_cap = int(cfg.get("max_model_calls", 400))
        self.vlm_calls = 0

    @property
    def frames_processed(self) -> int:
        return self.frames_decoded

    @property
    def frames_remaining(self) -> int:
        return max(0, self.frame_cap - self.frames_decoded)

    def spend_coarse(self, n: int = 1, timestamp: float = None) -> bool:
        if self.coarse_used + n > self.coarse_budget or self.frames_remaining < n:
            return False
        self.coarse_used += n
        self.frames_decoded += n
        if timestamp is not None:
            self.frame_records.append(FrameRecord(round(float(timestamp), 3), "coarse"))
        return True

    def spend_fine(self, n: int = 1, timestamp: float = None) -> bool:
        if self.fine_used + n > self.fine_budget or self.frames_remaining < n:
            return False
        self.fine_used += n
        self.frames_decoded += n
        if timestamp is not None:
            self.frame_records.append(FrameRecord(round(float(timestamp), 3), "fine"))
        return True

    def log_call(
        self,
        model: str,
        kind: str,
        cost_usd: float = 0.0,
        started_at: float = None,
        duration_seconds: float = 0.0,
        success: bool = True,
        error: str = None,
    ) -> None:
        if kind.startswith("vlm"):
            self.vlm_calls += 1
        started_at = time.perf_counter() if started_at is None else started_at
        self.calls.append(
            CallRecord(
                model=model,
                kind=kind,
                cost_usd=float(cost_usd),
                started_at_seconds=round(max(0.0, started_at - self.clock0), 4),
                duration_seconds=round(float(duration_seconds), 4),
                success=bool(success),
                error=(str(error)[:300] if error else None),
            )
        )

    @property
    def out_of_calls(self) -> bool:
        return len(self.calls) >= self.calls_cap

    @property
    def exhausted(self) -> bool:
        return self.out_of_time or self.out_of_calls

    @property
    def elapsed_since_global_start(self) -> float:
        return time.perf_counter() - self.clock0

    @property
    def out_of_time(self) -> bool:
        return self.elapsed_since_global_start >= self.wall_clock_limit_s

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def normalized_cost_per_60min(self) -> float:
        return self.total_cost_usd * (HOUR / self.duration_seconds)
