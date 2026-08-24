import json
import math
import re


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


TIME_PATTERNS = [
    (re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d+))?$"), "hms"),
    (re.compile(r"^(\d{1,2}):(\d{2})$"), "ms"),
    (re.compile(r"^(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?$"), "sec"),
]


def parse_time_to_seconds(text: str):
    if text is None:
        return None
    t = str(text).strip()
    m = re.search(r"\bat\s+(\d{1,2}:[\d:.]+)\b", t)
    if not m:
        m = re.search(r"\b(\d{1,2}:[\d:.]+)\b", t)
    target = m.group(1) if m else t
    for pat, kind in TIME_PATTERNS:
        mm = pat.match(target)
        if not mm:
            continue
        g = mm.groups()
        if kind == "hms":
            h = int(g[0] or 0)
            mi = int(g[1])
            s = int(g[2])
            frac = float("0." + g[3]) if g[3] else 0.0
            return h * 3600 + mi * 60 + s + frac
        if kind == "ms":
            mi = int(g[0]) if len(g) == 2 else int(g[1] or 0)
            s = int(g[-1])
            if mi >= 60 and s == 0 and len(g) == 2:
                return mi * 60.0
            return mi * 60 + s
        if kind == "sec":
            return float(g[0])
    num = re.fullmatch(r"(\d+(?:\.\d+)?)(?!\d)", target)
    if num:
        v = float(num.group(1))
        return v * 60.0 if v <= 59 else v
    return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def round_t(v, nd=1):
    return round(float(v), nd)


def span(t0, t1, duration=None):
    if duration is not None:
        t0 = clamp(t0, 0.0, duration)
        t1 = clamp(t1, 0.0, duration)
    if t1 < t0:
        t0, t1 = t1, t0
    return {"timestamp_start": round_t(t0), "timestamp_end": round_t(max(t1, t0 + 0.5))}
