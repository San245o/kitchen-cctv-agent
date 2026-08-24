import numpy as np

from .detector import boxes_in_zone, head_box
from .io_utils import clamp, span
from .vlm import consensus, crop_upscale

ZONE_WORDS = {
    "prep counter": "prep counter",
    "counter": "prep counter",
    "stove": "stove",
    "handoff": "handoff shelf",
    "shelf": "shelf",
    "sink": "sink sink area",
}


def not_visible(qid, reason="insufficient visual evidence"):
    return {
        "answer": "not_visible",
        "confidence": 0.3,
        "evidence": [],
        "reason": reason,
        "_qid": qid,
    }


class Ctx:
    def __init__(self, video_id, store, detector, vlm, budget, cfg):
        self.video_id = video_id
        self.store = store
        self.detector = detector
        self.vlm = vlm
        self.budget = budget
        self.cfg = cfg
        self.zone_cache: dict[str, tuple | None] = {}
        self.duration = store.duration

    def zone_for(self, text: str):
        low = (text or "").lower()
        phrase = None
        for k, v in ZONE_WORDS.items():
            if k in low:
                phrase = v
                break
        if phrase is None:
            return None
        if phrase in self.zone_cache:
            return self.zone_cache[phrase]
        frame_t = self.store.coarse_times[len(self.store.coarse_times) // 2]
        img = self.store.get_coarse(frame_t)
        if img is None:
            img = self.store.get_fine(frame_t)
        box = self.vlm.locate(img, phrase) if img is not None else None
        m = float(self.cfg["zones"]["margin"])
        if box:
            x1, y1, x2, y2 = box
            box = (
                clamp(x1 - m, 0, 1),
                clamp(y1 - m * 1.5, 0, 1),
                clamp(x2 + m, 0, 1),
                clamp(y2 + m, 0, 1),
            )
        self.zone_cache[phrase] = box
        return box

    def person_boxes(self, t):
        if not self.detector.available:
            return []
        times = self.neighbor_times(t, 3, float(self.cfg["sampling"]["coarse_stride_seconds"]))
        h, w = None, None
        per_frame = []
        for tt in times:
            img = self.store.get_fine(tt)
            if img is None:
                continue
            h, w = img.shape[:2]
            per_frame.append((tt, boxes_in_zone(self.detector.detect_persons(img), None, w, h)))
        return per_frame

    def neighbor_times(self, t, count, step):
        base = self.store.coarse_times
        if not base:
            return [t]
        nearest_idx = int(np.argmin(np.abs(np.array(base) - t)))
        lo = max(0, nearest_idx - count // 2)
        hi = min(len(base), lo + count)
        window = base[lo:hi]
        if t not in window:
            window = sorted(set(window + [self.store.coarse_times[nearest_idx]]))
        return window

    def neighbor_frames(self, t, count):
        frames = []
        step = float(self.cfg["sampling"]["coarse_stride_seconds"]) / 2
        for tt in (t - step, t, t + step):
            img = self.store.get_fine(tt) if abs(tt - (self.store.nearest_coarse_time(t) or t)) > 0.01 else None
            if img is None:
                img = self.store.get_coarse(self.store.nearest_coarse_time(t)) if tt == t else self.store.get_fine(tt)
            if img is not None:
                frames.append(img)
            if len(frames) >= count:
                break
        return frames


def handle_count(r, ctx: Ctx):
    t = r.target_time if r.target_time is not None else ctx.duration / 2
    zone = ctx.zone_for(r.raw_question)
    per_frame = ctx.person_boxes(t)
    usable = []
    for tt, boxes in per_frame:
        if zone:
            img = ctx.store.get_fine(tt)
            if img is not None:
                h, w = img.shape[:2]
                boxes = boxes_in_zone(boxes, zone, w, h)
        usable.append((tt, len(boxes)))
    if not usable or not ctx.detector.available:
        return not_visible("count", "person detector unavailable or no frames")
    counts = [c for _, c in usable]
    med = int(np.median(counts))
    stable = max(counts) - min(counts) <= 1
    conf = 0.85 if stable else 0.6
    ev = span(min(tt for tt, _ in usable), max(tt for tt, _ in usable) + 1.0, ctx.duration)
    return {
        "answer": str(med),
        "confidence": conf,
        "evidence": [{"video_id": ctx.video_id, **ev}],
        "method": "detector_median",
        "per_frame_counts": counts,
    }


def handle_state(r, ctx: Ctx):
    t = r.target_time if r.target_time is not None else ctx.duration / 2
    low = r.raw_question.lower()
    frames = ctx.neighbor_frames(t, int(ctx.cfg["thresholds"]["consensus_of"]))
    votes = []

    object_words = ["container", "lid", "bag", "box", "sealed", "closed"]
    uses_object = any(w in low for w in object_words) and "wearing" not in low and "cap" not in low and "hairnet" not in low

    if uses_object:
        obj_phrase = "container with a lid"
        zone = ctx.zone_for(r.raw_question)
        for img in frames:
            z = zone or ctx.vlm.locate(img, obj_phrase)
            if z is None:
                continue
            crop = crop_upscale(img, z, scale=2)
            v, c = ctx.vlm.yes_no(crop, "Is this container closed with its lid sealed on top?")
            votes.append((v, c))
    else:
        for img in frames:
            h, w = img.shape[:2]
            persons = ctx.detector.detect_persons(img)
            if not persons:
                votes.append((None, 0.3))
                continue
            best = max(persons, key=lambda b: b[4])
            hb = head_box(best, w, h)
            crop = crop_upscale(img, hb, scale=3)
            v, c = ctx.vlm.yes_no(crop, "Is the person in this image wearing a cap or hairnet on their head?")
            votes.append((v, c))

    ans, conf = consensus(votes, int(ctx.cfg["thresholds"]["consensus_min_agree"]))
    if ans is None or conf < float(ctx.cfg["thresholds"]["min_answer_confidence"]):
        return not_visible("state", f"weak evidence votes={votes}")
    ev = span(t - 1.0, t + 1.0, ctx.duration)
    return {
        "answer": ans,
        "confidence": conf,
        "evidence": [{"video_id": ctx.video_id, **ev}],
        "votes": [[v, round(c, 2)] for v, c in votes],
    }


def handle_timestamp(r, ctx: Ctx):
    phrase = r.phrase or "the described event"
    want_last = bool(r.first_last == "last") or "last" in r.phrase.lower()
    zone = ctx.zone_for(phrase)
    candidates = rank_candidates(ctx, zone)
    verified_onset = None
    used_span = None
    order = candidates[::-1] if want_last else candidates
    micro = float(ctx.cfg["sampling"]["micro_stride_seconds"])
    refine_max = int(ctx.cfg["thresholds"]["refine_max_steps"])

    for t_cand in order:
        if ctx.budget.out_of_time or ctx.budget.frames_remaining < 4:
            break
        frames = []
        for dt in (-micro, 0.0, micro):
            img = ctx.store.get_fine(t_cand + dt)
            if img is not None:
                frames.append(img)
        if not frames:
            continue
        v, c = ctx.vlm.verify_window(
            frames, f"is {phrase} happening or newly visible here?"
        )
        if v != "yes":
            continue
        onset = t_cand - micro
        steps = 0
        while steps < refine_max and ctx.budget.frames_remaining >= 1:
            img = ctx.store.get_fine(onset - micro)
            if img is None:
                break
            vv, _cc = ctx.vlm.verify_window([img], f"is {phrase} already happening or visible here?")
            if vv == "yes":
                onset -= micro
                steps += 1
            else:
                break
        verified_onset = max(0.0, onset)
        used_span = span(max(0, verified_onset - 0.5), verified_onset + 2.0, ctx.duration)
        break

    if verified_onset is None:
        return not_visible("timestamp", "no candidate moment verified")
    return {
        "answer": round(verified_onset, 1),
        "confidence": 0.8,
        "evidence": [{"video_id": ctx.video_id, **used_span}],
        "candidates_checked": len(order),
    }


def rank_candidates(ctx: Ctx, zone, top_k=None):
    top_k = top_k or int(ctx.cfg["thresholds"]["candidate_top_k"])
    scored = []
    for t, mot in ctx.store.motion.items():
        s = mot
        scored.append((s, t))
    scored.sort(reverse=True)
    return [t for _s, t in scored[:top_k]]


def handle_duration(r, ctx: Ctx):
    phrase = r.phrase or "the item"
    zone = ctx.zone_for(phrase)
    candidates = rank_candidates(ctx, zone, top_k=8)
    start = None
    for t_cand in candidates:
        if ctx.budget.frames_remaining < 4:
            break
        frames = [ctx.store.get_fine(t_cand + d) for d in (-0.5, 0.0, 0.5)]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        v, _c = ctx.vlm.verify_window(frames, f"has {phrase} just been placed or left here?")
        if v == "yes":
            start = t_cand
            break
    if start is None:
        return not_visible("duration", "start of interval not found")
    end = None
    coarse_after = [t for t in ctx.store.coarse_times if t > start + float(ctx.cfg["sampling"]["coarse_stride_seconds"])]
    for tt in coarse_after:
        img = ctx.store.get_fine(tt)
        if img is None:
            continue
        persons = ctx.detector.detect_persons(img)
        if zone:
            h, w = img.shape[:2]
            persons = boxes_in_zone(persons, zone, w, h)
        if persons and ctx.store.motion.get(tt, 0) > np.median(list(ctx.store.motion.values()) or [0]):
            end = tt
            break
    if end is None:
        end = min(start + 60.0, ctx.duration)
        conf = 0.5
    else:
        conf = 0.75
    dur = round(end - start, 1)
    return {
        "answer": dur,
        "confidence": conf,
        "evidence": [{"video_id": ctx.video_id, **span(start, end, ctx.duration)}],
        "unit": "seconds",
    }


def handle_order(r, ctx: Ctx):
    results = []
    for opt in r.options:
        sub = type("R", (), {})()
        sub.phrase = opt
        sub.first_last = "first"
        sub.target_time = None
        saved_cfg = ctx.cfg["thresholds"]
        ctx.cfg["thresholds"] = dict(saved_cfg, candidate_top_k=min(6, int(saved_cfg["candidate_top_k"])))
        res = handle_timestamp(sub, ctx)
        ctx.cfg["thresholds"] = saved_cfg
        t = res.get("answer")
        results.append((opt, t if isinstance(t, (int, float)) else None, res))
    found = [(o, t) for o, t, _res in results if isinstance(t, (int, float))]
    if len(found) < 2:
        return not_visible("order", f"only {len(found)} events located")
    found_sorted = sorted(found, key=lambda kv: kv[1])
    pick = found_sorted[-1][0] if (r.first_last or "last") == "last" else found_sorted[0][0]
    evidence = [
        {"video_id": ctx.video_id, **span(t, t + 1.5, ctx.duration)} for o, t in found_sorted
    ]
    return {
        "answer": pick,
        "confidence": 0.7,
        "evidence": evidence,
        "timeline": [[o, round(t, 1)] for o, t in found_sorted],
    }


def handle_ocr(r, ctx: Ctx):
    t = r.target_time
    if t is not None:
        base_t = ctx.store.nearest_coarse_time(t)
        tries = [base_t]
    else:
        ranked = rank_candidates(ctx, None, top_k=3)
        tries = ranked or ctx.store.coarse_times[:: max(1, len(ctx.store.coarse_times) // 3)]
    best_text, best_conf = None, 0.0
    for tt in tries[:3]:
        img = ctx.store.get_coarse(tt)
        if img is None:
            img = ctx.store.get_fine(tt)
        if img is None:
            continue
        text, c = ctx.vlm.read_text(img)
        if text and c > best_conf:
            best_text, best_conf = text, c
        if text is None:
            h, w = img.shape[:2]
            quads = [(0, 0, 0.5, 0.5), (0.5, 0, 1.0, 0.5), (0, 0.5, 0.5, 1.0), (0.5, 0.5, 1.0, 1.0)]
            for qz in quads:
                crop = crop_upscale(img, qz, scale=2)
                tx, cc = ctx.vlm.read_text(crop)
                if tx and cc > best_conf:
                    best_text, best_conf = tx, cc
    if not best_text or best_conf < float(ctx.cfg["thresholds"]["min_answer_confidence"]):
        return not_visible("ocr", "no readable text found")
    return {
        "answer": best_text,
        "confidence": best_conf,
        "evidence": [
            {"video_id": ctx.video_id, **span(tries[0], tries[0] + 1.0, ctx.duration)}
        ],
    }


def handle_general_yesno(r, ctx: Ctx):
    phrase = r.phrase or r.raw_question.rstrip("?")
    zone = ctx.zone_for(phrase)
    candidates = rank_candidates(ctx, zone, top_k=6)
    votes = []
    used = []
    for t_cand in candidates:
        if ctx.budget.frames_remaining < 4:
            break
        frames = [ctx.store.get_fine(t_cand + d) for d in (-0.5, 0.0, 0.5)]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        used.append(t_cand)
        v, c = ctx.vlm.verify_window(frames, phrase)
        votes.append((v, c))
        if v == "yes":
            break
    ans, conf = consensus(votes, 1)
    if not used:
        return not_visible("general_yesno", "no frames available to verify")
    if ans is None:
        return not_visible("general_yesno", "event never clearly observed")
    return {
        "answer": ans,
        "confidence": conf,
        "evidence": [{"video_id": ctx.video_id, **span(min(used), max(used) + 1.5, ctx.duration)}],
    }
