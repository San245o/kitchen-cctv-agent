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


def interpret_times(t, duration):
    return [float(t)]


class Ctx:
    def __init__(self, video_id, store, detector, vlm, budget, cfg, semantic=None):
        self.video_id = video_id
        self.store = store
        self.detector = detector
        self.vlm = vlm
        self.budget = budget
        self.cfg = cfg
        self.semantic = semantic
        self.zone_cache: dict[str, tuple | None] = {}
        self.duration = store.duration

    def rank_candidates(self, text, zone=None, top_k=None):
        semantic_scores = self.semantic.scores(text) if self.semantic is not None else None
        return self.store.rank_candidates(
            zone,
            top_k=top_k or int(self.cfg["thresholds"]["candidate_top_k"]),
            semantic_scores=semantic_scores,
            semantic_weight=float(self.cfg["thresholds"].get("semantic_weight", 0.75)),
        )

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
        if not self.store.coarse_times:
            return None
        frame_t = self.store.coarse_times[len(self.store.coarse_times) // 2]
        img = self.store.get_coarse(frame_t)
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
        per_frame = []
        for tt in times:
            if self.budget.exhausted:
                break
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
            if self.budget.exhausted or self.budget.frames_remaining < 1:
                break
            img = self.store.get_fine(tt) if abs(tt - (self.store.nearest_coarse_time(t) or t)) > 0.01 else None
            if img is None:
                img = self.store.get_coarse(self.store.nearest_coarse_time(t)) if tt == t else self.store.get_fine(tt)
            if img is not None:
                frames.append(img)
            if len(frames) >= count:
                break
        return frames

    def verify_window(self, t_cand, question):
        micro = float(self.cfg["sampling"]["micro_stride_seconds"])
        frames = []
        for dt in (-micro, 0.0, micro):
            if self.budget.exhausted:
                break
            img = self.store.get_fine(t_cand + dt)
            if img is not None:
                frames.append(img)
        if not frames:
            return None, 0.0, []
        v, c = self.vlm.verify_window(frames, question)
        return v, c, frames


def handle_count(r, ctx: Ctx):
    base_t = r.target_time if r.target_time is not None else ctx.duration / 2
    zone = ctx.zone_for(r.raw_question)
    best = None
    for t in interpret_times(base_t, ctx.duration):
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
            continue
        counts = [c for _, c in usable]
        med = int(np.median(counts))
        stable = max(counts) - min(counts) <= 1
        conf = 0.85 if stable else 0.6
        if best is None or med > best["answer"] or (med == best["answer"] and conf > best["confidence"]):
            ev = span(min(tt for tt, _ in usable), max(tt for tt, _ in usable) + 1.0, ctx.duration)
            best = {
                "answer": med,
                "confidence": conf,
                "evidence": [{"video_id": ctx.video_id, **ev}],
                "method": "detector_median",
                "per_frame_counts": counts,
                "_t_used": t,
            }
    if best is None:
        return not_visible("count", "person detector unavailable or no frames")
    return best


def handle_state(r, ctx: Ctx):
    base_t = r.target_time if r.target_time is not None else ctx.duration / 2
    low = r.raw_question.lower()
    is_ppe = any(w in low for w in ["wearing", "cap", "hairnet", "head cover"])
    object_words = ["container", "lid", "bag", "box", "sealed", "closed"]
    uses_object = (
        any(w in low for w in object_words)
        and not is_ppe
    )
    zone = ctx.zone_for(r.raw_question)

    if not uses_object and not is_ppe:
        best_timed = None
        for t in interpret_times(base_t, ctx.duration):
            frames = ctx.neighbor_frames(t, int(ctx.cfg["thresholds"]["consensus_of"]))
            if not frames:
                continue
            value, conf = ctx.vlm.verify_window(frames, r.raw_question)
            if value is not None and (best_timed is None or conf > best_timed[1]):
                best_timed = (value, conf, t)
        if best_timed is None or best_timed[1] < float(
            ctx.cfg["thresholds"]["min_answer_confidence"]
        ):
            return not_visible("state", f"timed action not clear at t={base_t}")
        value, conf, used_t = best_timed
        return {
            "answer": value,
            "confidence": conf,
            "evidence": [
                {"video_id": ctx.video_id, **span(used_t - 1.5, used_t + 1.5, ctx.duration)}
            ],
        }

    answer_best = None
    used_t = base_t
    for t in interpret_times(base_t, ctx.duration):
        frames = ctx.neighbor_frames(t, int(ctx.cfg["thresholds"]["consensus_of"]))
        round_votes = []
        for img in frames:
            if ctx.budget.exhausted:
                break
            if uses_object:
                z = zone or ctx.vlm.locate(img, "container with a lid")
                if z is None:
                    round_votes.append((None, 0.3))
                    continue
                crop = crop_upscale(img, z, scale=2)
                v, c = ctx.vlm.yes_no(crop, "Is this container closed with its lid sealed on top?")
            else:
                h, w = img.shape[:2]
                persons = ctx.detector.detect_persons(img)
                if zone:
                    persons = boxes_in_zone(persons, zone, w, h)
                if not persons:
                    round_votes.append((None, 0.3))
                    continue
                person = max(persons, key=lambda b: b[4])
                hb = head_box(person, w, h)
                crop = crop_upscale(img, hb, scale=3)
                v, c = ctx.vlm.yes_no(crop, "Is the person in this image wearing a cap or hairnet on their head?")
            round_votes.append((v, c))
        ans, conf = consensus(round_votes, int(ctx.cfg["thresholds"]["consensus_min_agree"]))
        if ans is not None and (answer_best is None or conf > answer_best[1]):
            answer_best = (ans, conf, list(round_votes))
            used_t = t
        if ans is not None and conf >= float(ctx.cfg["thresholds"]["min_answer_confidence"]):
            break

    if answer_best is None:
        return not_visible("state", f"weak evidence at t={base_t}")
    ans, conf, votes = answer_best
    if conf < float(ctx.cfg["thresholds"]["min_answer_confidence"]):
        return not_visible("state", f"weak evidence votes={votes}")
    ev = span(used_t - 1.0, used_t + 1.0, ctx.duration)
    return {
        "answer": ans,
        "confidence": conf,
        "evidence": [{"video_id": ctx.video_id, **ev}],
        "votes": [[v, round(c, 2)] for v, c in votes],
    }


def handle_timestamp(r, ctx: Ctx):
    phrase = r.phrase or "the described event"
    want_last = (r.first_last == "last") or ("last" in phrase.lower() and "how long" not in phrase.lower())
    zone = ctx.zone_for(phrase)
    top_k = int(ctx.cfg["thresholds"]["candidate_top_k"])
    candidates = sorted(ctx.rank_candidates(phrase, zone, top_k=top_k))
    refine_max = int(ctx.cfg["thresholds"]["refine_max_steps"])
    micro = float(ctx.cfg["sampling"]["micro_stride_seconds"])

    scan_order = candidates[::-1] if want_last else candidates
    verified_onset = None
    checked = 0
    for t_cand in scan_order:
        if ctx.budget.exhausted or ctx.budget.frames_remaining < 5:
            break
        checked += 1
        v, _c, _frames = ctx.verify_window(
            t_cand, f"is {phrase} happening or newly visible here?"
        )
        if v != "yes":
            continue
        onset = max(0.0, t_cand - micro)
        steps = 0
        while steps < refine_max and ctx.budget.frames_remaining >= 1 and not ctx.budget.exhausted:
            img = ctx.store.get_fine(onset - micro)
            if img is None:
                break
            vv, _cc = ctx.vlm.verify_window(
                [img], f"is {phrase} already happening or visible here?"
            )
            if vv == "yes":
                onset -= micro
                steps += 1
            else:
                break
        verified_onset = onset
        break

    if verified_onset is None:
        return not_visible("timestamp", "no candidate moment verified")
    return {
        "answer": round(float(verified_onset), 1),
        "confidence": 0.8,
        "evidence": [
            {
                "video_id": ctx.video_id,
                **span(max(0, verified_onset - 0.5), verified_onset + 2.0, ctx.duration),
            }
        ],
        "candidates_checked": checked,
        "zone_used": zone is not None,
    }


def locate_event(r, ctx: Ctx, top_k=None):
    saved = ctx.cfg["thresholds"]
    if top_k:
        ctx.cfg["thresholds"] = dict(saved, candidate_top_k=top_k)
    res = handle_timestamp(r, ctx)
    ctx.cfg["thresholds"] = saved
    t = res.get("answer")
    return t if isinstance(t, (int, float)) else None


def handle_duration(r, ctx: Ctx):
    sub = RoutedLike(r.phrase or "the item", "first")
    start = locate_event(sub, ctx, top_k=8)
    if start is None:
        return not_visible("duration", "start of interval not found")
    zone = ctx.zone_for(r.raw_question)
    motion_vals = list(ctx.store.motion.values())
    med_motion = float(np.median(motion_vals)) if motion_vals else 0.0
    end = None
    for tt in [t for t in ctx.store.coarse_times if t > start + float(ctx.cfg["sampling"]["coarse_stride_seconds"])]:
        if ctx.budget.exhausted:
            break
        img = ctx.store.get_fine(tt)
        if img is None:
            continue
        persons = ctx.detector.detect_persons(img)
        if zone:
            h, w = img.shape[:2]
            persons = boxes_in_zone(persons, zone, w, h)
        if persons and ctx.store.motion.get(tt, 0.0) > med_motion:
            end = tt
            break
    if end is None:
        return not_visible("duration", "end of interval never clearly observed")
    dur = round(end - start, 1)
    return {
        "answer": dur,
        "confidence": 0.75,
        "evidence": [{"video_id": ctx.video_id, **span(start, end, ctx.duration)}],
        "unit": "seconds",
    }


class RoutedLike:
    def __init__(self, phrase, first_last="first", target_time=None, raw_question=""):
        self.phrase = phrase
        self.first_last = first_last
        self.target_time = target_time
        self.raw_question = raw_question


def handle_order(r, ctx: Ctx):
    results = []
    for opt in r.options:
        sub = RoutedLike(opt, "first")
        t = locate_event(sub, ctx, top_k=min(6, int(ctx.cfg["thresholds"]["candidate_top_k"])))
        results.append((opt, t))
    found = [(o, t) for o, t in results if isinstance(t, (int, float))]
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


def handle_mc(r, ctx: Ctx):
    if not r.options:
        return not_visible("mc", "no options parsed from question")
    t_ref = r.target_time if r.target_time is not None else None
    scored = []
    for opt in r.options:
        if ctx.budget.exhausted or ctx.budget.frames_remaining < 4:
            break
        if t_ref is not None:
            frames = ctx.neighbor_frames(t_ref, 3)
            if not frames:
                v, c = None, 0.0
            else:
                v, c = ctx.vlm.verify_window(frames, f"is this showing {opt}?")
        else:
            zone = ctx.zone_for(opt)
            v, c = None, 0.0
            for cand in ctx.rank_candidates(opt, zone, top_k=3):
                vv, cc, _f = ctx.verify_window(cand, f"is this showing {opt}?")
                if vv == "yes":
                    v, c, hit_t = vv, cc, cand
                    break
        if v == "yes":
            scored.append((opt, c, t_ref if t_ref is not None else hit_t))
    if not scored:
        return not_visible("mc", "no option verifiably shown")
    scored.sort(key=lambda kv: kv[1], reverse=True)
    opt, conf, ev_t = scored[0]
    return {
        "answer": opt,
        "confidence": round(min(conf, 0.9), 2),
        "evidence": [{"video_id": ctx.video_id, **span(ev_t - 1.0, ev_t + 1.0, ctx.duration)}],
        "alternatives_rejected": [o for o, _c, _t in scored[1:]],
    }


def handle_ocr(r, ctx: Ctx):
    low = r.raw_question.lower()
    targeted = any(w in low for w in ["visible", "readable", "number", "written", "displayed"])
    question_text = r.raw_question.rstrip("?") + "?"
    if r.target_time is not None:
        tries = [ctx.store.nearest_coarse_time(r.target_time)]
    else:
        ranked = ctx.rank_candidates(r.raw_question, None, top_k=6)
        tries = ranked if ranked else ctx.store.coarse_times[:: max(1, len(ctx.store.coarse_times) // 3)]
    result = None
    for tt in tries[:3]:
        if ctx.budget.exhausted:
            break
        img = ctx.store.get_coarse(tt)
        if img is None:
            img = ctx.store.get_fine(tt)
        if img is None:
            continue
        if targeted:
            visible, text, conf = ctx.vlm.read_targeted(img, question_text)
            if visible and text and conf >= float(ctx.cfg["thresholds"]["min_answer_confidence"]):
                result = (text, conf, tt)
                break
        else:
            text, conf = ctx.vlm.read_text(img)
            if text and conf >= float(ctx.cfg["thresholds"]["min_answer_confidence"]):
                result = (text, conf, tt)
                break
        h, w = img.shape[:2]
        quads = [(0, 0, 0.5, 0.5), (0.5, 0, 1.0, 0.5), (0, 0.5, 0.5, 1.0), (0.5, 0.5, 1.0, 1.0)]
        for qz in quads:
            if ctx.budget.exhausted:
                break
            crop = crop_upscale(img, qz, scale=2)
            if targeted:
                visible, text, conf = ctx.vlm.read_targeted(crop, question_text)
                ok = visible and text
            else:
                text, conf = ctx.vlm.read_text(crop)
                ok = bool(text)
            if ok and conf >= float(ctx.cfg["thresholds"]["min_answer_confidence"]):
                result = (text, conf, tt)
                break
        if result:
            break
    if result is None:
        return not_visible("ocr", "requested text not readable in available frames")
    text, conf, tt = result
    return {
        "answer": text,
        "confidence": round(conf, 2),
        "evidence": [{"video_id": ctx.video_id, **span(tt, tt + 1.0, ctx.duration)}],
    }


def handle_general_yesno(r, ctx: Ctx):
    phrase = r.phrase or r.raw_question.rstrip("?")
    zone = ctx.zone_for(phrase)
    candidates = ctx.rank_candidates(phrase, zone, top_k=6)
    votes = []
    used = []
    for t_cand in candidates:
        if ctx.budget.exhausted or ctx.budget.frames_remaining < 4:
            break
        v, c, _frames = ctx.verify_window(t_cand, phrase)
        used.append(t_cand)
        votes.append((v, c))
        if v == "yes" and len([x for x in votes if x[0] == "yes"]) >= int(ctx.cfg["thresholds"]["consensus_min_agree"]):
            break
    ans, conf = consensus(votes, int(ctx.cfg["thresholds"]["consensus_min_agree"]))
    if not used:
        return not_visible("general_yesno", "no frames available to verify")
    if ans is None:
        return not_visible("general_yesno", "event never clearly observed")
    evidence_times = [t for (t, (v, _c)) in zip(used, votes) if v == ans]
    if not evidence_times:
        evidence_times = used
    return {
        "answer": ans,
        "confidence": conf,
        "evidence": [
            {"video_id": ctx.video_id, **span(t - 0.5, t + 0.5, ctx.duration)}
            for t in evidence_times[:3]
        ],
    }
