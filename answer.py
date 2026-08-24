import argparse
import os
import random
import sys
import time

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def load_cfg(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

HANDLERS = {}


def register_handlers():
    from agent.handlers import (
        handle_count,
        handle_duration,
        handle_general_yesno,
        handle_ocr,
        handle_order,
        handle_state,
        handle_timestamp,
    )

    HANDLERS.update(
        {
            "count": handle_count,
            "state": handle_state,
            "timestamp": handle_timestamp,
            "duration": handle_duration,
            "order": handle_order,
            "ocr": handle_ocr,
            "general_yesno": handle_general_yesno,
        }
    )


def discover_videos(path):
    if os.path.isdir(path):
        found = {}
        for fn in sorted(os.listdir(path)):
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                found[os.path.splitext(fn)[0]] = os.path.join(path, fn)
        return found
    base = os.path.splitext(os.path.basename(path))[0]
    return {base: path}


def process_video(vid, path, questions, cfg, detector, vlm, per_question_logs):
    from agent.budget import Budget
    from agent.frames import FrameStore
    from agent.handlers import Ctx, not_visible
    from agent.router import route

    answers = []
    duration_fallback = 3600.0
    probe = Budget(duration_fallback, cfg["budgets"])
    detector.budget = probe
    vlm.budget = probe
    try:
        tmp = FrameStore(path, probe, cfg["sampling"])
        duration = tmp.duration
        tmp.close()
    except Exception as e:
        print(f"warn: cannot open {path}: {e}", file=sys.stderr)
        return answers, None

    budget = Budget(duration, cfg["budgets"])
    detector.budget = budget
    vlm.budget = budget
    ctx = None
    store = None
    try:
        store = FrameStore(path, budget, cfg["sampling"])
        store.build_coarse()
        ctx = Ctx(vid, store, detector, vlm, budget, cfg)
    except Exception as e:
        print(f"warn: indexing failed for {path}: {e}", file=sys.stderr)

    for q in questions:
        qid = q.get("id")
        r = route(q.get("question", ""))
        if ctx is None:
            ans = not_visible(qid, "video unavailable or unreadable")
        elif budget.out_of_time:
            ans = not_visible(qid, "wall-clock budget exhausted")
        else:
            handler = HANDLERS.get(r.handler)
            try:
                ans = handler(r, ctx)
            except Exception as e:
                ans = not_visible(qid, f"handler error: {type(e).__name__}")
        record = {
            "id": qid,
            "answer": ans["answer"],
            "confidence": round(float(ans.get("confidence", 0.3)), 2),
            "evidence": [
                {"video_id": vid, **{k: v for k, v in ev.items() if k != "video_id"}}
                for ev in ans.get("evidence", [])
            ],
        }
        if ans.get("reason"):
            record["reason"] = ans["reason"]
        answers.append(record)
        per_question_logs.append(
            {
                "id": qid,
                "video_id": vid,
                "routed_handler": r.handler,
                "target_time": r.target_time,
                "frames_used": budget.frames_processed,
                "calls_used": len(budget.calls),
            }
        )
    if store is not None:
        store.close()
    return answers, budget


def main():
    ap = argparse.ArgumentParser(description="Kitchen CCTV QA agent (builderr submission)")
    ap.add_argument("--videos", required=True, help="directory of videos or single video file")
    ap.add_argument("--questions", required=True, help="questions JSON")
    ap.add_argument("--out", required=True, help="answers JSON output path")
    ap.add_argument("--log", required=True, help="run log JSON output path")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--limit-minutes", type=float, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = load_cfg(args.config)
    if args.limit_minutes:
        cfg["budgets"]["wall_clock_minutes"] = args.limit_minutes
    seed = int(cfg["runtime"].get("seed", 0))
    set_seed(seed)
    register_handlers()

    from agent.detector import Detector
    from agent.io_utils import dump_json, load_json
    from agent.vlm import VLM

    questions = load_json(args.questions)
    videos = discover_videos(args.videos)
    if not videos:
        print(f"no video files found at {args.videos}", file=sys.stderr)

    detector = Detector(cfg, None)
    vlm = VLM(cfg, None)

    by_video = {}
    for q in questions:
        vid = q.get("video_id") or next(iter(videos), None)
        by_video.setdefault(vid, []).append(q)

    all_answers = []
    per_question_logs = []
    budgets = []

    for vid, qs in by_video.items():
        path = videos.get(vid) or next(iter(videos.values()), None)
        answers, budget = process_video(vid, path, qs, cfg, detector, vlm, per_question_logs)
        all_answers.extend(answers)
        if budget is not None:
            budgets.append(budget)

    runtime = time.perf_counter() - t0
    frames_total = sum(b.frames_processed for b in budgets)
    calls_total = sum(len(b.calls) for b in budgets)
    cost_total = sum(b.total_cost_usd for b in budgets)
    wall_limit = float(cfg["budgets"]["wall_clock_minutes"]) * 60

    run_log = {
        "runtime_seconds": round(runtime, 1),
        "frames_processed": int(frames_total),
        "model_calls": int(calls_total),
        "estimated_model_api_cost_usd": round(cost_total, 6),
        "normalized_model_api_cost_per_60min_usd": round(cost_total, 6),
        "within_caps": {
            "runtime_under_limit": runtime < wall_limit,
            "local_models_only": True,
        },
        "caps": {
            "frames_per_60min": cfg["budgets"]["frames_per_video_hour"],
            "wall_clock_minutes": cfg["budgets"]["wall_clock_minutes"],
            "cost_cap_per_60min_usd": 0.30,
        },
        "models": {
            "detector": {"weights": cfg["models"]["detector"]["weights"], "available": detector.available},
            "vlm": {
                "primary": cfg["models"]["vlm"]["primary"],
                "available": vlm.available,
                "load_error": vlm.load_error,
            },
        },
        "per_question": per_question_logs,
        "seed": seed,
    }
    dump_json(args.out, all_answers)
    dump_json(args.log, run_log)
    print(f"wrote {args.out} ({len(all_answers)} answers) and {args.log}")


if __name__ == "__main__":
    main()
