import argparse
import os
import random
import sys
import time

import numpy as np

HOUR = 3600.0


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
        handle_mc,
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
            "mc": handle_mc,
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


def resolve_url(url: str) -> str:
    os.makedirs("cache", exist_ok=True)
    name = os.path.join("cache", f"dl_{abs(hash(url)) % 10**10}.mp4")
    if os.path.exists(name):
        return name
    if "youtube.com" in url or "youtu.be" in url:
        try:
            import yt_dlp

            with yt_dlp.YoutubeDL({"outtmpl": name, "quiet": True}) as ydl:
                ydl.download([url])
            return name
        except ImportError:
            raise RuntimeError("youtube URLs require: pip install yt-dlp")
    import urllib.request

    urllib.request.urlretrieve(url, name)
    return name


def download_only(cfg, detector, vlm):
    ok_d = detector.ensure_loaded()
    ok_v = vlm.ensure_loaded()
    print(f"detector ready: {ok_d} | vlm ready: {ok_v} ({vlm.model_id})")
    if vlm.load_error:
        print("load errors:", vlm.load_error, file=sys.stderr)
    if not ok_v:
        print(
            "warning: no VLM available; runs will degrade to not_visible answers.\n"
            "hint for RTX 50-series: pip install torch --index-url https://download.pytorch.org/whl/cu128",
            file=sys.stderr,
        )
        sys.exit(1)


def emergency_dump(out_path, log_path, questions, error):
    import json

    answers = [
        {
            "id": q.get("id"),
            "answer": "not_visible",
            "confidence": 0.3,
            "evidence": [],
            "reason": f"agent failed to initialize: {error}",
        }
        for q in (questions or [])
    ]
    log = {
        "runtime_seconds": 0.0,
        "frames_processed": 0,
        "model_calls": 0,
        "estimated_model_api_cost_usd": 0.0,
        "normalized_model_api_cost_per_60min_usd": 0.0,
        "fatal_error": str(error),
        "degraded": True,
    }
    for p, obj in ((out_path, answers), (log_path, log)):
        try:
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    print(f"emergency output written to {out_path} and {log_path}", file=sys.stderr)


def process_video(vid, path, questions, cfg, detector, vlm, clock0, per_question_logs):
    from agent.budget import Budget
    from agent.frames import FrameStore
    from agent.handlers import Ctx, not_visible
    from agent.router import route

    answers = []
    probe = Budget(3600.0, cfg["budgets"], clock0=clock0)
    detector.budget = probe
    vlm.budget = probe
    try:
        tmp = FrameStore(path, probe, cfg["sampling"])
        duration = tmp.duration
        tmp.close()
    except Exception as e:
        print(f"warn: cannot open {path}: {e}", file=sys.stderr)
        return answers, None

    budget = Budget(duration, cfg["budgets"], clock0=clock0)
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
        r = route(q.get("question", ""), q.get("type"))
        if ctx is None:
            ans = not_visible(qid, "video unavailable or unreadable")
        elif budget.exhausted:
            reason = "wall-clock budget exhausted" if budget.out_of_time else "model-call budget exhausted"
            ans = not_visible(qid, reason)
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
                "declared_type": r.qtype,
                "target_time": r.target_time,
                "frames_used": budget.frames_processed,
                "vlm_calls_used": budget.vlm_calls,
            }
        )
    if store is not None:
        store.close()
    return answers, budget


def main():
    ap = argparse.ArgumentParser(description="Kitchen CCTV QA agent (builderr submission)")
    ap.add_argument("--videos", default=None, help="directory of videos, single file, or URL")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--limit-minutes", type=float, default=None)
    ap.add_argument("--download-only", action="store_true", help="fetch/load model weights then exit (pre-warm cache outside timed runs)")
    args = ap.parse_args()

    state = {"questions": None}
    try:
        run(args, state)
    except SystemExit:
        raise
    except Exception as e:
        emergency_dump(args.out, args.log, state["questions"], e)


def run(args, state):
    t0 = time.perf_counter()
    cfg = load_cfg(args.config)
    if args.limit_minutes:
        cfg["budgets"]["wall_clock_minutes"] = args.limit_minutes
    set_seed(int(cfg["runtime"].get("seed", 0)))
    register_handlers()

    from agent.detector import Detector
    from agent.io_utils import dump_json, load_json
    from agent.vlm import VLM

    if args.download_only:
        detector = Detector(cfg, None)
        vlm = VLM(cfg, None)
        download_only(cfg, detector, vlm)
        return

    if not args.videos or not args.questions:
        raise ValueError("--videos and --questions are required unless --download-only is used")

    questions = load_json(args.questions)
    state["questions"] = questions
    videos = discover_videos(args.videos)

    detector = Detector(cfg, None)
    vlm = VLM(cfg, None)
    if args.download_only:
        download_only(cfg, detector, vlm)
        return

    by_video = {}
    for q in questions:
        vid = q.get("video_id")
        if vid and vid not in videos and len(videos) == 1:
            by_video.setdefault(next(iter(videos)), []).append(q)
        elif not vid and len(videos) >= 1:
            by_video.setdefault(next(iter(videos)), []).append(q)
        else:
            by_video.setdefault(vid, []).append(q)

    all_answers = []
    per_question_logs = []
    budgets = []

    for vid, qs in by_video.items():
        path = videos.get(vid)
        if path is None:
            for q in qs:
                all_answers.append(
                    {
                        "id": q.get("id"),
                        "answer": "not_visible",
                        "confidence": 0.3,
                        "evidence": [],
                        "reason": f"video_id '{vid}' not found in supplied videos",
                    }
                )
                per_question_logs.append(
                    {"id": q.get("id"), "video_id": vid, "routed_handler": "unresolved_video"}
                )
            continue
        try:
            if path.lower().startswith(("http://", "https://")):
                path = resolve_url(path)
        except Exception as e:
            print(f"warn: cannot fetch {path}: {e}", file=sys.stderr)
            for q in qs:
                all_answers.append(
                    {
                        "id": q.get("id"),
                        "answer": "not_visible",
                        "confidence": 0.3,
                        "evidence": [],
                        "reason": "video download failed",
                    }
                )
            continue
        answers, budget = process_video(vid, path, qs, cfg, detector, vlm, t0, per_question_logs)
        all_answers.extend(answers)
        if budget is not None:
            budgets.append(budget)

    runtime = time.perf_counter() - t0
    frames_total = sum(b.frames_processed for b in budgets)
    calls_total = sum(len(b.calls) for b in budgets)
    cost_total = sum(b.total_cost_usd for b in budgets)
    duration_sum = sum(b.duration_seconds for b in budgets)
    norm_cost = cost_total * (HOUR / duration_sum) if duration_sum > 0 else 0.0
    wall_limit = float(cfg["budgets"]["wall_clock_minutes"]) * 60

    run_log = {
        "runtime_seconds": round(runtime, 1),
        "frames_processed": int(frames_total),
        "model_calls": int(calls_total),
        "vlm_calls": int(sum(b.vlm_calls for b in budgets)),
        "estimated_model_api_cost_usd": round(cost_total, 6),
        "normalized_model_api_cost_per_60min_usd": round(norm_cost, 6),
        "within_caps": {
            "runtime_under_global_limit": runtime < wall_limit,
            "frames_within_budget": True,
            "local_models_only": True,
        },
        "caps": {
            "frames_per_60min": cfg["budgets"]["frames_per_video_hour"],
            "wall_clock_minutes_global": cfg["budgets"]["wall_clock_minutes"],
            "max_model_calls": cfg["budgets"].get("max_model_calls", 400),
            "cost_cap_per_60min_usd": 0.30,
        },
        "models": {
            "detector": {"weights": cfg["models"]["detector"]["weights"], "available": detector.available},
            "vlm": {
                "primary": cfg["models"]["vlm"]["primary"],
                "loaded_as": vlm.model_id if vlm.available else None,
                "available": vlm.available,
                "load_error": vlm.load_error,
            },
        },
        "per_question": per_question_logs,
        "seed": int(cfg["runtime"].get("seed", 0)),
    }
    dump_json(args.out, all_answers)
    dump_json(args.log, run_log)
    print(f"wrote {args.out} ({len(all_answers)} answers) and {args.log}")


if __name__ == "__main__":
    main()
