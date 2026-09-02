#!/usr/bin/env python3
"""
Next-Gen Kitchen CCTV Monitor (v2) — Builderr Challenge CLI.

Usage:
    python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from pipeline.question_router import group_questions_by_strategy
from pipeline.motion_indexer import MotionIndexer
from pipeline.visual_grounder import extract_frames_at_timestamps, extract_window_frames
from pipeline.cost_governor import CostGovernor
from pipeline.vlm_client import VLMClient


def resolve_video_path(videos_dir: Path, video_id: str) -> Path:
    """Finds video file matching video_id with standard extensions."""
    for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        candidate = videos_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
        candidate_upper = videos_dir / f"{video_id}{ext.upper()}"
        if candidate_upper.exists():
            return candidate_upper

    # Check if video_id itself has an extension
    direct = videos_dir / video_id
    if direct.exists():
        return direct

    # Try substring match
    for f in videos_dir.iterdir():
        if f.is_file() and video_id.lower() in f.stem.lower():
            return f

    return None


def process_video_questions(
    video_path: Path,
    video_id: str,
    questions: List[Dict[str, Any]],
    vlm: VLMClient,
    governor: CostGovernor,
    indexer: MotionIndexer,
) -> List[Dict[str, Any]]:
    """
    Executes question-driven coarse-to-fine inspection for a single video.
    """
    print(f"\n[Video: {video_id}] Indexing activity & routing {len(questions)} questions...")

    # Fast Tier 1 scan with OpenCV MotionIndexer
    motion_info = indexer.index_video(str(video_path))
    duration_sec = motion_info.get("duration_sec", 0.0)
    governor.record_source_duration(duration_sec)
    print(f"  [Indexer] Video duration: {duration_sec:.1f}s | Active intervals: {len(motion_info['active_intervals'])}")

    grouped_qs = group_questions_by_strategy(questions)
    video_answers: List[Dict[str, Any]] = []

    # Strategy 1: Point-in-time questions (T ± 2s)
    pit_group = grouped_qs.get("point_in_time", [])
    if pit_group:
        # Group questions sharing near-identical timestamps (within 3 seconds)
        time_clusters: Dict[float, List[Dict[str, Any]]] = {}
        for q_meta in pit_group:
            ts = q_meta["target_timestamp"]
            # Find an existing cluster within 3s
            matched_cluster = None
            for cluster_ts in time_clusters:
                if abs(cluster_ts - ts) <= 3.0:
                    matched_cluster = cluster_ts
                    break
            if matched_cluster is not None:
                time_clusters[matched_cluster].append(q_meta["raw"])
            else:
                time_clusters[ts] = [q_meta["raw"]]

        for anchor_ts, q_list in time_clusters.items():
            print(f"  [Route: Point-in-Time] Querying {len(q_list)} Qs at T={anchor_ts:.1f}s")
            # Extract 11 high-density frames at 2 FPS across [anchor - 2.5s, anchor + 2.5s]
            ts_to_fetch = [
                round(anchor_ts + delta, 2)
                for delta in (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
                if (anchor_ts + delta) >= 0.0 and (anchor_ts + delta) <= duration_sec
            ]
            frames = extract_frames_at_timestamps(str(video_path), ts_to_fetch)
            if frames:
                answers, prompt_text, resp_text = vlm.call_gemini(frames, q_list, video_id)
                governor.record_call(len(frames), prompt_text, resp_text)
                video_answers.extend(answers)
            else:
                video_answers.extend([
                    {"id": q["id"], "answer": "not_visible", "confidence": 0.0, "evidence": []}
                    for q in q_list
                ])

    # Strategy 2: Event search (actions, first/last, state changes)
    event_group = grouped_qs.get("event_search", [])
    if event_group:
        raw_event_qs = [item["raw"] for item in event_group]
        print(f"  [Route: Event Search] Querying {len(raw_event_qs)} Qs across dense candidate intervals")
        
        # Build dense temporal candidate set:
        candidate_ts = []
        # 1. Sample at 1.0 FPS across all active motion bursts (up to 200 burst frames)
        for start, end in motion_info.get("active_intervals", [])[:15]:
            curr = start
            while curr <= end and len(candidate_ts) < 200:
                candidate_ts.append(round(curr, 2))
                curr += 1.0

        # 2. Add continuous coarse timeline coverage (every 20s) across the entire video
        if duration_sec > 0:
            curr = 0.0
            while curr <= duration_sec and len(candidate_ts) < 300:
                candidate_ts.append(round(curr, 2))
                curr += 20.0
        elif not candidate_ts:
            candidate_ts = [0.0, 2.0, 5.0]

        candidate_ts = sorted(list(set(candidate_ts)))[:250]

        frames = extract_frames_at_timestamps(str(video_path), candidate_ts)
        if frames:
            # Check if cache is beneficial (>= 32,768 tokens, ~130 frames)
            cache = vlm.create_context_cache(frames, ttl="300s")
            if cache:
                answers, prompt_text, resp_text, cached_tokens = vlm.call_gemini_with_cache(cache, raw_event_qs, video_id)
                governor.record_cached_call(cached_tokens, prompt_text, resp_text)
                vlm.delete_cache(cache)
            else:
                answers, prompt_text, resp_text = vlm.call_gemini(frames, raw_event_qs, video_id)
                governor.record_call(len(frames), prompt_text, resp_text)
            video_answers.extend(answers)
        else:
            video_answers.extend([
                {"id": q["id"], "answer": "not_visible", "confidence": 0.0, "evidence": []}
                for q in raw_event_qs
            ])

    # Strategy 3: OCR Detail (order number, tickets, screens)
    ocr_group = grouped_qs.get("ocr_detail", [])
    if ocr_group:
        raw_ocr_qs = [item["raw"] for item in ocr_group]
        print(f"  [Route: OCR Detail] Querying {len(raw_ocr_qs)} Qs for readable slips/screens")
        # Check active counter moments at high resolution
        candidate_ts = motion_info["activity_spikes"][:8] if motion_info["activity_spikes"] else [5.0, 10.0, 15.0, 20.0]
        frames = extract_frames_at_timestamps(str(video_path), candidate_ts, max_width=1024)
        if frames:
            answers, prompt_text, resp_text = vlm.call_gemini(frames, raw_ocr_qs, video_id)
            governor.record_call(len(frames), prompt_text, resp_text)
            video_answers.extend(answers)
        else:
            video_answers.extend([
                {"id": q["id"], "answer": "not_visible", "confidence": 0.0, "evidence": []}
                for q in raw_ocr_qs
            ])

    # Strategy 4: General questions
    general_group = grouped_qs.get("general", [])
    if general_group:
        raw_gen_qs = [item["raw"] for item in general_group]
        print(f"  [Route: General] Querying {len(raw_gen_qs)} Qs with representative keyframes")
        coarse_ts = [round(i * (duration_sec / 8.0), 2) for i in range(8)] if duration_sec > 0 else [0.0]
        frames = extract_frames_at_timestamps(str(video_path), coarse_ts)
        if frames:
            answers, prompt_text, resp_text = vlm.call_gemini(frames, raw_gen_qs, video_id)
            governor.record_call(len(frames), prompt_text, resp_text)
            video_answers.extend(answers)
        else:
            video_answers.extend([
                {"id": q["id"], "answer": "not_visible", "confidence": 0.0, "evidence": []}
                for q in raw_gen_qs
            ])

    return video_answers


def main():
    parser = argparse.ArgumentParser(description="Kitchen CCTV Monitor v2")
    parser.add_argument("--videos", required=True, help="Path to videos directory")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--out", required=True, help="Path to output answers JSON file")
    parser.add_argument("--log", default=None, help="Path to output run log JSON file")
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    questions_file = Path(args.questions)
    out_file = Path(args.out)
    log_file = Path(args.log) if args.log else out_file.with_name("run_log.json")

    if not questions_file.exists():
        print(f"ERROR: Questions file not found: {questions_file}", file=sys.stderr)
        sys.exit(1)

    with open(questions_file, "r") as f:
        questions_data = json.load(f)

    # Initialize modular pipeline components
    governor = CostGovernor()
    vlm = VLMClient()
    indexer = MotionIndexer()

    # Group questions by video_id
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for q in questions_data:
        vid = q.get("video_id", "default")
        by_video.setdefault(vid, []).append(q)

    all_answers: List[Dict[str, Any]] = []

    for vid, q_list in by_video.items():
        vpath = resolve_video_path(videos_dir, vid)
        if not vpath or not vpath.exists():
            print(f"WARNING: Video '{vid}' not found in {videos_dir}. Returning 'not_visible' for {len(q_list)} questions.")
            for q in q_list:
                all_answers.append({
                    "id": q["id"],
                    "answer": "not_visible",
                    "confidence": 0.0,
                    "evidence": []
                })
            continue

        answers = process_video_questions(vpath, vid, q_list, vlm, governor, indexer)
        all_answers.extend(answers)

    # Ensure all original question IDs are present and in order
    answer_map = {a["id"]: a for a in all_answers}
    final_answers = []
    for q in questions_data:
        qid = q["id"]
        if qid in answer_map:
            final_answers.append(answer_map[qid])
        else:
            final_answers.append({
                "id": qid,
                "answer": "not_visible",
                "confidence": 0.0,
                "evidence": []
            })

    # Ensure parent directory exists for output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_answers, f, indent=2)

    # Write official run log
    run_log_data = governor.get_run_log()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(run_log_data, f, indent=2)

    print("\n" + "=" * 60)
    print("AUDIT EXECUTION COMPLETE")
    print("=" * 60)
    print(f"  Answers written: {len(final_answers)} -> {out_file}")
    print(f"  Run log written: {log_file}")
    print(f"  Runtime:         {run_log_data['runtime_seconds']}s")
    print(f"  Frames:          {run_log_data['frames_processed']}")
    print(f"  Model Calls:     {run_log_data['model_calls']}")
    print(f"  Est Cost:        ${run_log_data['estimated_model_api_cost_usd']}")
    print(f"  Norm Cost/60m:   ${run_log_data['normalized_model_api_cost_per_60min_usd']} (Cap: $0.30)")
    print(f"  Budget Status:   {run_log_data['budget_status']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
