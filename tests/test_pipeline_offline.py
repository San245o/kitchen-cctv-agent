"""
Comprehensive Offline Test Suite for Kitchen CCTV Monitor v2.
Validates all pipeline components without requiring an active Gemini API key.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np

# Ensure v2 is on Python path
V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from pipeline.question_router import parse_timestamp_from_text, classify_question, group_questions_by_strategy
from pipeline.motion_indexer import MotionIndexer
from pipeline.visual_grounder import extract_frames_at_timestamps, burn_timestamp_overlay
from pipeline.cost_governor import CostGovernor
from pipeline.vlm_client import VLMClient


PASS_COUNT = 0
FAIL_COUNT = 0


def assert_test(name: str, condition: bool, details: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}" + (f" ({details})" if details else ""))
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name}" + (f" -> {details}" if details else ""))


def make_synthetic_video(path: str, duration_sec: float = 10.0, fps: int = 20) -> str:
    """Creates synthetic video with moving rectangle between 3s and 6s."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        ts = i / fps
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Static background
        frame[:] = (30, 30, 30)

        # Motion event between 3s and 6s
        if 3.0 <= ts <= 6.0:
            x = int(50 + (ts - 3.0) * 40)
            cv2.rectangle(frame, (x, 80), (x + 60, 140), (200, 200, 200), -1)

        out.write(frame)

    out.release()
    return path


def test_timestamp_parsing():
    print("\n--- Testing Deterministic Timestamp Parsing ---")
    assert_test("Parse MM:SS 00:45", parse_timestamp_from_text("Cook wearing cap at 00:45?") == 45.0)
    assert_test("Parse MM:SS 01:30", parse_timestamp_from_text("Status at 01:30 in prep area") == 90.0)
    assert_test("Parse HH:MM:SS 01:00:15", parse_timestamp_from_text("What happened at 01:00:15?") == 3615.0)
    assert_test("Parse T=10s", parse_timestamp_from_text("How many people at T=10s?") == 10.0)
    assert_test("Parse T=12.5s", parse_timestamp_from_text("Event at T=12.5s") == 12.5)
    assert_test("No timestamp present", parse_timestamp_from_text("At what timestamp was the first bag sealed?") is None)


def test_question_routing():
    print("\n--- Testing Question Classification & Routing ---")
    q1 = {"id": "q1", "type": "yes_no", "question": "Was cook wearing cap at 00:45?"}
    m1 = classify_question(q1)
    assert_test("Route point-in-time", m1["category"] == "POINT_IN_TIME" and m1["target_timestamp"] == 45.0)

    q2 = {"id": "q2", "type": "yes_no", "question": "Is the order number visible?"}
    m2 = classify_question(q2)
    assert_test("Route OCR detail", m2["category"] == "OCR_DETAIL" and m2["is_ocr"])

    q3 = {"id": "q3", "type": "timestamp", "question": "When was first sealed bag placed on handoff shelf?"}
    m3 = classify_question(q3)
    assert_test("Route station event / sequence", m3["category"] in ("STATION_EVENT", "TEMPORAL_SEQUENCE"))


def test_motion_indexing_and_extraction():
    print("\n--- Testing Motion Indexer & Frame Extraction ---")
    with tempfile.TemporaryDirectory() as td:
        vid_path = os.path.join(td, "test_clip.mp4")
        make_synthetic_video(vid_path, duration_sec=8.0, fps=20)

        indexer = MotionIndexer(sample_fps=2.0)
        info = indexer.index_video(vid_path)

        assert_test("Video duration measured", abs(info["duration_sec"] - 8.0) < 0.5, f"{info['duration_sec']}s")
        assert_test("Motion intervals detected", len(info["active_intervals"]) >= 1)

        # Test frame extraction at specific timestamp
        frames = extract_frames_at_timestamps(vid_path, [2.0, 4.5, 7.0])
        assert_test("Extracted 3 frames", len(frames) == 3)
        ts0, img0 = frames[0]
        assert_test("Frame shape valid", img0.shape[0] > 0 and img0.shape[1] > 0)


def test_cost_governor():
    print("\n--- Testing Cost Governor Ledger & Pricing ---")
    gov = CostGovernor()
    gov.record_source_duration(3600.0)  # 60 minutes
    gov.record_call(num_images=50, prompt_text="A" * 1000, response_text="B" * 500)

    cost = gov.get_estimated_cost_usd()
    assert_test("Cost is calculated and positive", cost > 0.0, f"${cost}")
    assert_test("Cost well below $0.30 cap", cost < 0.05, f"${cost} vs $0.30 cap")

    log_data = gov.get_run_log()
    assert_test("Log contains required fields", all(k in log_data for k in [
        "runtime_seconds", "frames_processed", "model_calls",
        "estimated_model_api_cost_usd", "normalized_model_api_cost_per_60min_usd",
        "budget_status"
    ]))
    assert_test("Budget status is PASS", log_data["budget_status"] == "PASS")

    # Test context caching discount calculation
    gov_cache = CostGovernor()
    gov_cache.record_source_duration(3600.0)
    # 388,500 cached tokens for 300 seconds
    gov_cache.record_cached_call(cached_tokens=388500, prompt_text="A" * 500, response_text="B" * 500, cache_lifetime_sec=300.0)
    cached_cost = gov_cache.get_estimated_cost_usd()
    # 388.5k tokens at $0.075/1M = $0.0291, storage = $0.0162 -> total around $0.045
    assert_test("Cached cost gets 90% discount rate", 0.03 < cached_cost < 0.08, f"${cached_cost}")
    assert_test("Context caching flag in run_log", gov_cache.get_run_log()["context_caching_used"] is True)


def test_vlm_output_normalization():
    print("\n--- Testing Output Schema & Not_Visible Normalization ---")
    vlm = VLMClient()
    mock_questions = [
        {"id": "q01", "type": "yes_no", "question": "Cap?"},
        {"id": "q02", "type": "count", "question": "People?"},
        {"id": "q03", "type": "yes_no", "question": "Slip visible?"},
    ]

    mock_llm_json = """
    {
      "audit_reasoning": "Inspected frames. Person wore a cap. Slip occluded.",
      "answers": [
        {"id": "q01", "answer": "yes", "confidence": 0.95, "evidence": [{"timestamp_start": 44.0, "timestamp_end": 46.0}]},
        {"id": "q02", "answer": 2, "confidence": 0.85, "evidence": [45.0]},
        {"id": "q03", "answer": "not visible", "confidence": 0.2, "evidence": [10.0]}
      ]
    }
    """

    results = vlm._parse_and_normalize(mock_llm_json, mock_questions, "test_clip")
    assert_test("Parsed all 3 questions", len(results) == 3)

    r1 = results[0]
    assert_test("Q01 answer is yes", r1["answer"] == "yes")
    assert_test("Q01 evidence format valid", len(r1["evidence"]) == 1 and r1["evidence"][0]["timestamp_start"] == 44.0)

    r3 = results[2]
    assert_test("Q03 normalized to 'not_visible'", r3["answer"] == "not_visible")
    assert_test("Q03 evidence strictly empty for not_visible", r3["evidence"] == [])


def test_cli_end_to_end():
    print("\n--- Testing Full CLI End-to-End Execution ---")
    with tempfile.TemporaryDirectory() as td:
        vid_dir = os.path.join(td, "videos")
        os.makedirs(vid_dir)
        vid_path = os.path.join(vid_dir, "sample_cctv.mp4")
        make_synthetic_video(vid_path, duration_sec=5.0)

        questions = [
            {"id": "q1", "video_id": "sample_cctv", "type": "yes_no", "question": "Cook wearing cap at 00:02?"},
            {"id": "q2", "video_id": "sample_cctv", "type": "count", "question": "How many items at T=3s?"},
            {"id": "q3", "video_id": "sample_cctv", "type": "yes_no", "question": "Is order number visible?"},
        ]
        q_path = os.path.join(td, "questions.json")
        with open(q_path, "w") as f:
            json.dump(questions, f)

        out_path = os.path.join(td, "answers.json")
        log_path = os.path.join(td, "run_log.json")

        cmd = [
            sys.executable,
            str(V2_ROOT / "answer.py"),
            "--videos", vid_dir,
            "--questions", q_path,
            "--out", out_path,
            "--log", log_path,
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert_test("CLI exits cleanly with code 0", res.returncode == 0, f"Code: {res.returncode}")

        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                answers = json.load(f)
            assert_test("Answers file has 3 answers", len(answers) == 3)
            ids = [a["id"] for a in answers]
            assert_test("All IDs match original order", ids == ["q1", "q2", "q3"])

        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                rlog = json.load(f)
            assert_test("Run log written with PASS status", rlog.get("budget_status") == "PASS")


def main():
    print("=" * 60)
    print("RUNNING KITCHEN CCTV v2 OFFLINE TEST SUITE")
    print("=" * 60)

    test_timestamp_parsing()
    test_question_routing()
    test_motion_indexing_and_extraction()
    test_cost_governor()
    test_vlm_output_normalization()
    test_cli_end_to_end()

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {PASS_COUNT} PASSED, {FAIL_COUNT} FAILED")
    print("=" * 60)

    if FAIL_COUNT > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
