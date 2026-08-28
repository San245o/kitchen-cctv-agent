import os
import time
import unittest
from types import SimpleNamespace

import numpy as np

from agent.budget import Budget
from agent.frames import FrameStore
from agent.handlers import handle_state, interpret_times
from agent.router import route
from agent.vlm import VLM


class RegressionTests(unittest.TestCase):
    def test_explicit_mmss_is_not_reinterpreted_as_minutes(self):
        routed = route("Was the cook wearing a cap at 00:45?", "yes_no")
        self.assertEqual(routed.target_time, 45.0)
        self.assertEqual(interpret_times(routed.target_time, 3600), [45.0])

    def test_ppe_state_does_not_reuse_person_box_as_answer(self):
        class Detector:
            def detect_persons(self, _img):
                return [(10, 10, 50, 100, 0.9)]

        class Vision:
            def yes_no(self, _img, _question):
                return "yes", 0.85

        ctx = SimpleNamespace(
            duration=3600,
            detector=Detector(),
            vlm=Vision(),
            budget=SimpleNamespace(exhausted=False),
            cfg={
                "thresholds": {
                    "consensus_of": 3,
                    "consensus_min_agree": 2,
                    "min_answer_confidence": 0.55,
                }
            },
            video_id="sample",
            zone_for=lambda _q: None,
            neighbor_frames=lambda _t, _n: [np.zeros((120, 160, 3), dtype=np.uint8)] * 3,
        )
        routed = SimpleNamespace(
            target_time=45.0,
            raw_question="Was the cook wearing a cap at 00:45?",
        )
        answer = handle_state(routed, ctx)
        self.assertEqual(answer["answer"], "yes")

    def test_generic_timed_yesno_is_not_forced_into_ppe_logic(self):
        class Vision:
            def verify_window(self, _frames, question):
                self.question = question
                return "yes", 0.85

        vision = Vision()
        ctx = SimpleNamespace(
            duration=3600,
            vlm=vision,
            cfg={
                "thresholds": {
                    "consensus_of": 3,
                    "min_answer_confidence": 0.55,
                }
            },
            video_id="sample",
            zone_for=lambda _q: None,
            neighbor_frames=lambda _t, _n: [np.zeros((10, 10, 3), dtype=np.uint8)] * 3,
        )
        routed = SimpleNamespace(
            target_time=900.0,
            raw_question="Did the worker touch the island at 15:00?",
        )
        answer = handle_state(routed, ctx)
        self.assertEqual(answer["answer"], "yes")
        self.assertIn("touch the island", vision.question)

    def test_zone_motion_respects_vertical_bounds(self):
        store = FrameStore.__new__(FrameStore)
        first = np.zeros((90, 160), dtype=np.uint8)
        second = first.copy()
        second[:30, :] = 255
        store.thumbs = {0.0: first, 1.0: second}
        store.motion = {0.0: 0.0, 1.0: 1.0}
        scores = store.zone_motion((0.0, 0.6, 1.0, 1.0))
        self.assertEqual(scores[1.0], 0.0)

    def test_vlm_grounding_coordinates_are_normalized_from_1000(self):
        vlm = VLM.__new__(VLM)
        vlm._generate = lambda _images, _prompt: '{"box_2d": [100, 200, 500, 600]}'
        box = vlm.locate(np.zeros((720, 1280, 3), dtype=np.uint8), "prep counter")
        self.assertEqual(box, (0.2, 0.1, 0.6, 0.5))

    def test_unreadable_video_still_returns_one_answer_per_question(self):
        from answer import process_video

        cfg = {
            "budgets": {
                "frames_per_video_hour": 1450,
                "coarse_fraction": 0.72,
                "wall_clock_minutes": 1,
                "max_model_calls": 10,
            },
            "sampling": {"coarse_stride_seconds": 3.0},
        }
        detector = SimpleNamespace(budget=None)
        vision = SimpleNamespace(budget=None)
        questions = [{"id": "q1", "type": "count", "question": "How many people?"}]
        answers, budget = process_video(
            "missing",
            os.path.join(os.getcwd(), "definitely-missing-video.mp4"),
            questions,
            cfg,
            detector,
            vision,
            time.perf_counter(),
            [],
        )
        self.assertIsNone(budget)
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0]["id"], "q1")
        self.assertEqual(answers[0]["answer"], "not_visible")

    def test_budget_records_auditable_frames_and_calls(self):
        cfg = {
            "frames_per_video_hour": 1450,
            "coarse_fraction": 0.72,
            "wall_clock_minutes": 1,
            "max_model_calls": 10,
        }
        budget = Budget(3600, cfg)
        self.assertTrue(budget.spend_coarse(timestamp=12.5))
        started = time.perf_counter()
        budget.log_call("model", "vlm_generate", started_at=started, duration_seconds=0.1)
        self.assertEqual(budget.frame_records[0].to_dict()["timestamp"], 12.5)
        self.assertEqual(budget.calls[0].to_dict()["kind"], "vlm_generate")


if __name__ == "__main__":
    unittest.main()
