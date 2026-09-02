"""
Gemini 3.8 Flash Client with Visibility Gate, Strict Evidence Formatting,
and Google Context Caching support ($0.075 / 1M token rate).
"""

import sys
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import PIL.Image
import cv2

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


SYSTEM_PROMPT = """You are a rigorous, highly disciplined CCTV Operational Auditor for commercial kitchens.
Your job is to answer operational questions from fixed-view kitchen camera frames with absolute ground truth.

### RULES:
1. GROUNDING OVER GUESSING:
   - If a subject, person, hairnet, container seal, or order number is blurred, occluded, or cannot be definitively confirmed from the frames, you MUST answer "not_visible".
   - Guessing is severely penalized. "not_visible" earns full credit when the video does not show enough.

2. PHYSICAL TIMESTAMPS:
   - High-contrast timestamp tags [T=XX.XXs] are burned in the top-left of every frame.
   - Read this text directly. For your evidence timestamps, cite the exact [T=...] values from the frames that support your answer.

3. REQUIRED JSON OUTPUT SCHEMA:
Return a JSON object with this exact structure:
{
  "audit_reasoning": "Brief check: was subject visible? What exact [T=...] frames prove the answer?",
  "answers": [
    {
      "id": "question_id",
      "answer": <depends on question: "yes", "no", integer count, float timestamp/duration, string category, or "not_visible">,
      "confidence": <float 0.0 to 1.0>,
      "evidence": [
        {
          "timestamp_start": <float seconds>,
          "timestamp_end": <float seconds>
        }
      ]
    }
  ]
}

CRITICAL: If the answer is "not_visible", the "evidence" array MUST BE EMPTY: [].
Do NOT guess timestamps for things that are not visible.
"""


class VLMClient:
    MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    MIN_CACHE_TOKENS = 32768   # Google AI Studio minimum token threshold for caching
    TOKENS_PER_IMAGE = 259

    def __init__(self):
        self.client = None
        if _GENAI_AVAILABLE:
            try:
                import os
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    self.client = genai.Client(api_key=api_key)
                else:
                    self.client = genai.Client()
            except Exception as e:
                print(f"[vlm] Note: Gemini client init: {e}", file=sys.stderr)

    def is_available(self) -> bool:
        return self.client is not None

    def create_context_cache(
        self,
        frames: List[Tuple[float, Any]],
        ttl: str = "300s"
    ) -> Optional[Any]:
        """
        Uploads frames to Google Context Cache (qualifies only if >= 32,768 tokens, ~130 frames).
        Allows subsequent queries at the 90% discounted $0.075 / 1M token rate.
        """
        if not self.client or not frames:
            return None

        est_tokens = len(frames) * self.TOKENS_PER_IMAGE
        if est_tokens < self.MIN_CACHE_TOKENS:
            # Below Google minimum cache threshold, skip caching
            return None

        try:
            pil_images = []
            for ts, bgr in frames:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_images.append(PIL.Image.fromarray(rgb))

            cache = self.client.caches.create(
                model=self.MODELS[0],
                config=types.CreateCachedContentConfig(
                    contents=pil_images,
                    ttl=ttl,
                    system_instruction=SYSTEM_PROMPT,
                )
            )
            print(f"  [Cache] Successfully cached {len(frames)} frames ({est_tokens} tokens) on Google servers (TTL: {ttl})")
            return cache
        except Exception as e:
            print(f"  [Cache] Warning: Context cache creation failed ({e}), falling back to direct mode", file=sys.stderr)
            return None

    def delete_cache(self, cache: Any):
        """Safely cleans up server-side context cache."""
        if not self.client or not cache:
            return
        try:
            cache_name = getattr(cache, "name", str(cache))
            self.client.caches.delete(name=cache_name)
            print(f"  [Cache] Cleaned up cache: {cache_name}")
        except Exception as e:
            print(f"  [Cache] Cleanup notice: {e}", file=sys.stderr)

    def call_gemini_with_cache(
        self,
        cache: Any,
        questions: List[Dict[str, Any]],
        video_id: str
    ) -> Tuple[List[Dict[str, Any]], str, str, int]:
        """
        Queries Gemini using pre-cached frames at $0.075 / 1M tokens.
        Returns (parsed_answers, user_prompt, response_text, cached_tokens).
        """
        fallback = [
            {"id": q["id"], "answer": "not_visible", "confidence": 0.0, "evidence": []}
            for q in questions
        ]

        if not self.client or not cache:
            return fallback, "", "", 0

        user_prompt = self._build_prompt(questions, video_id, 0)
        cache_name = getattr(cache, "name", str(cache))
        cached_tokens = getattr(cache, "usage_metadata", {}).get("total_token_count", 350000) if hasattr(cache, "usage_metadata") else 350000

        raw_response_text = ""
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.MODELS[0],
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache_name,
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                raw_response_text = resp.text or ""
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = min(2 ** attempt * 10, 60)
                    time.sleep(wait)
                    continue
                else:
                    break

        if not raw_response_text:
            return fallback, user_prompt, "", cached_tokens

        parsed_answers = self._parse_and_normalize(raw_response_text, questions, video_id)
        return parsed_answers, user_prompt, raw_response_text, cached_tokens

    def _build_prompt(self, questions: List[Dict[str, Any]], video_id: str, num_frames: int) -> str:
        prompt_lines = [
            f"AUDIT VIDEO CLIP: {video_id}",
        ]
        if num_frames > 0:
            prompt_lines.append(f"You are provided {num_frames} inspected CCTV frames with timestamp overlays [T=XX.XXs].")
        else:
            prompt_lines.append("Refer to the cached video frames.")

        prompt_lines.append("\nQUESTIONS TO ANSWER:")
        for q in questions:
            prompt_lines.append(f"- ID: {q['id']}")
            prompt_lines.append(f"  Type: {q.get('type', 'general')}")
            prompt_lines.append(f"  Question: {q.get('question', q.get('prompt', ''))}")
            if q.get("options"):
                prompt_lines.append(f"  Options: {json.dumps(q['options'])}")
            prompt_lines.append("")

        prompt_lines.append("Analyze the frames and return your JSON object with 'answers' now.")
        return "\n".join(prompt_lines)

    def call_gemini(
        self,
        frames: List[Tuple[float, Any]],
        questions: List[Dict[str, Any]],
        video_id: str,
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        Direct VLM call for targeted inspection (< 32,768 tokens).
        Returns: (parsed_answers_list, raw_prompt_str, raw_response_str)
        """
        fallback_answers = [
            {
                "id": q["id"],
                "answer": "not_visible",
                "confidence": 0.0,
                "evidence": []
            }
            for q in questions
        ]

        if not self.client or not frames:
            return fallback_answers, "", ""

        # Convert OpenCV BGR to PIL RGB Images
        pil_images = []
        for ts, bgr in frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(PIL.Image.fromarray(rgb))

        user_prompt = self._build_prompt(questions, video_id, len(frames))
        contents = pil_images + [user_prompt]

        raw_response_text = ""
        for model_name in self.MODELS:
            for attempt in range(3):
                try:
                    resp = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    )
                    raw_response_text = resp.text or ""
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait = min(2 ** attempt * 10, 60)
                        time.sleep(wait)
                        continue
                    else:
                        break
            if raw_response_text:
                break

        if not raw_response_text:
            return fallback_answers, user_prompt, ""

        parsed_answers = self._parse_and_normalize(raw_response_text, questions, video_id)
        return parsed_answers, user_prompt, raw_response_text

    def _parse_and_normalize(
        self,
        json_str: str,
        questions: List[Dict[str, Any]],
        video_id: str
    ) -> List[Dict[str, Any]]:
        """
        Parses output JSON and standardizes fields to match the official spec.
        """
        try:
            clean_str = json_str.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            data = json.loads(clean_str.strip())
        except Exception:
            return [
                {"id": q["id"], "answer": "not_visible", "confidence": 0.0, "evidence": []}
                for q in questions
            ]

        raw_answers = data.get("answers", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        answer_by_id = {item.get("id"): item for item in raw_answers if isinstance(item, dict) and "id" in item}

        normalized_list = []
        for q in questions:
            qid = q["id"]
            if qid in answer_by_id:
                item = answer_by_id[qid]
                ans = item.get("answer")

                # Normalize "not visible" variations to standard "not_visible"
                if ans is None or str(ans).strip().lower() in ("not visible", "not_visible", "none", "n/a", "null", "unknown"):
                    ans = "not_visible"
                    evidence = []
                else:
                    raw_evidence = item.get("evidence", [])
                    evidence = []
                    if isinstance(raw_evidence, list):
                        for ev in raw_evidence:
                            if isinstance(ev, dict) and "timestamp_start" in ev:
                                evidence.append({
                                    "video_id": video_id,
                                    "timestamp_start": float(ev["timestamp_start"]),
                                    "timestamp_end": float(ev.get("timestamp_end", ev["timestamp_start"]))
                                })
                            elif isinstance(ev, (int, float)):
                                evidence.append({
                                    "video_id": video_id,
                                    "timestamp_start": float(ev),
                                    "timestamp_end": float(ev)
                                })
                    elif isinstance(raw_evidence, (int, float)):
                        evidence.append({
                            "video_id": video_id,
                            "timestamp_start": float(raw_evidence),
                            "timestamp_end": float(raw_evidence)
                        })

                conf = float(item.get("confidence", 0.8)) if ans != "not_visible" else 0.0
                conf = max(0.0, min(1.0, conf))

                normalized_list.append({
                    "id": qid,
                    "answer": ans,
                    "confidence": round(conf, 2),
                    "evidence": evidence
                })
            else:
                normalized_list.append({
                    "id": qid,
                    "answer": "not_visible",
                    "confidence": 0.0,
                    "evidence": []
                })

        return normalized_list
