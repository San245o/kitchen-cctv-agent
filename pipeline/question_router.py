"""
Question classification and deterministic timestamp routing.
"""

import re
from typing import Dict, Any, Optional, List


def parse_timestamp_from_text(text: str) -> Optional[float]:
    """
    Extract deterministic timestamp in seconds from question text.
    Handles:
      - '00:45' -> 45.0
      - '01:30' -> 90.0
      - '01:15:30' -> 4530.0
      - 'T=10s', 'T=12.5s', 'T=15'
      - 'at 45s', 'around 10 seconds'
    """
    # 1. Match MM:SS or HH:MM:SS
    # Look for patterns like 00:45 or 12:30 or 01:23:45
    colon_match = re.search(r'\b(?:at\s+)?(\d{1,2}):(\d{2})(?::(\d{2}))?\b', text, re.IGNORECASE)
    if colon_match:
        parts = [p for p in colon_match.groups() if p is not None]
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return float(minutes * 60 + seconds)
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return float(hours * 3600 + minutes * 60 + seconds)

    # 2. Match T=XX or T=XX.Xs
    t_match = re.search(r'\bT\s*=\s*(\d+(?:\.\d+)?)\s*s?\b', text, re.IGNORECASE)
    if t_match:
        return float(t_match.group(1))

    # 3. Match 'at 45s' or 'at 45 seconds' or 'at timestamp 45'
    sec_match = re.search(r'\b(?:at|around|timestamp)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\b', text, re.IGNORECASE)
    if sec_match:
        val = float(sec_match.group(1))
        # Sanity check: don't match question IDs or small counts like "at station 1"
        if "station" not in text.lower() and "count" not in text.lower():
            return val

    return None


def classify_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a question to route it through the optimal inspection path.
    Returns metadata with:
      - 'category': 'POINT_IN_TIME', 'OCR_DETAIL', 'STATION_EVENT', 'TEMPORAL_SEQUENCE', or 'GENERAL'
      - 'target_timestamp': float or None
      - 'stations': list of detected stations
      - 'is_ocr': bool
    """
    prompt = q.get("question", q.get("prompt", ""))
    q_type = q.get("type", "")

    timestamp = parse_timestamp_from_text(prompt)

    # Detect station keywords
    stations = []
    station_keywords = {
        "handoff shelf": ["handoff", "hand-off", "shelf"],
        "prep counter": ["prep", "cutting", "slicing", "chopping"],
        "stove": ["stove", "pan", "cooker"],
        "fryer line": ["fryer", "frying"],
        "packing station": ["pack", "packing", "bagging"],
        "sink": ["sink", "wash", "dish"],
    }
    prompt_lower = prompt.lower()
    for station_name, kws in station_keywords.items():
        if any(kw in prompt_lower for kw in kws):
            stations.append(station_name)

    # Detect OCR/Text intent
    is_ocr = any(w in prompt_lower for w in [
        "order number", "order #", "receipt", "ticket", "label", "screen", "readable"
    ])

    # Determine category
    if timestamp is not None:
        category = "POINT_IN_TIME"
    elif is_ocr:
        category = "OCR_DETAIL"
    elif q_type in ("duration", "timestamp") or any(w in prompt_lower for w in ["which happened last", "order of events", "sequence", "how long", "duration"]):
        category = "TEMPORAL_SEQUENCE"
    elif stations:
        category = "STATION_EVENT"
    else:
        category = "GENERAL"

    return {
        "category": category,
        "target_timestamp": timestamp,
        "stations": stations,
        "is_ocr": is_ocr,
        "raw": q,
    }


def group_questions_by_strategy(questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups questions by execution strategy to maximize batching efficiency.
    - 'point_in_time': grouped by similar target timestamps (within +/- 3s)
    - 'event_search': questions requiring timeline candidate scans
    - 'ocr_detail': questions checking slips/screens
    - 'general': remaining questions
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "point_in_time": [],
        "event_search": [],
        "ocr_detail": [],
        "general": [],
    }

    for q in questions:
        meta = classify_question(q)
        cat = meta["category"]
        if cat == "POINT_IN_TIME":
            grouped["point_in_time"].append(meta)
        elif cat == "OCR_DETAIL":
            grouped["ocr_detail"].append(meta)
        elif cat in ("STATION_EVENT", "TEMPORAL_SEQUENCE"):
            grouped["event_search"].append(meta)
        else:
            grouped["general"].append(meta)

    return grouped
