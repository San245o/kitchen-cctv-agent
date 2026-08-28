import re

from .io_utils import parse_time_to_seconds


class Routed:
    def __init__(self):
        self.handler = "general_yesno"
        self.qtype = None
        self.target_time = None
        self.options = []
        self.phrase = ""
        self.first_last = None
        self.raw_question = ""


ORDER_SPLIT = re.compile(r"\s*,\s*|\s+or\s+", re.I)
FILLER = re.compile(
    r"^(?:at what timestamp|what time|when)\s*(?:was|were|did|is|are)?\b[\s:]*",
    re.I,
)
ORDER_HINT = re.compile(r"happened last|happened first|which .* (last|first)|before or after|order of", re.I)
TARGET_TEXT = re.compile(
    r"is the (.+?) visible|does the (.+?) (?:show|say|read)|(?:order|receipt|ticket|label|screen) (?:number|text)?",
    re.I,
)


def _extract_options(q: str):
    body = q.split(":", 1)[1] if ":" in q else q
    opts = [o.strip(" ?.") for o in ORDER_SPLIT.split(body) if len(o.strip(" ?.")) > 2]
    return opts[:6]


def route(question: str, qtype: str = None) -> Routed:
    q = question.strip()
    r = Routed()
    r.raw_question = q
    r.qtype = (qtype or "").lower().strip() or None
    low = q.lower()
    r.target_time = parse_time_to_seconds(q)

    t = r.qtype
    if t == "count":
        r.handler = "count"
    elif t == "timestamp":
        r.handler = "timestamp"
    elif t == "duration":
        r.handler = "duration"
    elif t == "ocr":
        r.handler = "ocr"
    elif t == "yes_no":
        r.handler = "state" if r.target_time is not None else "general_yesno"
    elif t == "multiple_choice" or t == "short_structured":
        opts = _extract_options(q)
        if ORDER_HINT.search(low) and len(opts) >= 2:
            r.handler = "order"
            r.options = opts
            r.first_last = "last" if re.search(r"\blast\b", low) else "first"
        else:
            r.handler = "mc"
            r.options = opts

    if r.handler == "general_yesno":
        if re.search(r"how many|number of (people|persons|workers)|count of", low):
            r.handler = "count"
        elif re.search(r"how long|duration|unattended|left alone|sat there", low):
            r.handler = "duration"
        elif ORDER_HINT.search(low):
            opts = _extract_options(q)
            if len(opts) >= 2:
                r.handler = "order"
                r.options = opts
                r.first_last = "last" if re.search(r"\blast\b", low) else "first"
            else:
                r.handler = "general_yesno"
        elif re.search(r"order number|visible.*read|readable|written|displayed|receipt|screen|label|text", low):
            r.handler = "ocr"
        elif re.search(r"\btimestamp\b|\bwhen\b|\bwhat time\b", low):
            r.handler = "timestamp"
        elif r.target_time is not None:
            r.handler = "state"

    if r.handler == "order":
        if not r.options:
            r.options = _extract_options(q)
        if not r.first_last:
            r.first_last = "last" if re.search(r"\blast\b", low) else "first"

    if r.handler in ("timestamp", "duration"):
        cleaned = FILLER.sub("", q).strip(" ?.")
        cleaned = re.sub(r"^the\s+", "", cleaned, flags=re.I)
        r.phrase = cleaned

    m = TARGET_TEXT.search(q)
    if m:
        groups = [g for g in m.groups() if g]
        r.phrase = groups[0] if groups else r.phrase
    return r
