import re

from .io_utils import parse_time_to_seconds


class Routed:
    def __init__(self):
        self.handler = "general_yesno"
        self.target_time = None
        self.options = []
        self.phrase = ""
        self.first_last = None


ORDER_SPLIT = re.compile(r"\s*,\s*|\s+or\s+", re.I)
FILLER = re.compile(
    r"^(at what timestamp|what time|when)(was|were|did|is|are)?\b[\s:]*", re.I
)


def route(question: str) -> Routed:
    q = question.strip()
    r = Routed()
    r.raw_question = q
    low = q.lower()
    r.target_time = parse_time_to_seconds(q)

    if re.search(r"how many|number of (people|persons|workers)|count of", low):
        r.handler = "count"
    elif re.search(r"how long|duration|unattended|left alone|sat there", low):
        r.handler = "duration"
    elif re.search(r"happened last|happened first|which .* (last|first)|before or after", low):
        r.handler = "order"
    elif re.search(r"order number|visible.*read|readable|written|displayed|receipt|screen|label|text", low):
        r.handler = "ocr"
    elif re.search(r"\btimestamp\b|\bwhen\b|\bwhat time\b", low):
        r.handler = "timestamp"
    elif r.target_time is not None:
        r.handler = "state"
    else:
        m = re.search(r"^(?:which|what)\s+(?:happened|event)\s+(last|first)", low)
        if m:
            r.handler = "order"
            r.first_last = m.group(1)

    if r.handler == "order":
        body = q.split(":", 1)[1] if ":" in q else q
        opts = [o.strip(" ?.") for o in ORDER_SPLIT.split(body) if len(o.strip(" ?.")) > 2]
        r.options = opts[:6]
        if not r.first_last:
            r.first_last = "last" if re.search(r"\blast\b", low) else "first"
    if r.handler in ("timestamp", "duration"):
        cleaned = FILLER.sub("", q).strip(" ?.")
        cleaned = re.sub(r"^the\s+", "", cleaned, flags=re.I)
        r.phrase = cleaned
    return r
