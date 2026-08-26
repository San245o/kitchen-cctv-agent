# Kitchen CCTV Monitor Agent

Answers operational questions (caps/hairnets worn, person counts, event timestamps,
durations, event order, on-screen text) from fixed-camera kitchen footage. Built for
the builderr Kitchen CCTV challenge.

## Evaluator setup (reproducible warm-up path)

Cross-platform: plain `pip` only, no system packages, no compilation. Models are
fetched automatically from Hugging Face on first load.

```bash
# Linux / macOS
bash scripts/setup_models.sh

# Windows PowerShell
powershell scripts/setup_models.ps1    # uses CUDA 12.8 torch wheels (RTX 50-series)
```

Or manually:

```bash
pip install -r requirements.txt
python answer.py --out .warmup.json --log .warmup_log.json --download-only   # verify weights load
python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
```

First run downloads ~4.7 GB (YOLO26s + Qwen3-VL-2B). The VLM ladder is
Qwen3-VL-2B → Qwen3-VL-4B → Cosmos-Reason2-2B → degraded; the serving tier is
recorded in `run_log.json`. Model loading happens **before** the wall-clock
clock starts, so fixed load cost never eats a short clip's scaled budget.

Guarantees:

- `answer.py` **always writes a schema-valid `answers.json` and `run_log.json`** — even if
  model weights are missing, downloads fail, or an unexpected error occurs. In that case
  answers degrade to `not_visible` with reasons, and `run_log.json` records the failure.
- The VLM loads via a fallback ladder (see above). The tier that actually served is
  recorded in `run_log.json`.
- `--download-only` fetches and verifies all weights without answering anything; run it
  before the timed evaluation to keep model download out of the wall-clock budget.
- Minimal-deps mode: `pip install -r requirements-minimal.txt` alone supports a complete,
  valid (degraded) run.

## One-command run

```bash
python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
```

## Outputs

`answers.json` — one record per question:

```json
{
  "id": "q001",
  "answer": "yes",
  "confidence": 0.72,
  "evidence": [{"video_id": "sample_01", "timestamp_start": 91.2, "timestamp_end": 94.8}]
}
```

When the footage does not show enough, the answer is `not_visible` with a `reason`.
Guessing is never emitted by design.

`run_log.json` — runtime, frames processed, model calls, estimated cost
(local models = $0.00), cap compliance, per-question routing and budget usage.

## How it works

```
video ──> coarse skim (1 frame / ~3s) ──> YOLO26s person timeline + motion map
              │                                   │
              │                        question router (6 archetypes)
              │                                   │
              └───────── candidate moments ◄──────┘
                             │
                   crop + upscale (2–3×)
                             │
                 Qwen3-VL-4B verification (YES/NO/UNSURE, 2-of-3 frame consensus)
                             │
                structured observations + timestamps
                             │
              deterministic Python temporal rules
                             │
                 answer + evidence span   or   not_visible
```

- Counts come from detector boxes (never VLM guessing).
- Timestamps/durations/order are computed in Python from verified facts.
- Every decoded frame is counted once against the ~1,500 frames/hour budget.
- All models run locally → estimated model/API cost is $0.00 per 60 minutes.

## Install

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

GPU notes: NVIDIA RTX 50-series needs CUDA 12.8+ wheels:
`pip install torch --index-url https://download.pytorch.org/whl/cu128`
CPU-only runs work but will be slow; the code degrades gracefully when model
weights are unavailable (answers become `not_visible` rather than guesses).

First run downloads weights automatically: `yolo26s.pt` (~20MB) and
`Qwen/Qwen3-VL-2B-Instruct` (~4.4GB), with larger tiers as fallback. Set `HF_HOME`
to control cache location.

## Question archetypes routed

| Type | Example | Method |
|---|---|---|
| count | "How many people at the prep counter at 00:45?" | zone-filtered detector boxes, median over 3 frames |
| state/yes_no | "Cap or hairnet at 00:45?" | head crop ×3 zoom, VLM consensus |
| timestamp | "When was the first sealed bag placed?" | motion-ranked candidates → window verify → backward onset refinement |
| duration | "How long unattended?" | placement detection → next-interaction scan |
| order | "Which happened last: A, B, C?" | locate each event's time → sort in Python |
| ocr | "Is the order number visible?" | full-frame then quadrant crops, NOT_READABLE gate |

## Budget compliance

| Cap | Limit | This run |
|---|---|---|
| Model/API cost | $0.30 / 60 min video | **$0.00** (local models only) |
| Wall clock | 25 min eval | enforced internally (`budgets.wall_clock_minutes`) |
| Sampled frames | ~1,500 / 60 min | ledger-enforced split: 72% coarse skim, 28% fine verification |

## Determinism

Fixed sampling grid, greedy decoding (`do_sample=False`), seeded RNG
(`runtime.seed`). Outputs are reproducible given the same hardware and the same
wall-clock headroom; if a slow machine exhausts the global time budget mid-run,
later questions degrade to `not_visible` and this is visible in `run_log.json`.

## Performance notes

- Pre-fetch weights outside timed runs: `python answer.py --videos x --questions x --out x --log x --download-only`
- 8GB VRAM: run the VLM quantized (GGUF Q4_K_M ~2.7GB or AWQ INT4 ~3GB); bf16 4B does not fit.
- Global budget guards: wall clock (all videos combined), total frames, and a hard VLM-call cap (`budgets.max_model_calls`).

## Bench models (promotion triggers)

| Model | Status | Promote if |
|---|---|---|
| Cosmos-Reason2-2B | bench | beats Qwen3-VL-4B on sample bake-off (`models.vlm.alternate`) |
| RF-DETR-base | bench | YOLO misses people at your camera angle |
| PP-OCRv5 | bench | VLM OCR fails on burned-in clocks/receipts |
| fine-tuned PPE nano-YOLO | bench | cap/hairnet accuracy < target on samples |

## Licenses

Qwen3-VL: Apache-2.0 · YOLO26/Ultralytics: AGPL-3.0 · challenge-safe.
