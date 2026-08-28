# Builderr submission

Send this information to `submit@builderr.ai`.

- Repository: https://github.com/San245o/kitchen-cctv-agent
- Branch: `master`
- Commit: use the latest commit on `master`
- Agent name: Kitchen CCTV Monitor Agent
- Python: 3.11+
- Models: `yolo26s.pt`, `google/siglip2-base-patch16-224`,
  `Qwen/Qwen3-VL-2B-Instruct`
- Model/API cost: `$0.00` per scored run (local models only)
- API keys: none

Evaluator setup:

```bash
python -m pip install -r requirements.txt
python answer.py --out .warmup_answers.json --log .warmup_run_log.json --download-only
```

The second command downloads and actually loads every declared component. It exits non-zero
if YOLO26s, SigLIP2, or Qwen3-VL-2B is unavailable. No 4B model is declared or attempted.

Scored run:

```bash
python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
```

The normal command also performs a best-effort warm-up before its internal budget clock and
always emits `answers.json` and `run_log.json` if video processing or a question handler fails.
