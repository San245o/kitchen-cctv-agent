# Builderr Kitchen CCTV Challenge Round 1 Submission

Email this information to: submit@builderr.ai
Subject: Kitchen CCTV monitor submission

---

Hi Builderr Team,

Kitchen CCTV monitor Round 1 submission:

- Repo URL: https://github.com/San245o/kitchen-cctv-agent
- Branch: master
- Agent name: Kitchen CCTV Gemini 3.8 Flash Agent
- Models or APIs used: gemini-3.8-flash (with fallbacks to gemini-2.5-flash, gemini-2.0-flash)
- Expected cost per scored run: ~.005 - .02 USD per 60 minutes (well under .30 cap)
- Anything I should know:
  - Physical timestamp OCR burn-in directly into frame pixels for sub-2-second temporal precision.
  - Question-routed coarse-to-fine inspection: isolates high-resolution native crops for fine details (caps, hairnets, slips).
  - Google Context Caching supported for large timeline sweeps, cutting input costs to $0.075 / 1M tokens.
  - Zero-guessing discipline: returns 
ot_visible with empty evidence [] when details are occluded.
  - Single reproducible entry point: python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
  - Fully passes 28-test offline suite (python tests/test_pipeline_offline.py).

---

## Evaluator Instructions

1. Set your Gemini API key:
   export GOOGLE_API_KEY="your_key"
   # or
   export GEMINI_API_KEY="your_key"

2. Install dependencies:
   pip install -r requirements.txt

3. Execute run:
   python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
