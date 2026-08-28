# Architecture rationale

This agent is intentionally a small, in-process adaptation of stronger long-video search
systems. It does not reproduce their infrastructure or pretend that full-video captioning is
affordable under the challenge limits.

## What the stronger systems agree on

NVIDIA's Video Search and Summarization (VSS) blueprint separates video ingestion from
retrieval. It samples each video chunk, generates visual embeddings and timestamped metadata,
then uses multi-embedding retrieval, fusion, reranking, and a clip critic before answering.
Its current deployment targets hardware such as L40S/A100/H100-class GPUs, so Milvus,
Elasticsearch, Kafka, graph RAG, an 8B VLM, and a separate LLM would be counterproductive for
this challenge.

Research systems converge on the same useful algorithmic core:

- Video-in-the-Loop and LongVT: skim globally, localize spans, then reallocate frames to the
  selected spans under a fixed token/frame budget.
- LVNet: hierarchical scene/keyframe selection beats uniform frame selection.
- DeVi: hierarchical event retrieval plus temporal memory and self-consistency checking.
- VideoHV-Agent: form an explicit hypothesis, retrieve decision-relevant observations, then
  verify rather than aggregating generic captions.
- LongVideoAgent: keep grounding and fine-grained visual inspection as separate tools in a
  bounded reasoning loop.

## Challenge-sized implementation

For one run, the video itself is the database:

1. Decode a frame-capped timeline and record every sampled timestamp.
2. Encode the coarse frames once with SigLIP2 and encode all question/event phrases once.
3. Fuse semantic similarity with zone-aware motion, then apply temporal non-maximum
   suppression so the candidate budget covers distinct moments.
4. Route each question to a typed verifier. Counts use person detection; PPE uses head crops;
   timed actions use chronological multi-frame checks; OCR uses targeted crops; event order is
   deterministic sorting of separately verified timestamps.
5. Refine event boundaries with nearby frames and require visual agreement. If the evidence is
   weak or unreadable, return `not_visible`.
6. Emit the exact decoded frames and individual model calls in the run log.

This keeps the important VSS ideas—multi-signal indexing, fusion, reranking/criticism, and
evidence-first answers—without a service mesh or database.

## Model choices

- `google/siglip2-base-patch16-224` (0.4B, Apache-2.0): native Transformers support and strong
  image-text retrieval. It is used only to rank the already-budgeted coarse frames.
- `Qwen/Qwen3-VL-2B-Instruct`: long-video/timestamp-aware VLM with strong OCR and spatial
  perception, while remaining plausible on an 8GB evaluator GPU. The previous 4B-first setup
  did not complete in the public evaluation, so this submission declares and warms only 2B.
- `yolo26s.pt`: fast person counts and head crops; the VLM is not asked to invent counts.

Cosmos Reason 2 is highly relevant to physical/video reasoning and is used by NVIDIA VSS, but
the public Hugging Face checkpoint is gated and NVIDIA documents substantially larger validated
hardware for its VSS deployment. It remains a research candidate and is not part of the
reproducible evaluator setup. InternVideo2.5 and newer temporal-grounding agents are excellent research baselines,
but their 8B-class footprints do not fit the evaluator risk profile.

## Sources

- NVIDIA VSS blueprint: https://build.nvidia.com/nvidia/video-search-and-summarization/blueprintcard
- NVIDIA VSS architecture: https://docs.nvidia.com/vss/2.2.0/content/architecture.html
- NVIDIA VSS search workflow: https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/main/skills/vss-search-archive/SKILL.md
- Qwen3-VL-2B model card: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- SigLIP2 model card: https://huggingface.co/google/siglip2-base-patch16-224
- Cosmos Reason 2 model card: https://huggingface.co/nvidia/Cosmos-Reason2-2B
- Florence-2 model card: https://huggingface.co/microsoft/Florence-2-base
- Video-in-the-Loop: https://arxiv.org/abs/2510.04022
- LongVT: https://github.com/agentic-practice/agentic_longvideo
- LVNet: https://github.com/jongwoopark7978/LVNet
- DeVi / DeVE-QA: https://github.com/QHUni/DeVE-QA
- VideoHV-Agent: https://github.com/Haorane/VideoHV-Agent
- LongVideoAgent: https://github.com/longvideoagent/LongVideoAgent
