# Thesis Outline — Needle

Working title: **Training an Evidence-Seeking Search Agent with Reinforcement Learning (GRPO) for Multi-Hop Question Answering**

Standard 5-chapter structure; adapt chapter names/numbering to the university template.

---

## Abstract
- Problem: LLMs answering multi-hop questions guess or skip evidence; prompting alone does not teach good search behavior.
- Approach: an agent that issues `SEARCH[...]` / `ANSWER[...] CITATIONS[...]` actions over HotpotQA paragraphs (BM25), trained with GRPO + LoRA on Qwen2.5-7B-Instruct using a reward for answer, evidence, citations, and search cost.
- Result: _fill in_ — EM/F1, evidence coverage, citation precision, searches per episode vs. the un-tuned baseline.
- Takeaway: _fill in_.

## Chapter 1 — Introduction
1.1 Background: LLMs, hallucination, retrieval-augmented generation (RAG), agentic RAG.
1.2 Problem statement: answer-only rewards and prompting do not ensure the agent finds and cites the right evidence or searches efficiently.
1.3 Research questions
- RQ1: Can GRPO training improve answer accuracy of a 7B search agent on HotpotQA over the un-tuned model?
- RQ2: Does a reward that includes evidence and citation terms improve grounding (evidence coverage, citation precision)?
- RQ3: How does training change search behavior (searches per episode, invalid actions, shortcut answers)?
1.4 Objectives
1.5 Scope and limitations: HotpotQA distractor setting, BM25 over per-question paragraphs (not open web), 7B LoRA, ~$15–30 compute, small holdout (300).
1.6 Contributions: an open, reproducible environment + reward + GRPO pipeline; an empirical study at small budget.
1.7 Thesis structure

## Chapter 2 — Literature Review
2.1 Large language models and instruction tuning (Qwen2.5).
2.2 Retrieval-augmented generation; BM25.
2.3 Multi-hop QA and HotpotQA (supporting facts, distractor setting, EM/F1).
2.4 LLM agents and tool use (ReAct-style action loops).
2.5 Reinforcement learning for LLMs: policy gradient, PPO, GRPO; LoRA/PEFT.
2.6 RL for search agents: Search-R1 (outcome reward, retrieved-token masking).
2.7 Grounding and evidence rewards: CaRR / C-GRPO (citation rubrics), STAMP (step-level provenance credit), EVO-RAG, efficiency work (SAAS, AutoSearch, SlimSearcher).
2.8 Research gap / positioning: small-budget, deterministic, fully reproducible study combining answer + evidence + efficiency rewards.
2.9 Summary table of related work (method, reward, model size, benchmark).

Sources: `docs/research/2026-10-03-related-work-and-directions.md`.

## Chapter 3 — Methodology
3.1 Research design overview (diagram: question → agent → SEARCH/ANSWER → environment → reward → GRPO update).
3.2 Dataset and splits: HotpotQA train/dev, deterministic hash split, frozen 300-question holdout (`prepare_splits.py`).
3.3 Environment: `SearchEnvironment`, BM25 top-k=3, max 3 searches, action grammar, citation validation, failure types (invalid, exhausted).
3.4 Reward design: exact match, F1, evidence coverage, citation precision, retrieval recall, search cost, duplicate-query penalty; protocol reward −1 (formula + weights from `RewardConfig`).
3.5 Policy model: Qwen2.5-7B-Instruct, chat prompt (`build_prompt`), LoRA config (r=16, α=32, all projection layers).
3.6 Training algorithm: multi-turn GRPO — G rollouts per question, group-normalized advantage, sequence-mean log-prob over generated action tokens only, clipped objective (ε=0.2), no KL; pseudocode.
3.7 Evaluation protocol: greedy decoding, no retries, holdout only; metrics (EM, F1, evidence coverage, citation precision, retrieval recall, searches/episode, invalid rate, shortcut rate = correct answer with poor evidence).
3.8 Baselines: scripted/BM25-oracle, OpenRouter Qwen (reference only), local un-tuned Qwen (main baseline B0).
3.9 Implementation and compute: Python/uv, PyTorch, Transformers, PEFT; RunPod L40S 48 GB; reproducibility (seeds, manifests, run metadata).

## Chapter 4 — Results and Discussion
4.1 Experimental setup table (hyperparameters, data sizes, GPU hours, cost).
4.2 Baseline results (B0, oracle, OpenRouter reference).
4.3 GRPO training dynamics: reward mean, zero-variance group rate, invalid rate, searches per episode over steps (plots from `metrics.jsonl`).
4.4 Main results: B0 vs. GRPO checkpoint on the holdout (table, with bootstrap 95% CIs).
4.5 Answering RQ1–RQ3.
4.6 Optional ablation (if budget allows): reward variant, e.g. answer-only vs. full reward, or gated search cost.
4.7 Error analysis: failure categories (`analyze_eval.py`), 5–10 qualitative traces (good, shortcut, over-search, format failure).
4.8 Discussion: what worked, what did not, comparison to literature, threats to validity (single seed, small holdout, BM25 closed corpus).

## Chapter 5 — Conclusion
5.1 Summary of findings per RQ.
5.2 Contributions.
5.3 Limitations.
5.4 Future work: step-level credit (STAMP-style), correctness-gated efficiency penalty, more seeds/datasets (2Wiki, MuSiQue), larger budgets, open-web retrieval.

## References
Search-R1, CaRR, STAMP, EVO-RAG, GRPO/DeepSeekMath, PPO, LoRA, HotpotQA, BM25, ReAct, Qwen2.5 (full list in the research notes).

## Appendices
- A. Prompt template and action grammar.
- B. Full hyperparameters (`configs/training/grpo_smoke.json`, evaluation configs).
- C. Example trajectories.
- D. Reproduction commands (`docs/runbooks/runpod.md`).

---

## Writing checklist
- [ ] Ch2 can be written now (research notes ready).
- [ ] Ch3 can be written now (code is final for the first run).
- [ ] Ch4 after baseline + GRPO runs finish.
- [ ] Ch1, Ch5, abstract last.
