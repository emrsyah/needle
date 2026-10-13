# Chapter 1 — Introduction

## 1.1 Background (Latar Belakang)

Large language models (LLMs) can answer a wide range of questions, but their knowledge is fixed when training ends, and they sometimes state false information with confidence. A common remedy is **retrieval-augmented generation** (RAG): before answering, the system retrieves relevant documents and conditions its answer on them. Many questions cannot be answered with a single retrieval. A *multi-hop* question such as "Which company formed the boy group that debuted with the album *2014 S/S*?" requires finding one fact (the group), then using it to find another (the company).

This has motivated **search agents**: LLMs that decide for themselves when to search, what to search for, and when to stop and answer (Yao et al., 2023). Prompting alone does not reliably produce good search behaviour, so recent work trains search agents with **reinforcement learning** (RL). In particular, **Search-R1** (Jin et al., 2025) trains a model with an outcome reward, meaning the model is rewarded only when its final answer is correct. It uses **GRPO** (Shao et al., 2024), an RL algorithm that compares several attempts at the same question instead of training a separate value model. GRPO with rule-checked rewards has also been central to recent reasoning models (DeepSeek-AI, 2025).

Rewarding only the final answer creates a problem. A correct answer does not show that the agent found, used, or could point to the evidence behind it. In RL terms, the answer reward is a **proxy** for what is actually wanted: a correct answer *supported by evidence*. Optimizing a proxy can produce behaviour that scores well on the proxy but not on the real goal (Skalse et al., 2022).

HotpotQA (Yang et al., 2018) offers a way to study this cheaply. Its questions are annotated with the exact sentences that support the answer. If an agent cites sentences, the citations can be checked by a simple rule against these gold labels, with no judge.

## 1.2 Problem Statement (Rumusan Masalah)

This thesis addresses four related gaps:

- **G1, correct is not grounded.** An agent trained with answer-only reward can learn to answer without showing evidence, as long as the final answer is right. Recent work reports such shortcut behaviour in outcome-trained search agents (Zhang et al., 2026).
- **G2, judge-free evidence rewards are rarely tested against an outcome-only control.** Several evidence rewards use an LLM as a judge (Zhang et al., 2026; Xu et al., 2026), which adds cost and possible bias. Rule-based alternatives exist (Chapter 2), but they are rarely compared over several seeds with an outcome-only reward in an otherwise identical setup.
- **G3, a compute barrier.** RL for search agents is reported at multi-GPU scale: Search-R1 trains on 8 H100 GPUs and AutoRefine on 8 A100 GPUs (Chapter 2). It is unclear whether the same effects can be studied on one GPU.
- **G4, weak evidence about what matters.** RL results vary considerably between random seeds (Henderson et al., 2018). Search-R1 and CaRR train each configuration once (Chapter 2), and single runs cannot separate the effect of a method from seed noise (Agarwal et al., 2021).

These gaps lead to three research questions:

- **RQ1.** Does GRPO training improve a 7B search agent on HotpotQA over the un-tuned model?
- **RQ2.** Does a verifiable evidence-grounded reward (VEGR) prevent the shortcut behaviour of answer-only reward, without losing accuracy?
- **RQ3.** How do reward variants (gated penalties, a larger coverage weight) and optimizer changes (advantages without std normalisation, dynamic sampling) affect grounding and search behaviour?

RQ1 and RQ2 are the primary questions. RQ3 is exploratory: its variants were partly designed after the first results were seen (Section 3.8).

## 1.3 Objectives (Tujuan)

Each objective answers one research question:

1. **For RQ1:** train a search agent with GRPO-style RL and LoRA on a single GPU, and measure its accuracy and protocol compliance against the un-tuned model on a frozen holdout.
2. **For RQ2:** design a judge-free reward, VEGR, that rewards correct answers, correct citations and useful searches, and compare it with an answer-only reward over three seeds, on both accuracy and grounding, with seed-aware bootstrap confidence intervals.
3. **For RQ3:** test whether reward variants and optimizer changes alter grounding, coverage and training stability.

## 1.4 Scope and Limitations (Batasan Masalah)

This thesis is deliberately narrow:

- **Data and retrieval.** It uses HotpotQA in the distractor setting. The agent searches **a closed pool of each question's 10 paragraphs** with BM25, and this pool always contains both gold paragraphs. The un-tuned model already retrieves 83% of the gold documents (retrieval recall 0.83), so this is much easier than open-web or open-corpus search, and "search agent" in this thesis means an agent searching this closed pool.
- **Model.** The model is Qwen2.5-7B-Instruct adapted with LoRA.
- **Scale.** Each training run uses 256 questions for one epoch, and evaluation uses a frozen 300-question holdout. The project was budgeted at about $45; the actual compute spend was about $49 (Table 3.2).
- **Labels.** VEGR needs sentence-level gold evidence, which few datasets provide. The method cannot be applied as-is where such labels do not exist.
- **What is not claimed.** The thesis does not propose a new RL algorithm or a new kind of reward, does not claim state-of-the-art accuracy, and does not claim that the findings transfer to open-web search, other datasets or larger models.
- **Known loophole.** The environment accepts an answer with no citations. This was kept on purpose so that all runs are comparable, and its consequences are reported.

Chapter 5 discusses the further limitations.

## 1.5 Benefits (Manfaat)

**Contributions.**

1. **VEGR**, a judge-free, sentence-level reward for training search agents. Every term is checked by rule against gold supporting facts. VEGR also enforces the action protocol: invented citations, malformed actions, and running out of budget all receive a −1 penalty. The individual reward terms are not new; the contribution is the combination and the controlled evaluation of it.
2. **Evidence of when answer-only reward loses citations, and why.** Two of three outcome-only seeds stopped citing entirely (Clopper–Pearson 95% CI for the rate [0.09, 0.99]). Training logs show why: under the answer-only reward, invalid citations cost −1 and correct ones earn nothing. The full VEGR bundle reached the same accuracy (+1.0 EM points, seed-aware 95% CI [−3.6, +5.4]) but **24.4 points higher precision-grounded EM** (CI [+8.8, +37.3]).
3. **A controlled multi-seed comparison** with seed-aware bootstrap confidence intervals. Exploratory results show that:
   - a single-seed improvement of +6 EM points (gated penalties, R2) did not replicate across seeds;
   - VEGR makes citations correct but not complete, and a larger coverage weight (R3) trades precision for completeness, ranking best on HotpotQA's Joint-EM;
   - on the one seed tested, dynamic sampling raised outcome-only accuracy (together with twice the training questions) but did not prevent the loss of citations.
4. **A documented single-GPU pipeline.** One training run plus evaluation costs about $2.25 on one 48 GB GPU. Splits, configs, training logs and per-episode results are in the repository; LoRA adapters are kept for 15 of the 16 runs.

**Benefits.** For researchers, the thesis shows how an evidence reward can be evaluated cheaply and with honest uncertainty, and it gives a concrete account of how answer-only rewards can lose citations. For practitioners building retrieval-augmented systems, it shows that answer accuracy alone does not reveal whether answers are grounded, and that grounding has to be rewarded and measured explicitly.

## 1.6 Thesis Structure

- **Chapter 2** reviews the background: LLMs, retrieval, multi-hop QA and citation evaluation, LLM agents, RL for LLMs (PPO, GRPO and its variants, LoRA), RL-trained search agents, evidence rewards, and reward hacking and RL evaluation. It ends with the research gaps.
- **Chapter 3** describes the methodology: data splits, the search environment, the VEGR reward and its variants, the policy model, the training algorithm, the evaluation protocol, and the experiment matrix.
- **Chapter 4** reports the results by research question, diagnoses the training dynamics, presents example traces, and discusses threats to validity.
- **Chapter 5** concludes, answers the research questions, and outlines ethics, limitations and future work.
