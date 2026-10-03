# Needle — Related Work and Research Directions

Date: 2026-10-03
Sources: alphaXiv and Consensus searches. Summaries are from titles/abstracts only; read the full papers before citing claims.

## Where Needle sits

- **Base recipe** — outcome reward + GRPO + Qwen2.5‑7B + retrieved-token masking comes from Search‑R1 [1]. Needle's trainer plan (loss only on generated action tokens) matches this.
- **Evidence/citation rewards are no longer novel on their own:**
  - CaRR / C‑GRPO [2] — decomposes multi-hop questions into single-hop rubrics requiring correct citations, combined with outcome reward. Closest prior work; Needle must differentiate from it.
  - STAMP [4] — notes citation-aware and evidence-coverage rewards are now common; moves to provenance-guided credit for the search actions that surfaced cited evidence.
  - EVO‑RAG [3] — HotpotQA, step-level rewards (relevance, redundancy, efficiency), ~15% shallower retrieval. Competes with the "avoid unnecessary retrieval" claim.

## Proposed directions (ranked)

### 1. Turn-level evidence credit assignment (strongest angle)
Vanilla GRPO gives every turn the same advantage. Related: TRACE [5], BiCAA [6], ABSeeker [7], fact-utility process supervision [8].

Needle's edge: deterministic BM25 + HotpotQA gold supporting facts give exact per-search attribution of which query retrieved which gold paragraph. `information_gain` is already computed per search but diagnostic-only — promote it to a per-turn advantage term.

Ablation: outcome-only vs. trajectory reward vs. trajectory + turn credit.

### 2. Multi-reward normalization
The scalar sums 5 quality terms and 2 penalties at weight 1.0; EM and F1 are highly correlated and over-weight answer correctness. See CorrGRPO [9]. Minimum: per-component group normalization before summing, plus leave-one-out reward ablations.

### 3. Gated efficiency / learned stopping
The flat 0.1 per-search cost is crude and risks teaching the model to answer without searching.
- SlimSearcher [10] — apply the efficiency penalty only when the answer is correct. Near one-line change in `RewardConfig`.
- SAAS [11] (over-search), AutoSearch [12] (adaptive depth), MetaRAG [13] (confidence/stop alignment).

### 4. Shortcut-robustness metric
CaRR [2] reports outcome-only rewards cause shortcut exploitation. Measure: correct answer with wrong or missing evidence citations in the HotpotQA distractor setting. Candidate headline metric.

### 5. Practical items for the first run
- Log the fraction of zero-variance groups (all 4 rollouts equal). If >~50%, move to G=8 or filter for medium-difficulty questions.
- Log first-query quality; the first search move matters [14].
- Add 2WikiMultiHopQA or MuSiQue as eval-only sets to address generalization.

## Recommendation
Run the planned GRPO smoke test unchanged. Then frame the paper around **#1 (turn-level evidence credit) + #3 (gated efficiency)**, both exploiting Needle's deterministic retrieval and gold supporting facts.

Follow-ups: read CaRR [2] and STAMP [4] in full to sharpen the novelty claim; prototype the reward-gating change.

## Full-text review: CaRR and STAMP (2026-10-03)

### CaRR / C-GRPO [2] (Tsinghua/Zhipu, Jan 2026)
- LLM decomposes each synthetic multi-hop question into single-hop rubrics with hidden entities. A judge LLM checks: entity named in answer → rubric supported by cited content → rubric connected to the answer entity (BFS). Rubric reward = fraction satisfied.
- C-GRPO: `R = (1-α)·R_outcome + α·R_outcome·R̂_rubric` — rubric reward **only counts when the answer is correct** (outcome-gated). Format/overlength errors get 0.
- Qwen3-4B and 30B-A3B, live web (Serper/Jina), DeepDive data, BrowseComp-style benchmarks. Trajectory-level only; no per-step credit.
- **Key finding for Needle:** under outcome-only GRPO, average tool calls keep *decreasing* during training and the agent learns shortcut answers from the last few hops. C-GRPO keeps tool calls higher. **Fewer searches is not automatically good.**

### STAMP [4] (Baidu/PKU, Jul 2026)
- Reference-based LLM verifier checks cited documents against a training-time evidence graph (entities + relations). Each supporting document is credited to the **step that first exposed it** (first-exposure attribution). Per-step credit capped at C<1.
- Credit enters via **sign-preserving advantage modulation**: `A_step = A·(1 + sign(A)·credit_t)`. Positive trajectories amplify evidence steps; negative trajectories attenuate penalties on evidence steps. Reward and group ranking unchanged.
- +2.0/+5.5/+3.0 on BrowseComp/-ZH/xbench over GRPO; composes with C-GRPO. Ablations: first-exposure > all-step, capped > uncapped, both signs > positive only. String matching verifier much worse than the LLM judge.
- Explicitly leaves "information-gain estimation" attribution to future work. No efficiency objective. Only one model scale (30B-A3B), needs SFT cold start, live web.

### Impact on Needle's thesis
- **Direction #1 (turn-level evidence credit) is essentially STAMP.** Needle cannot claim it as a novel method. It can still be used as a component/baseline, implemented as STAMP's modulation with exact gold-paragraph attribution.
- **Direction #3 gating is partly CaRR's outcome-gating idea** applied to penalties instead of bonuses.
- **What remains open:**
  1. Neither paper studies the **efficiency ↔ grounding trade-off**. CaRR shows that outcome RL already shrinks search and produces shortcuts; adding a search-cost penalty (Needle's design) likely makes this worse. Needle can measure this exactly, because HotpotQA gold supporting facts make shortcut detection deterministic (no LLM judge).
  2. **Noise-free attribution**: STAMP's credit depends on an LLM verifier (string matching collapses gains). Needle has exact gold-paragraph labels, so it can isolate "does step credit help when attribution is perfect?" and test sensitivity to injected attribution noise. That is a clean, cheap controlled study.
  3. **Small-model, no-SFT, low-budget regime** (7B LoRA, ~$30), where neither paper reports results.

### Revised thesis (proposed)
> Efficiency penalties in search-agent RL trade grounding for cost: they accelerate the shortcut collapse that outcome-only GRPO already exhibits. With exact evidence attribution, provenance-style step credit (STAMP) recovers grounding while keeping the efficiency gains. Needle measures this trade-off precisely on HotpotQA, where shortcuts and evidence are detected deterministically.

Main comparison becomes: outcome-only, + search penalty (ungated), + gated penalty, + STAMP-style credit with gold attribution, with attribution-noise sweep as the analysis section.

## References

1. Jin et al., 2025. *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.* https://consensus.app/papers/details/58192257da745001bdf401595ac7eaf2/
2. Zhang et al., 2026. *Chaining the Evidence: Robust RL for Deep Search Agents with Citation-Aware Rubric Rewards (CaRR).* https://www.alphaxiv.org/abs/2601.06021
3. Ji et al., 2025. *Curriculum Guided RL for Efficient Multi Hop RAG (EVO-RAG).* https://consensus.app/papers/details/206bc46a08df5c779992d9b5d59f0cf7/
4. *STAMP: Provenance-Guided Credit Assignment for Deep Search Agents.* https://www.alphaxiv.org/abs/2607.11172
5. *TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents.* https://www.alphaxiv.org/abs/2607.13988
6. *BiCAA: Bidirectional Credit Assignment for Search-Augmented Agent.* https://www.alphaxiv.org/abs/2608.01321
7. *ABSeeker: Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment.* https://www.alphaxiv.org/abs/2608.05102
8. *Dense Process Supervision for Search Agents via Fact Utility Estimation.* https://www.alphaxiv.org/abs/2609.00833
9. *CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning.* https://www.alphaxiv.org/abs/2609.36820
10. *SlimSearcher: Training Efficiency-Aware Web Agents via Adaptive Reward Gating.* https://www.alphaxiv.org/abs/2606.07074
11. *SAAS: Self-Aware RL for Over-Search Mitigation in Agentic Search.* https://www.alphaxiv.org/abs/2605.29796
12. *AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via RL.* https://www.alphaxiv.org/abs/2604.17337
13. *MetaRAG: Belief-Action Aligned Policy Optimization for Agentic RAG.* https://www.alphaxiv.org/abs/2608.24214
14. *Question's Gambit: The First Move Matters in Agentic Deep Search.* https://www.alphaxiv.org/abs/2609.14412

Also relevant: GRASP (https://www.alphaxiv.org/abs/2607.10463), GTA-RAG (https://www.alphaxiv.org/abs/2608.22479), Contribution-Weighted GRPO (https://www.alphaxiv.org/abs/2604.14267), Diagnosing Search Behavior (https://www.alphaxiv.org/abs/2608.01913).
