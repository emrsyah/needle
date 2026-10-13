# Chapter 5 — Conclusion

## 5.1 Summary

This thesis asked whether a search agent trained with reinforcement learning can be rewarded for *grounded* answers, rather than only correct ones, without an LLM judge and on a student budget.

- **Setup.** A Qwen2.5-7B-Instruct agent with LoRA was trained with a GRPO-style group-normalised policy gradient on HotpotQA. It searched a closed pool of each question's 10 paragraphs with BM25, and answered with sentence-level citations that a rule could check against HotpotQA's gold supporting facts.
- **Runs.** Sixteen runs were trained on one GPU for about $49 in total: five reward and optimizer configurations on three seeds each, plus one diagnostic run.
- **Evaluation.** Each run was evaluated on a frozen 300-question holdout, with seed-aware bootstrap confidence intervals.

**RQ1: Does GRPO training improve a 7B search agent over the un-tuned model?** Yes, mostly through protocol learning. Every method gains about +20 EM points over the un-tuned model (22.3%), and every seed-aware confidence interval excludes zero. Most of the gain comes from producing valid actions. The un-tuned model fails 130 of 300 episodes, while trained runs fail at most 22. EM on completed episodes rises only from 39.4% to 45.2% (R1).

**RQ2: Does VEGR prevent the shortcuts of answer-only reward without losing accuracy?** Yes, within this setting.

- **Accuracy.** The answer-only reward and the full VEGR bundle reach the same accuracy (43.6% vs. 42.6% EM; +1.0, seed-aware CI [−3.6, +5.4]).
- **Grounding.** VEGR's precision-grounded EM is **24.4 points higher** (CI [+8.8, +37.3]).
- **Citation loss under the answer-only reward.** Two of three answer-only seeds stopped citing. The training rollouts explain why: under the answer-only reward, an invalid citation costs −1 and a correct one earns nothing, and early in training about half of all citation attempts were invalid. The third seed kept citing but drifted into repeated searches, which that reward neither punishes nor rewards.
- **What the comparison changes.** R1 differs from R0 in six reward terms and in its number of updates, so the result concerns the VEGR bundle as a whole.
- **Correct, not complete.** VEGR makes citations correct but not complete. Its evidence coverage is no higher than the un-tuned model's, and its Joint-EM is low and varies between seeds.

**RQ3: How do reward variants and optimizer changes affect grounding and search behaviour?** These results are exploratory, and none of the variants is shown to improve on plain VEGR.

- Gated penalties (R2) and advantages without std normalisation (R4) gave inconclusive results. R2's apparent single-seed gain of +6 EM points did not replicate.
- Doubling the coverage weight (R3) raised evidence coverage from 0.44 to 0.57 and gave the best HotpotQA Joint-EM (+6.3 over R1, CI [+0.2, +13.1]). It also lowered precision-grounded EM by 8.0 points (CI [−16.1, −0.2]). Which method is better depends on the metric.
- On the one seed tested, dynamic sampling (R0-dyn) raised answer-only accuracy by 8.0 EM points, together with twice the training questions, but did not prevent the loss of citations.

## 5.2 Contributions

1. **VEGR**, a judge-free, sentence-level evidence reward that also enforces the action protocol. Its individual terms are not new; the contribution is the combination and its controlled evaluation.
2. **An explanation of how answer-only reward loses citations**, in a small, reproducible setting where the behaviour can be detected by rule. Invalid citations are penalised and correct ones are never rewarded. The thesis also shows that restoring the updates lost to zero-variance groups did not prevent the collapse on the seed tested.
3. **A controlled multi-seed comparison** with seed-aware confidence intervals. It shows how a single-seed result can mislead, and that grounding conclusions depend on the grounding metric used.
4. **A documented single-GPU pipeline.** One run costs about $2.25. Splits, configurations, training logs, per-episode results and the analysis scripts are in the repository. The LoRA adapters of 15 of the 16 runs are kept; the R1 seed-7 adapter was lost.

## 5.3 Limitations

- **Setting.** Retrieval is BM25 over each question's own 10 paragraphs, on one dataset, so the search problem is far easier than open-web search. The un-tuned model already retrieves 83% of the gold documents.
- **External validity.** VEGR needs sentence-level gold evidence, which few datasets provide, and it was tested on one dataset with one model size. The results say nothing yet about settings without such labels.
- **Scale.** Training uses 256 questions per run for one epoch, on a 7B model with LoRA, and stops before rewards converge.
- **Statistics.** Three seeds and 300 holdout questions resolve large effects, such as the grounding gap between R1 and R0. They cannot resolve EM differences of a few points, and the frequency of R0's collapse is essentially unknown (2 of 3 seeds, Clopper–Pearson 95% CI [0.09, 0.99]).
- **Design.** The R1–R0 comparison changes the whole reward bundle and the number of updates. R2–R4 and R0-dyn are exploratory and uncorrected for multiple comparisons. R0-dyn has one seed and reached 250 of its 256 target updates.
- **Metric–reward overlap.** VEGR is trained on the same gold facts that the grounding metrics use. Precision-grounded EM is lenient about missing evidence, and no gold-independent attribution check was run.
- **Algorithm.** The training update is a single on-policy step per question group, so PPO clipping never activated. The algorithm is a GRPO-style policy gradient, not full GRPO.
- **Environment.** The environment accepts uncited answers, the citation separator collides with '&' in some titles, and the reward weights were not tuned.

## 5.4 Ethics and Broader Impact

Answers with citations look trustworthy, and that appearance can lead users to over-trust them. A citation that matches a gold sentence shows that the agent pointed to relevant evidence. It does not show that the answer was actually derived from that evidence, so precision-grounded EM is not a measure of faithfulness. Even VEGR's best agents leave about a quarter of their correct answers with at least one non-gold citation, and they rarely cite the complete evidence. Systems built this way should therefore present citations as pointers for checking, not as proof.

The work uses only public data and models. HotpotQA is distributed under the CC BY-SA 4.0 license. Qwen2.5-7B-Instruct is released by the Qwen team under its own model license, which governs any use or redistribution of the trained adapters. The total compute footprint was modest: about $49 of rented time on a single L40S GPU (Table 3.2). The findings are limited to one dataset with gold evidence labels, one model and a closed retrieval pool, and should not be read as evidence about open-web search agents.

## 5.5 Future Work

- **Separate the parts of the VEGR bundle.** Train VEGR without its evidence terms (coverage and precision), keeping the answer and search terms. This is the missing ablation needed to tell whether the grounding gain comes from rewarding evidence or from a denser reward with more updates. An evidence-terms-only variant would complete the picture.
- **Close the no-citation loophole and fix the citation format.** Require at least one citation, and use a separator that cannot occur in titles (Section 4.7). Then test whether answer-only training still loses citations, or finds a different shortcut.
- **Reward complete evidence without losing precision.** Examples are a reward based on the F1 of the citation set, or a coverage term counted only for gold citations, instead of a larger weight on coverage alone (R3).
- **Check attribution without the gold labels.** Use an NLI-based citation check (Gao et al., 2023) to test whether VEGR's gains hold under a measure it was not trained on.
- **Add a prompt-only baseline.** Since most of RQ1's gain is protocol learning, compare with a few-shot prompt that shows worked examples of the action format.
- **Step-level credit.** Reward individual searches by the information they add, as in StepSearch (Wang et al., 2025) and STAMP (Xu et al., 2026), rather than scoring only the final episode.
- **Harder retrieval and more data.** Extend the setup to an open corpus, to other multi-hop datasets with gold evidence such as 2WikiMultiHopQA and MuSiQue, and to larger training sets.
- **More seeds for the diagnostic runs.** Run R0-dyn on all three seeds with the same number of training questions as R0, so that dynamic sampling can be separated from extra data.
