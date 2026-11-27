# 2026-11-19 — Thesis analysis checks (CPU only)

Four AI-assisted reviews of the Indonesian thesis (`docs/thesis/reviews/review-*.md`) raised claims that needed numbers not yet in the repo. `scripts/thesis_analysis_checks.py` computes them from existing results, with no new training or evaluation.

Command: `uv run --no-project --with numpy python scripts/thesis_analysis_checks.py results > results/2026-11-19-thesis-analysis-checks/output.md`

Raw output: [`results/2026-11-19-thesis-analysis-checks/output.md`](../results/2026-11-19-thesis-analysis-checks/output.md)

## Findings

- **Citing vs not citing during training (first 32 steps).** The share of citation attempts that fail validation is 51% for R0 s8, 23% for R0 s9 and 36% for R0 s7. The expected reward of an attempt is −0.27 (s8), +0.29 (s9) and −0.05 (s7), against 0.36, 0.26 and 0.30 for valid rollouts without a gold citation. R0 s7 did not collapse, and R0 s9 collapsed even though attempting to cite paid slightly more in its first block. This statistic therefore does not separate runs that collapsed from runs that did not. The within-group gap between valid rollouts with and without a gold citation is about 0.1–0.4 for R0 and about 2 for R1.
- **Citations to paragraphs never retrieved.** The environment only checks that a cited title is one of the question's 10 paragraphs and that the sentence index is in range. At least 4 holdout episodes cite gold sentences from paragraphs no search returned: 2 in B0, 1 in R2 s7 and 1 in R4 s7. This is a lower bound.
- **Cited but precision 0.** Completed holdout episodes that cite only non-gold sentences: 35 (B0), 62 (R0 s7), 29–70 for R1–R4 runs.
- **R0-dyn vs R0 s8.** Same 256 questions in the same order (256/256). Rollout rewards differ from step 2. Skipped steps: 29 vs 19 in steps 1–96, 128 vs 107 in steps 1–256. R0-dyn ran 512 steps with 250 updates.
- **Gradient clipping.** The pre-clip gradient norm exceeds the 1.0 threshold on 86–100% of update steps for R0–R3 and on 59–72% for R4.
- **Clopper–Pearson.** R0 collapse 2/3, [0.09, 0.99]. R1 collapse 0/3, [0.00, 0.71].
- **Seed pairs, precision-grounded EM.** All 9 R1 × R0 seed pairs are positive; the smallest gap is +3.3 (R1 s8 vs R0 s7). Mean R1 is 32.7 against 24.7 for R0 s7, the only R0 seed that kept citing.
- **Questions every run completed (136).** EM: B0 37.5, method means 43.4–45.6 (R1 45.6, +8.1). Coverage: B0 0.463, R1 0.460.
- **Single-run CIs (question-only, numpy seed 0, same method as `make_thesis_figures.py`).** R2 s9 − B0 +16.0 [+10.7, +21.3]; R0-dyn − B0 +26.3 [+21.0, +31.7]; R2 s8 − R1 s8 +6.0 [+1.3, +11.0]; R0-dyn − R0 s8 +8.0 [+3.3, +12.7]. These match the thesis text. The [+1.3, +10.7] in the 2026-10-10 batch log came from an earlier bootstrap with a different generator.
- **Citations per cited episode.** R1 s7 1.49, s8 1.69, s9 1.03; R4 s9 1.04; R3 1.83–2.06.
