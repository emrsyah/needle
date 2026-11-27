## Training rollouts: attempts to cite, first two 32-step blocks

'Attempt to cite' = evidence failure + valid rollout with at least one gold citation. Rollout logs store no citation list, so 'valid, no gold citation' also contains rollouts that cited only non-gold sentences.

| Run | Steps | attempts | evidence failures | share invalid | E[reward \| attempt] | mean reward: valid, ≥1 gold citation | mean reward: valid, no gold citation |
|---|---|---|---|---|---|---|---|
| r0_outcome_seed7 | 1–32 | 174 | 62 | 36% | -0.05 | 0.47 | 0.30 |
| r0_outcome_seed7 | 33–64 | 205 | 37 | 18% | +0.31 | 0.60 | 0.45 |
| r0_outcome_seed8 | 1–32 | 176 | 90 | 51% | -0.27 | 0.50 | 0.36 |
| r0_outcome_seed8 | 33–64 | 173 | 79 | 46% | -0.20 | 0.48 | 0.20 |
| r0_outcome_seed9 | 1–32 | 209 | 48 | 23% | +0.29 | 0.67 | 0.26 |
| r0_outcome_seed9 | 33–64 | 138 | 45 | 33% | +0.07 | 0.59 | 0.24 |
| r1_full_seed8 | 1–32 | 176 | 91 | 52% | +1.10 | 3.35 | 1.37 |
| r1_full_seed8 | 33–64 | 169 | 59 | 35% | +1.69 | 3.13 | 1.09 |

## Holdout episodes whose coverage exceeds retrieval recall

Coverage > retrieval recall is only possible when a gold sentence is cited from a paragraph that no search returned. Citations to non-gold sentences of unretrieved paragraphs are not detectable this way, so the counts are a lower bound.

| Run | completed | coverage > retrieval recall |
|---|---|---|
| B0 | 170 | 2 |
| R0 s7 | 278 | 0 |
| R0 s8 | 299 | 0 |
| R0 s9 | 298 | 0 |
| R1 s7 | 287 | 0 |
| R1 s8 | 287 | 0 |
| R1 s9 | 293 | 0 |
| R2 s7 | 283 | 1 |
| R2 s8 | 290 | 0 |
| R2 s9 | 286 | 0 |
| R3 s7 | 289 | 0 |
| R3 s8 | 284 | 0 |
| R3 s9 | 282 | 0 |
| R4 s7 | 288 | 1 |
| R4 s8 | 293 | 0 |
| R4 s9 | 292 | 0 |
| R0-dyn s8 | 300 | 0 |

Total over all runs: 4

## Completed holdout episodes that cite only non-gold sentences (precision 0)

| Run | completed | cited, precision 0 |
|---|---|---|
| B0 | 170 | 35 |
| R0 s7 | 278 | 62 |
| R0 s8 | 299 | 0 |
| R0 s9 | 298 | 0 |
| R1 s7 | 287 | 43 |
| R1 s8 | 287 | 35 |
| R1 s9 | 293 | 70 |
| R2 s7 | 283 | 35 |
| R2 s8 | 290 | 48 |
| R2 s9 | 286 | 41 |
| R3 s7 | 289 | 29 |
| R3 s8 | 284 | 30 |
| R3 s9 | 282 | 32 |
| R4 s7 | 288 | 51 |
| R4 s8 | 293 | 47 |
| R4 s9 | 292 | 70 |
| R0-dyn s8 | 300 | 0 |

## R0 s8 vs R0-dyn s8

- Same question in the same position, first 256 steps: 256/256
- First step whose rollout rewards differ: 2
- Skipped steps 1–96: R0 s8 29, R0-dyn 19
- Skipped steps 1–256: R0 s8 128, R0-dyn 107
- R0-dyn total steps 512, updates 250

## Share of update steps with pre-clip gradient norm > 1.0 (clip threshold)

| Run | update steps | grad norm > 1 | median grad norm |
|---|---|---|---|
| r0_dynsample_seed8 | 250 | 100% | 2.65 |
| r0_outcome_seed7 | 149 | 99% | 2.28 |
| r0_outcome_seed8 | 128 | 98% | 2.64 |
| r0_outcome_seed9 | 132 | 98% | 2.67 |
| r1_full_seed8 | 248 | 98% | 2.07 |
| r1_full_seed9 | 238 | 86% | 1.93 |
| r2_gated_seed7 | 237 | 97% | 2.14 |
| r2_gated_seed8 | 236 | 96% | 1.99 |
| r2_gated_seed9 | 246 | 95% | 1.81 |
| r3_coverage2_seed7 | 245 | 98% | 1.94 |
| r3_coverage2_seed8 | 241 | 97% | 1.99 |
| r3_coverage2_seed9 | 246 | 90% | 1.60 |
| r4_drgrpo_seed7 | 248 | 61% | 1.42 |
| r4_drgrpo_seed8 | 246 | 72% | 1.82 |
| r4_drgrpo_seed9 | 242 | 59% | 1.27 |

## Clopper–Pearson 95% CI for citation collapse

- R0: 2/3 seeds, [0.09, 0.99]
- R1: 0/3 seeds, [0.00, 0.71]

## Precision-grounded EM, all R1 × R0 seed pairs

- Positive pairs: 9/9; smallest gap +3.3 (R1 s8 vs R0 s7)
- R1 vs R0 s7 only (the R0 seed that kept citing): mean R1 32.7 vs 24.7

## Questions completed by every run: 136

| Method | EM (mean over seeds) | Δ vs B0 | coverage (mean over seeds) |
|---|---|---|---|
| B0 | 37.5 | – | 0.463 |
| R0 | 43.4 | +5.9 | 0.152 |
| R1 | 45.6 | +8.1 | 0.460 |
| R2 | 44.1 | +6.6 | 0.490 |
| R3 | 43.4 | +5.9 | 0.589 |
| R4 | 45.1 | +7.6 | 0.446 |
| R0-dyn s8 | 46.3 | +8.8 | 0.000 |

## Single-run ΔEM vs B0 (question-only paired bootstrap, 10k, numpy seed 0)

- R2 s8 − B0: +26.3 [+21.0, +31.7]
- R2 s9 − B0: +16.0 [+10.7, +21.3]
- R0-dyn s8 − B0: +26.3 [+21.0, +31.7]
- R2 s8 − R1 s8: +6.0 [+1.3, +11.0]
- R0-dyn s8 − R0 s8: +8.0 [+3.3, +12.7]

## Mean citations per cited, completed episode

- R1 s7: 1.49
- R1 s8: 1.69
- R1 s9: 1.03
- R4 s9: 1.04
- R3 s7: 2.06
- R3 s8: 2.05
- R3 s9: 1.83
