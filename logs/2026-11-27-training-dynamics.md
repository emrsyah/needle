# 2026-11-27 Training dynamics for every run (thesis revision)

Requested in the 2026-11-27 supervisor meeting: the per-step ("per epoch") training diagnostic was shown only for R0 s8 and R1 s8. It now covers all 16 trained runs.

- Script: `scripts/make_training_dynamics.py` (no GPU; reads each run's `metrics.jsonl`).
- Command: `uv run --no-project --with matplotlib python scripts/make_training_dynamics.py`
- Outputs:
  - `docs/thesis/figures/fig4_4_dynamics_r0_r1.{png,pdf}`: R0 and R1, seeds 7–9 (thesis Gambar 4.4).
  - `docs/thesis/figures/figB_1_dynamics_variants.{png,pdf}`: R2, R3, R4 and R0-dyn (Lampiran B).
  - `results/2026-11-27-training-dynamics/training_dynamics.{json,csv}` and `tables.md`.
- Per 32-step block: mean citation precision of valid rollouts, share of valid rollouts with a gold citation, share of rollouts scored −1 (status `invalid` or `exhausted`), skipped (zero-variance) steps, and mean training reward.
- Regression checks in the script match numbers already in BAB 4: R0 s8 precision 0.568 / 0.499 / 0.126; R0 s8 skipped 128; R0-dyn skipped 262; R1 s7 invalid 46 / 29 / 13 / 13 / 7 %.

## What it shows

- R0 s8, R0 s9 and R0-dyn lose citations in block 3 (steps 65–96) and stay at 0 afterwards, including R0-dyn's extra 256 steps.
- R0 s7 keeps citing (precision 0.55–0.73 across blocks).
- All VEGR-family runs (R1–R4, 12 runs) keep precision between 0.52 and 0.85 in every block. None loses citations.
- Invalid rollouts fall from 24–52% in block 1 to 0–15% in block 8 for every run. R0 s9 rises to 15% in block 8: 38 of 256 rollouts exhaust the search budget without answering.
- Skipped steps per block: R0 runs 4–24, VEGR-family runs 0–7.
