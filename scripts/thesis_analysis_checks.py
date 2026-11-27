"""CPU checks for the thesis review round of 2026-11-19. Reads existing results only.

Usage: python scripts/thesis_analysis_checks.py results > results/2026-11-19-thesis-analysis-checks/output.md
"""
import json, statistics, sys
from collections import Counter
from math import lgamma, exp
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])
F = R / "2026-10-11-fixes"
NAMES = {"R0": "r0_outcome", "R1": "r1_full", "R2": "r2_gated", "R3": "r3_coverage2", "R4": "r4_drgrpo"}


def ep_path(name, s):
    if (name, s) == ("r1_full", 7):
        return R / "2026-10-10-grpo-run2/grpo_run2_episodes.jsonl"
    return F / "results" / f"{name}_seed{s}" / "episodes.jsonl"


runs = {"B0": R / "2026-10-10-grpo-smoke/b0_episodes.jsonl"}
for t, n in NAMES.items():
    for s in (7, 8, 9):
        runs[f"{t} s{s}"] = ep_path(n, s)
runs["R0-dyn s8"] = F / "results/r0_dynsample_seed8/episodes.jsonl"
load = lambda p: {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
D = {k: load(p) for k, p in runs.items()}
ids = sorted(D["B0"])


def m(r, k):
    return (r.get("metrics") or {}).get(k) if r["status"] == "completed" else None


def em(r): return 1.0 if m(r, "answer_exact_match") == 1 else 0.0
def cited(r): return r["status"] == "completed" and bool(r.get("citations"))
def gem(r): return 1.0 if em(r) and cited(r) and m(r, "citation_precision") == 1 else 0.0


def steps_of(run):
    return [json.loads(l) for l in open(F / "runs" / run / "metrics.jsonl", encoding="utf-8")]


# --- 1. Citing vs not citing during training, first two 32-step blocks ------------------------
print("## Training rollouts: attempts to cite, first two 32-step blocks\n")
print("'Attempt to cite' = evidence failure + valid rollout with at least one gold citation. "
      "Rollout logs store no citation list, so 'valid, no gold citation' also contains rollouts "
      "that cited only non-gold sentences.\n")
print("| Run | Steps | attempts | evidence failures | share invalid | E[reward \\| attempt] | mean reward: valid, ≥1 gold citation | mean reward: valid, no gold citation |")
print("|---|---|---|---|---|---|---|---|")
for run in ("r0_outcome_seed7", "r0_outcome_seed8", "r0_outcome_seed9", "r1_full_seed8"):
    steps = steps_of(run)
    for b in (0, 32):
        ro = [x for s in steps[b:b + 32] for x in s.get("rollouts", [])]
        ev = [x for x in ro if x["status"] != "valid" and x.get("failure_kind") == "evidence"]
        val = [x for x in ro if x["status"] == "valid"]
        vc = [x for x in val if (x.get("citation_precision") or 0) > 0]
        vn = [x for x in val if (x.get("citation_precision") or 0) == 0]
        att = len(vc) + len(ev)
        e_att = (sum(x["reward"] for x in vc) - len(ev)) / att if att else float("nan")
        mr = lambda xs: f"{statistics.mean(x['reward'] for x in xs):.2f}" if xs else "–"
        print(f"| {run} | {b+1}–{b+32} | {att} | {len(ev)} | {100*len(ev)/att:.0f}% | {e_att:+.2f} | {mr(vc)} | {mr(vn)} |")

# --- 2. Citations to paragraphs that were never retrieved (lower bound) ------------------------
print("\n## Holdout episodes whose coverage exceeds retrieval recall\n")
print("Coverage > retrieval recall is only possible when a gold sentence is cited from a paragraph "
      "that no search returned. Citations to non-gold sentences of unretrieved paragraphs are not "
      "detectable this way, so the counts are a lower bound.\n")
print("| Run | completed | coverage > retrieval recall |")
print("|---|---|---|")
total = 0
for k, d in D.items():
    comp = [d[i] for i in ids if d[i]["status"] == "completed"]
    n = sum(1 for r in comp if (m(r, "evidence_coverage") or 0) > (m(r, "retrieval_recall") or 0) + 1e-9)
    total += n
    print(f"| {k} | {len(comp)} | {n} |")
print(f"\nTotal over all runs: {total}")

# --- 3. Completed episodes that cite but have precision 0 --------------------------------------
print("\n## Completed holdout episodes that cite only non-gold sentences (precision 0)\n")
print("| Run | completed | cited, precision 0 |")
print("|---|---|---|")
for k, d in D.items():
    comp = [d[i] for i in ids if d[i]["status"] == "completed"]
    print(f"| {k} | {len(comp)} | {sum(1 for r in comp if cited(r) and (m(r, 'citation_precision') or 0) == 0)} |")

# --- 4. R0 s8 vs R0-dyn s8 -------------------------------------------------------------------
a, b = steps_of("r0_outcome_seed8"), steps_of("r0_dynsample_seed8")
same_order = sum(1 for x, y in zip(a[:256], b[:256]) if x["question_id"] == y["question_id"])
first_diff = next((i + 1 for i, (x, y) in enumerate(zip(a, b))
                   if [r["reward"] for r in x.get("rollouts", [])] != [r["reward"] for r in y.get("rollouts", [])]), None)
skip = lambda st, n: sum(1 for s in st[:n] if s.get("skipped_reason"))
print("\n## R0 s8 vs R0-dyn s8\n")
print(f"- Same question in the same position, first 256 steps: {same_order}/256")
print(f"- First step whose rollout rewards differ: {first_diff}")
print(f"- Skipped steps 1–96: R0 s8 {skip(a, 96)}, R0-dyn {skip(b, 96)}")
print(f"- Skipped steps 1–256: R0 s8 {skip(a, 256)}, R0-dyn {skip(b, 256)}")
print(f"- R0-dyn total steps {len(b)}, updates {len(b) - skip(b, len(b))}")

# --- 5. Gradient clipping ---------------------------------------------------------------------
print("\n## Share of update steps with pre-clip gradient norm > 1.0 (clip threshold)\n")
print("| Run | update steps | grad norm > 1 | median grad norm |")
print("|---|---|---|---|")
for run in sorted(p.name for p in (F / "runs").iterdir()):
    g = [s["grad_norm"] for s in steps_of(run) if not s.get("skipped_reason") and s.get("grad_norm") is not None]
    if g:
        print(f"| {run} | {len(g)} | {100*sum(x > 1 for x in g)/len(g):.0f}% | {statistics.median(g):.2f} |")

# --- 6. Clopper–Pearson -------------------------------------------------------------------------
def beta_ppf(q, a, b, lo=0.0, hi=1.0):
    def inc(x):
        n = 4000; s = 0.0
        for i in range(n):
            t = (i + 0.5) / n * x
            s += t ** (a - 1) * (1 - t) ** (b - 1)
        return s * x / n / exp(lgamma(a) + lgamma(b) - lgamma(a + b))
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if inc(mid) < q else (lo, mid)
    return (lo + hi) / 2


print("\n## Clopper–Pearson 95% CI for citation collapse\n")
print(f"- R0: 2/3 seeds, [{beta_ppf(0.025, 2, 2):.2f}, {beta_ppf(0.975, 3, 1):.2f}]")
print(f"- R1: 0/3 seeds, [0.00, {1 - 0.025 ** (1 / 3):.2f}]")

# --- 7. Seed pairs, precision-grounded EM -----------------------------------------------------
pg = {k: 100 * statistics.mean(gem(d[i]) for i in ids) for k, d in D.items()}
pairs = [(s1, s0, pg[f"R1 s{s1}"] - pg[f"R0 s{s0}"]) for s1 in (7, 8, 9) for s0 in (7, 8, 9)]
print("\n## Precision-grounded EM, all R1 × R0 seed pairs\n")
print(f"- Positive pairs: {sum(d > 0 for *_, d in pairs)}/9; smallest gap {min(d for *_, d in pairs):+.1f} "
      f"(R1 s{min(pairs, key=lambda p: p[2])[0]} vs R0 s{min(pairs, key=lambda p: p[2])[1]})")
print(f"- R1 vs R0 s7 only (the R0 seed that kept citing): mean R1 {statistics.mean(pg[f'R1 s{s}'] for s in (7,8,9)):.1f} "
      f"vs {pg['R0 s7']:.1f}")

# --- 8. EM on questions every run completed ---------------------------------------------------
common = [i for i in ids if all(d[i]["status"] == "completed" for d in D.values())]
cem = {k: 100 * statistics.mean(em(d[i]) for i in common) for k, d in D.items()}
cov = {k: statistics.mean(m(d[i], "evidence_coverage") or 0 for i in common) for k, d in D.items()}
print(f"\n## Questions completed by every run: {len(common)}\n")
print("| Method | EM (mean over seeds) | Δ vs B0 | coverage (mean over seeds) |")
print("|---|---|---|---|")
print(f"| B0 | {cem['B0']:.1f} | – | {cov['B0']:.3f} |")
for t in NAMES:
    e = statistics.mean(cem[f"{t} s{s}"] for s in (7, 8, 9))
    c = statistics.mean(cov[f"{t} s{s}"] for s in (7, 8, 9))
    print(f"| {t} | {e:.1f} | {e - cem['B0']:+.1f} | {c:.3f} |")
print(f"| R0-dyn s8 | {cem['R0-dyn s8']:.1f} | {cem['R0-dyn s8'] - cem['B0']:+.1f} | {cov['R0-dyn s8']:.3f} |")

# --- 9. Single-run EM vs B0, question bootstrap ---------------------------------------------
def paired_bootstrap(x, y, resamples=10_000, seed=0):
    diff = np.array([em(x[i]) - em(y[i]) for i in ids])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(ids), size=(resamples, len(ids)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return 100 * diff.mean(), 100 * lo, 100 * hi


print("\n## Single-run ΔEM vs B0 (question-only paired bootstrap, 10k, numpy seed 0)\n")
for k in ("R2 s8", "R2 s9", "R0-dyn s8"):
    p, lo, hi = paired_bootstrap(D[k], D["B0"])
    print(f"- {k} − B0: {p:+.1f} [{lo:+.1f}, {hi:+.1f}]")
p, lo, hi = paired_bootstrap(D["R2 s8"], D["R1 s8"])
print(f"- R2 s8 − R1 s8: {p:+.1f} [{lo:+.1f}, {hi:+.1f}]")
p, lo, hi = paired_bootstrap(D["R0-dyn s8"], D["R0 s8"])
print(f"- R0-dyn s8 − R0 s8: {p:+.1f} [{lo:+.1f}, {hi:+.1f}]")

# --- 10. Citations per completed episode --------------------------------------------------------
print("\n## Mean citations per cited, completed episode\n")
for k in ("R1 s7", "R1 s8", "R1 s9", "R4 s9", "R3 s7", "R3 s8", "R3 s9"):
    c = [len(D[k][i]["citations"]) for i in ids if cited(D[k][i])]
    print(f"- {k}: {statistics.mean(c):.2f}")
