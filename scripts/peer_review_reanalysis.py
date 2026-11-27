"""CPU re-analysis for peer-review issues #2, #4, #6, #9, #16, #24. Reads existing results only."""

import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

R = Path(sys.argv[1])
F = R / "2026-10-11-fixes"


def ep(name, s):
    if (name, s) == ("r1_full", 7):
        return R / "2026-10-10-grpo-run2/grpo_run2_episodes.jsonl"
    return F / "results" / f"{name}_seed{s}" / "episodes.jsonl"


NAMES = {
    "R0": "r0_outcome",
    "R1": "r1_full",
    "R2": "r2_gated",
    "R3": "r3_coverage2",
    "R4": "r4_drgrpo",
}
runs = {"B0": R / "2026-10-10-grpo-smoke/b0_episodes.jsonl"}
for t, n in NAMES.items():
    for s in (7, 8, 9):
        runs[f"{t} s{s}"] = ep(n, s)
runs["R0-dyn s8"] = F / "results/r0_dynsample_seed8/episodes.jsonl"
load = lambda p: {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
D = {k: load(p) for k, p in runs.items()}
ids = sorted(D["B0"])


def m(r, k):
    return (r.get("metrics") or {}).get(k) if r["status"] == "completed" else None


def em(r):
    return 1.0 if m(r, "answer_exact_match") == 1 else 0.0


def cited(r):
    return r["status"] == "completed" and bool(r.get("citations"))


def gem(r):
    return 1.0 if em(r) and cited(r) and m(r, "citation_precision") == 1 else 0.0


def gem_lenient(r):
    return 1.0 if em(r) and cited(r) and (m(r, "citation_precision") or 0) > 0 else 0.0


def sup_em(r):
    return (
        1.0
        if cited(r) and m(r, "citation_precision") == 1 and m(r, "evidence_coverage") == 1
        else 0.0
    )


def joint_em(r):
    return em(r) * sup_em(r)


def sup_f1(r):
    p, c = (m(r, "citation_precision") or 0), (m(r, "evidence_coverage") or 0)
    return 2 * p * c / (p + c) if cited(r) and p + c else 0.0


def joint_f1(
    r,
):  # HotpotQA joint F1: P = Pa*Ps, R = Ra*Rs; approximated with answer F1 for both sides
    a = m(r, "answer_f1") or 0.0
    p, c = (m(r, "citation_precision") or 0), (m(r, "evidence_coverage") or 0)
    if not cited(r):
        return 0.0
    P, Rc = a * p, a * c
    return 2 * P * Rc / (P + Rc) if P + Rc else 0.0


def pct(f, d):
    return 100 * sum(f(d[i]) for i in ids) / len(ids)


# --- #6, #9, #16: extended per-run metrics ---------------------------------------------
print("## Extended metrics per run (all 300 questions; failures = 0 unless noted)\n")
print(
    "| Run | EM | EM on completed | Grounded EM | Lenient gEM | Sup-EM | Sup-F1 | Joint-EM | Joint-F1 | Citation rate | Precision when cited |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
rows = {}
for k, d in D.items():
    comp = [d[i] for i in ids if d[i]["status"] == "completed"]
    cit = [r for r in comp if cited(r)]
    prec_cited = (
        statistics.mean(m(r, "citation_precision") or 0 for r in cit) if cit else float("nan")
    )
    rows[k] = dict(
        em=pct(em, d),
        emc=100 * sum(map(em, comp)) / len(comp),
        gem=pct(gem, d),
        lgem=pct(gem_lenient, d),
        sem=pct(sup_em, d),
        sf1=pct(sup_f1, d),
        jem=pct(joint_em, d),
        jf1=pct(joint_f1, d),
        crate=100 * len(cit) / len(comp),
        pc=prec_cited,
    )
    x = rows[k]
    print(
        f"| {k} | {x['em']:.1f} | {x['emc']:.1f} | {x['gem']:.1f} | {x['lgem']:.1f} | {x['sem']:.1f} | {x['sf1']:.1f} | "
        f"{x['jem']:.1f} | {x['jf1']:.1f} | {x['crate']:.1f}% | {x['pc']:.3f} |"
    )

print("\n## 3-seed means (± sd)\n")
print(
    "| Method | EM | EM on completed | Grounded EM | Lenient gEM | Sup-EM | Joint-EM | Joint-F1 | Citation rate | Precision when cited |"
)
print("|---|---|---|---|---|---|---|---|---|---|")
for t in NAMES:
    xs = [rows[f"{t} s{s}"] for s in (7, 8, 9)]

    def ms(key, fmt="{:.1f}", xs=xs):
        v = [x[key] for x in xs if not math.isnan(x[key])]
        return (
            (fmt + " ± " + fmt).format(statistics.mean(v), statistics.stdev(v))
            if len(v) > 1
            else (fmt.format(v[0]) if v else "–")
        )

    print(
        f"| {t} | {ms('em')} | {ms('emc')} | {ms('gem')} | {ms('lgem')} | {ms('sem')} | {ms('jem')} | {ms('jf1')} | {ms('crate')} | {ms('pc', '{:.3f}')} |"
    )

# Precision on questions every run completed (#16 selection bias)
common = [i for i in ids if all(D[k][i]["status"] == "completed" for k in D)]
print(f"\n## Questions completed by every run: {len(common)}/300\n")
print("| Run | Cit. precision (common set) | Coverage (common set) | EM (common set) |")
print("|---|---|---|---|")
for k, d in D.items():
    cs = [d[i] for i in common]
    print(
        f"| {k} | {statistics.mean(m(r, 'citation_precision') or 0 for r in cs):.3f} | "
        f"{statistics.mean(m(r, 'evidence_coverage') or 0 for r in cs):.3f} | {100 * statistics.mean(map(em, cs)):.1f} |"
    )


# --- #4: hierarchical (seed x question) bootstrap -----------------------------------------
def hboot(f, A, B, n=10000, seed=0):
    rng = random.Random(seed)
    va = [[f(a[i]) for i in ids] for a in A]
    vb = [[f(b[i]) for i in ids] for b in B]
    point = 100 * (
        statistics.mean(map(statistics.mean, va)) - statistics.mean(map(statistics.mean, vb))
    )
    from itertools import product

    N = len(ids)
    out = []

    # per-question seed-averages for every possible seed multiset draw (3^3 = 27 combos)
    def combos(v):
        return [
            [sum(v[s][j] for s in c) / len(c) for j in range(N)]
            for c in product(range(len(v)), repeat=len(v))
        ]

    ca, cb = combos(va), combos(vb)
    for _ in range(n):
        A_, B_ = ca[rng.randrange(len(ca))], cb[rng.randrange(len(cb))]
        q = [rng.randrange(N) for _ in range(N)]
        out.append(100 * sum(A_[j] - B_[j] for j in q) / N)
    out.sort()
    return point, out[int(0.025 * n)], out[int(0.975 * n)]


G = {t: [D[f"{t} s{s}"] for s in (7, 8, 9)] for t in NAMES}
print("\n## Seed-aware (hierarchical) paired bootstrap, 10k resamples, 95% CI (points)\n")
print(
    "Each resample draws 3 seeds with replacement per method and 300 questions with replacement (shared across methods).\n"
)
print("| Comparison | ΔEM | ΔGrounded EM | ΔJoint-EM |")
print("|---|---|---|---|")
for a, b in [
    ("R0", None),
    ("R1", None),
    ("R2", None),
    ("R3", None),
    ("R4", None),
    ("R1", "R0"),
    ("R2", "R1"),
    ("R3", "R1"),
    ("R4", "R1"),
]:
    cells = []
    for f in (em, gem, joint_em):
        p, lo, hi = hboot(f, G[a], G[b] if b else [D["B0"]])
        cells.append(f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    print(f"| {a} − {b or 'B0'} | " + " | ".join(cells) + " |")


# Collapse frequency with Clopper-Pearson (2 of 3 R0 seeds cite nothing)
def beta_ppf(q, a, b, lo=0.0, hi=1.0):
    from math import exp, lgamma

    def inc(x):  # regularized incomplete beta via simple numeric integration (fine for small a, b)
        n = 4000
        s = 0.0
        for i in range(n):
            t = (i + 0.5) / n * x
            s += t ** (a - 1) * (1 - t) ** (b - 1)
        return s * x / n / exp(lgamma(a) + lgamma(b) - lgamma(a + b))

    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if inc(mid) < q else (lo, mid)
    return (lo + hi) / 2


k, n = 2, 3
print(
    f"\nR0 citation collapse: {k}/{n} seeds; Clopper–Pearson 95% CI "
    f"[{beta_ppf(0.025, k, n - k + 1):.2f}, {beta_ppf(0.975, k + 1, n - k):.2f}]"
)

# --- #2: why R0 drops citations: invalid rollouts by kind during training ------------------
print("\n## Training rollouts by outcome, per 32-step block (R0 s8, R0 s9, R1 s8)\n")
for run in ("r0_outcome_seed8", "r0_outcome_seed9", "r1_full_seed8"):
    steps = [json.loads(l) for l in open(F / "runs" / run / "metrics.jsonl", encoding="utf-8")]
    print(f"**{run}**\n")
    print(
        "| Steps | rollouts | invalid: evidence | invalid: protocol | valid with citations | valid, no citations | mean reward: valid+cited | mean reward: valid, uncited |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for b in range(0, len(steps), 32):
        ro = [x for s in steps[b : b + 32] for x in s.get("rollouts", [])]
        inv = Counter(x.get("failure_kind") for x in ro if x["status"] != "valid")
        val = [x for x in ro if x["status"] == "valid"]
        # rollouts log no citation list; precision > 0 means "cited at least one gold sentence"
        vc = [x for x in val if (x.get("citation_precision") or 0) > 0]
        vn = [x for x in val if (x.get("citation_precision") or 0) == 0]
        mr = lambda xs: f"{statistics.mean(x['reward'] for x in xs):.2f}" if xs else "–"
        print(
            f"| {b + 1}–{b + 32} | {len(ro)} | {inv.get('evidence', 0)} | {inv.get('protocol', 0)} | {len(vc)} | {len(vn)} | {mr(vc)} | {mr(vn)} |"
        )
    print()

# --- #24: protocol failures caused by '&' in titles colliding with the citation format -------
print("## Holdout protocol failures caused by '&' / '&amp;' in titles\n")
print("| Run | protocol failures | of which title contains '&' | evidence failures |")
print("|---|---|---|---|")
for k, d in D.items():
    fails = [d[i] for i in ids if d[i]["status"] != "completed"]
    prot = [r for r in fails if r.get("failure_kind") == "protocol"]
    amp = [r for r in prot if "&" in str((r.get("outputs") or [""])[-1])]
    print(
        f"| {k} | {len(prot)} | {len(amp)} | {sum(1 for r in fails if r.get('failure_kind') == 'evidence')} |"
    )
