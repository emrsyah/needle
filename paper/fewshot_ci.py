"""Seed-aware bootstrap intervals for the few-shot control (B0-FS).

Uses the episode loaders and scores of ``scripts/make_thesis_figures.py`` (loaded,
not edited) and the seed-aware paired bootstrap of the thesis analysis: each of
10,000 resamples draws seeds with replacement per method and the 300 questions
with replacement. B0 and B0-FS are single runs, so only the trained side resamples
seeds. As a check, R1 - B0 must reproduce the published +21.2 [+16.4, +26.0].

Usage (from the repo root)::

    python -P paper/fewshot_ci.py results/2026-11-27-b0-fewshot
"""

from __future__ import annotations

import importlib.util
import random
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "make_thesis_figures.py"


def load_thesis_module():
    spec = importlib.util.spec_from_file_location("make_thesis_figures", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def seed_bootstrap(a_runs, b_runs, score, resamples=10_000, seed=0):
    rng = random.Random(seed)
    ids = sorted(set.intersection(*(set(r) for r in a_runs + b_runs)))
    va = [[score(r[q]) for q in ids] for r in a_runs]
    vb = [[score(r[q]) for q in ids] for r in b_runs]
    point = 100 * (
        sum(map(sum, va)) / (len(va) * len(ids)) - sum(map(sum, vb)) / (len(vb) * len(ids))
    )

    def combos(v):
        return [
            [sum(v[s][j] for s in c) / len(c) for j in range(len(ids))]
            for c in product(range(len(v)), repeat=len(v))
        ]

    ca, cb = combos(va), combos(vb)
    out = []
    for _ in range(resamples):
        x, y = ca[rng.randrange(len(ca))], cb[rng.randrange(len(cb))]
        q = [rng.randrange(len(ids)) for _ in range(len(ids))]
        out.append(100 * sum(x[j] - y[j] for j in q) / len(ids))
    out.sort()
    return point, out[int(0.025 * resamples)], out[int(0.975 * resamples)], len(ids)


def main() -> None:
    figs = load_thesis_module()
    fewshot = ROOT / sys.argv[1] / "episodes.jsonl"
    runs = {name: figs.episode_table(path) for name, path in figs.EPISODES.items()}
    runs["B0-FS"] = figs.episode_table(fewshot)
    by = {
        "B0": [runs["B0"]],
        "B0-FS": [runs["B0-FS"]],
        **{m: [runs[f"{m} s{s}"] for s in (7, 8, 9)] for m in ("R0", "R1", "R2", "R3", "R4")},
    }
    pairs = [("R1", "B0"), ("B0-FS", "B0")] + [(m, "B0-FS") for m in ("R0", "R1", "R2", "R3", "R4")]
    for metric, score in (("EM", figs.em_all), ("P-gEM", figs.grounded_em)):
        for a, b in pairs:
            d, lo, hi, n = seed_bootstrap(by[a], by[b], score)
            print(f"{metric:6s} {a:5s} - {b:5s} {d:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]  n={n}")


if __name__ == "__main__":
    main()
