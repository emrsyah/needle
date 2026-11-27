"""Training-dynamics figure for the IEEE paper (all runs, 32-step blocks).

Rows are diagnostics, columns are methods, one line per seed. R0-dyn is drawn
in its own column and runs to 512 questions drawn. Every quantity is
independent of the reward scale, so R0 and the VEGR variants are comparable:

* invalid rollouts (% of the 8 x 32 rollouts in a block),
* rollout EM (% of all rollouts, invalid counted as 0),
* citation precision (mean over valid rollouts; 0 when nothing is cited),
* skipped steps (% of steps whose 8 rewards were all equal).

Run lists come from ``scripts/make_thesis_figures.py`` (``RUNS``), which is
loaded, not edited. Writes ``paper/figures/fig_dynamics.pdf`` (+ PNG) and
prints the checks quoted in the paper text.

Usage (from the repo root)::

    python -P paper/make_dynamics.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "make_thesis_figures.py"
OUT = ROOT / "paper" / "figures"
BLOCK = 32
METHODS = ["R0", "R1", "R2", "R3", "R4"]
TITLES = {
    "R0": "R0 outcome-only",
    "R1": "R1 VEGR",
    "R2": "R2 gated",
    "R3": "R3 coverage ×2",
    "R4": "R4 no std norm.",
}
SEED_ALPHA = {"s7": 1.0, "s8": 0.7, "s9": 0.45}
SEED_MARKER = {"s7": "o", "s8": "s", "s9": "^"}


def load_thesis_module():
    spec = importlib.util.spec_from_file_location("make_thesis_figures", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def blocks(rows: list[dict]) -> list[dict]:
    """Aggregate one run's metrics.jsonl rows into 32-step blocks."""
    out: dict[int, dict] = {}
    for row in rows:
        b = out.setdefault(
            (row["step"] - 1) // BLOCK,
            {
                "steps": 0,
                "skipped": 0,
                "rollouts": 0,
                "invalid": 0,
                "evidence": 0,
                "em": 0.0,
                "prec": [],
                "cov": [],
            },
        )
        b["steps"] += 1
        b["skipped"] += row["skipped_reason"] is not None
        for r in row.get("rollouts", []):
            b["rollouts"] += 1
            if r["status"] != "valid":
                b["invalid"] += 1
                b["evidence"] += r.get("failure_kind") == "evidence"
                continue
            b["em"] += r.get("exact_match") or 0.0
            if r.get("citation_precision") is not None:
                b["prec"].append(r["citation_precision"])
                b["cov"].append(r["evidence_coverage"])
    result = []
    for k, b in sorted(out.items()):
        n = b["rollouts"] or 1
        result.append(
            {
                "x": (k + 1) * BLOCK,
                "invalid": 100 * b["invalid"] / n,
                "evidence_failures": b["evidence"],
                "em": 100 * b["em"] / n,
                "precision": sum(b["prec"]) / len(b["prec"]) if b["prec"] else 0.0,
                "coverage": sum(b["cov"]) / len(b["cov"]) if b["cov"] else 0.0,
                "skipped": 100 * b["skipped"] / b["steps"],
            }
        )
    return result


ROWS = [
    ("invalid", "Invalid\nrollouts (%)", (0, 60)),
    ("em", "Rollout\nEM (%)", (0, 60)),
    ("precision", "Citation\nprecision", (0, 1)),
    ("skipped", "Skipped\nsteps (%)", (0, 70)),
]


def main() -> None:
    figs = load_thesis_module()
    plt = figs.plt
    data = {name: blocks(figs.load(path)) for name, path in figs.RUNS.items()}

    # Checks against numbers quoted in the paper.
    skipped = {
        n: sum(1 for r in figs.load(p) if r["skipped_reason"] is not None)
        for n, p in figs.RUNS.items()
    }
    print("skipped steps:", skipped)
    print("R1 s7 invalid % by block:", [round(b["invalid"]) for b in data["R1 s7"]])
    print("R0 s8 evidence failures, block 1:", data["R0 s8"][0]["evidence_failures"])
    for name in figs.RUN_ORDER:
        print(
            f"{name:10s} last-block invalid {data[name][-1]['invalid']:5.1f}  "
            f"EM {data[name][-1]['em']:5.1f}  prec {data[name][-1]['precision']:.3f}"
        )

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )
    columns = [*METHODS, "R0-dyn"]
    fig, axes = plt.subplots(
        len(ROWS),
        len(columns),
        figsize=(7.0, 3.5),
        sharey="row",
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.25]},
    )
    for col, method in enumerate(columns):
        color = figs.STYLE[method]["color"]
        seeds = ("s8",) if method == "R0-dyn" else ("s7", "s8", "s9")
        for row, (key, label, ylim) in enumerate(ROWS):
            ax = axes[row, col]
            for seed in seeds:
                series = data[f"{method} {seed}"]
                ax.plot(
                    [b["x"] for b in series],
                    [b[key] for b in series],
                    color=color,
                    alpha=SEED_ALPHA[seed],
                    lw=1.0,
                    marker=SEED_MARKER[seed],
                    ms=2.2,
                )
            ax.set_ylim(*ylim)
            ax.set_xlim(0, 525 if method == "R0-dyn" else 265)
            ax.set_xticks([0, 256, 512] if method == "R0-dyn" else [0, 128, 256])
            ax.grid(alpha=0.25, lw=0.4)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(TITLES.get(method, "R0-dyn (seed 8)"))
            if row == len(ROWS) - 1:
                ax.set_xlabel("Training step")
            else:
                ax.tick_params(labelbottom=False)
    # One shared legend for the seed markers (colour = method, as in the titles).
    handles = [
        plt.Line2D(
            [],
            [],
            color="0.3",
            alpha=SEED_ALPHA[s],
            marker=SEED_MARKER[s],
            ms=3,
            lw=1.0,
            label=f"seed {s[1]}",
        )
        for s in ("s7", "s8", "s9")
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.4, w_pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_dynamics.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_dynamics.png", dpi=200, bbox_inches="tight")
    print("wrote", OUT / "fig_dynamics.pdf")


if __name__ == "__main__":
    main()
