"""Per-block training dynamics for every GRPO run (thesis Gambar 4.4 and Lampiran B).

Each run is one pass (one epoch) over its training questions, so "per epoch" progress is shown
per block of 32 training steps. For every run and block this script computes, from the run's
metrics.jsonl:

- mean citation precision of the valid training rollouts (invalid rollouts have no precision),
- share of valid rollouts with at least one gold citation (precision > 0),
- share of invalid rollouts (reward -1),
- number of skipped (zero-variance) steps.

Outputs:
- docs/thesis/figures/fig4_4_dynamics_r0_r1.{png,pdf}   (body: R0 and R1, all seeds)
- docs/thesis/figures/figB_1_dynamics_variants.{png,pdf} (appendix: R2, R3, R4, R0-dyn)
- results/2026-11-27-training-dynamics/training_dynamics.{json,csv}
- results/2026-11-27-training-dynamics/tables.md  (Markdown tables used in Lampiran B)

Usage (from the repo root):
    uv run --no-project --with matplotlib python scripts/make_training_dynamics.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "thesis" / "figures"
OUT = ROOT / "results" / "2026-11-27-training-dynamics"
BLOCK = 32

BATCH = ROOT / "results/2026-10-10-batch"
FIXES = ROOT / "results/2026-10-11-fixes/runs"
RUNS = {
    "R0 s7": BATCH / "r0_outcome_seed7", "R0 s8": BATCH / "r0_outcome_seed8",
    "R0 s9": FIXES / "r0_outcome_seed9",
    "R1 s7": ROOT / "results/2026-10-10-grpo-run2", "R1 s8": BATCH / "r1_full_seed8",
    "R1 s9": FIXES / "r1_full_seed9",
    "R2 s7": BATCH / "r2_gated_seed7", "R2 s8": BATCH / "r2_gated_seed8",
    "R2 s9": FIXES / "r2_gated_seed9",
    "R3 s7": FIXES / "r3_coverage2_seed7", "R3 s8": FIXES / "r3_coverage2_seed8",
    "R3 s9": FIXES / "r3_coverage2_seed9",
    "R4 s7": FIXES / "r4_drgrpo_seed7", "R4 s8": FIXES / "r4_drgrpo_seed8",
    "R4 s9": FIXES / "r4_drgrpo_seed9",
    "R0-dyn s8": FIXES / "r0_dynsample_seed8",
}
METHOD_TITLE = {"R0": "R0 (outcome-only)", "R1": "R1 (VEGR)", "R2": "R2 (gated)",
                "R3": "R3 (cakupan ×2)", "R4": "R4 (tanpa norm. std)", "R0-dyn": "R0-dyn (seed 8, 512 langkah)"}
SEED_STYLE = {"s7": ("#0072B2", "o", "-"), "s8": ("#D55E00", "s", "--"), "s9": ("#009E73", "^", ":")}


def load(path: Path) -> list[dict]:
    with (path / "metrics.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def blocks(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["step"] - 1) // BLOCK, []).append(row)
    out = []
    for b, steps in sorted(grouped.items()):
        rollouts = [r for s in steps for r in s.get("rollouts", [])]
        valid = [r["citation_precision"] for r in rollouts if r.get("citation_precision") is not None]
        # "invalid" (protocol/evidence) and "exhausted" (budget) rollouts both score -1.
        invalid = sum(r["status"] != "valid" for r in rollouts)
        out.append({
            "block": b + 1,
            "steps": f"{b * BLOCK + 1}-{(b + 1) * BLOCK}",
            "n_rollouts": len(rollouts),
            "n_valid": len(valid),
            "mean_precision": sum(valid) / len(valid) if valid else None,
            "gold_cite_share": sum(v > 0 for v in valid) / len(valid) if valid else None,
            "invalid_share": invalid / len(rollouts) if rollouts else None,
            "skipped": sum(s["skipped_reason"] is not None for s in steps),
            "mean_reward": sum(r["reward"] for r in rollouts) / len(rollouts) if rollouts else None,
        })
    return out


def method_of(name: str) -> str:
    return name.split()[0]


def panel(ax, data: dict[str, list[dict]], names: list[str], title: str, n_blocks: int) -> None:
    for name in names:
        seed = name.split()[-1]
        color, marker, linestyle = SEED_STYLE[seed]
        ys = [b["mean_precision"] for b in data[name]]
        xs = [i + 1 for i, y in enumerate(ys) if y is not None]
        ax.plot(xs, [y for y in ys if y is not None], color=color, marker=marker, linestyle=linestyle,
                linewidth=1.6, markersize=4, label=f"seed {seed[1:]}")
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlim(0.5, n_blocks + 0.5)
    ax.set_xticks(range(1, n_blocks + 1))
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    data = {name: blocks(load(path)) for name, path in RUNS.items()}

    # Regression checks against numbers already quoted in BAB 4 (Subbab 4.5).
    r0s8 = [round(b["mean_precision"], 3) for b in data["R0 s8"][:3]]
    assert r0s8 == [0.568, 0.499, 0.126], r0s8
    assert [b["skipped"] for b in data["R0 s8"]] and sum(b["skipped"] for b in data["R0 s8"]) == 128
    assert sum(b["skipped"] for b in data["R0-dyn s8"]) == 262
    r1s7 = [round(b["invalid_share"] * 100) for b in data["R1 s7"][:5]]
    assert r1s7 == [46, 29, 13, 13, 7], r1s7  # Subbab 4.3, logs/2026-10-10-grpo-run2.md

    ylabel = "Presisi sitasi rata-rata\n(rollout valid, skala 0–1)"
    xlabel = "Blok pelatihan (1 blok = 32 langkah)"

    # Body figure: R0 and R1, all seeds.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, layout="constrained")
    for ax, method in zip(axes, ("R0", "R1")):
        panel(ax, data, [f"{method} s{s}" for s in (7, 8, 9)], METHOD_TITLE[method], 8)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="lower right", frameon=False)
    fig.savefig(FIG / "fig4_4_dynamics_r0_r1.pdf")
    fig.savefig(FIG / "fig4_4_dynamics_r0_r1.png", dpi=300)
    plt.close(fig)

    # Appendix figure: R2, R3, R4 (all seeds) and R0-dyn (16 blocks).
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), layout="constrained")
    for ax, method in zip(axes.flat[:3], ("R2", "R3", "R4")):
        panel(ax, data, [f"{method} s{s}" for s in (7, 8, 9)], METHOD_TITLE[method], 8)
    ax = axes.flat[3]
    panel(ax, data, ["R0-dyn s8"], METHOD_TITLE["R0-dyn"], 16)
    ax.set_xticks(range(1, 17, 2))
    ax.axvline(8.5, color="grey", linewidth=0.8, linestyle="--")
    ax.text(8.6, 0.9, "langkah 257–512", fontsize=7, color="grey")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[1, :]:
        ax.set_xlabel(xlabel)
    axes.flat[0].legend(loc="lower right", frameon=False)
    fig.savefig(FIG / "figB_1_dynamics_variants.pdf")
    fig.savefig(FIG / "figB_1_dynamics_variants.png", dpi=300)
    plt.close(fig)

    # Data files.
    sources = {name: str((path / "metrics.jsonl").relative_to(ROOT)).replace("\\", "/")
               for name, path in RUNS.items()}
    (OUT / "training_dynamics.json").write_text(
        json.dumps({"block_size": BLOCK, "sources": sources, "runs": data}, indent=2), encoding="utf-8")
    with (OUT / "training_dynamics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "block", "steps", "n_rollouts", "n_valid", "mean_precision",
                         "gold_cite_share", "invalid_share", "skipped", "mean_reward"])
        for name, rows in data.items():
            for b in rows:
                writer.writerow([name, b["block"], b["steps"], b["n_rollouts"], b["n_valid"],
                                 b["mean_precision"], b["gold_cite_share"], b["invalid_share"],
                                 b["skipped"], b["mean_reward"]])

    # Markdown tables for Lampiran B (Indonesian decimal comma).
    def fmt(v, digits=2, pct=False):
        if v is None:
            return "–"
        return (f"{v * 100:.0f}" if pct else f"{v:.{digits}f}").replace(".", ",")

    def table(key, **kw):
        header = "| *Run* | " + " | ".join(f"B{i}" for i in range(1, 9)) + " |"
        lines = [header, "|---|" + "---|" * 8]
        for name, rows in data.items():
            chunks = [rows[:8]] if len(rows) <= 8 else [rows[:8], rows[8:16]]
            for c, chunk in enumerate(chunks):
                label = name if c == 0 else f"{name} (B9–B16)"
                cells = [fmt(b[key], **kw) if key != "skipped" else str(b[key]) for b in chunk]
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md = ["## Presisi sitasi rata-rata (rollout valid)", table("mean_precision"),
          "## Rollout tidak valid (%)", table("invalid_share", pct=True),
          "## Langkah yang dilewati per blok", table("skipped")]
    (OUT / "tables.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    for name, rows in data.items():
        print(name, [fmt(b["mean_precision"]) for b in rows])


if __name__ == "__main__":
    main()
