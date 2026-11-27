"""Build thesis Figures 4.1 and 4.2 from training metrics.jsonl files.

Figure 4.1: skipped (zero-variance) GRPO steps per run.
Figure 4.2: mean training-rollout citation precision per 32-step block, R0 s8 vs. R1 s8.

Writes PDF + PNG and a JSON of the plotted values to docs/thesis/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "thesis" / "figures"
BLOCK = 32

FIXES = ROOT / "results/2026-10-11-fixes"
FIXES_RUNS = {
    "R0 s9": "r0_outcome_seed9",
    "R1 s9": "r1_full_seed9",
    "R2 s9": "r2_gated_seed9",
    "R3 s7": "r3_coverage2_seed7",
    "R3 s8": "r3_coverage2_seed8",
    "R3 s9": "r3_coverage2_seed9",
    "R4 s7": "r4_drgrpo_seed7",
    "R4 s8": "r4_drgrpo_seed8",
    "R4 s9": "r4_drgrpo_seed9",
    "R0-dyn s8": "r0_dynsample_seed8",
}

RUNS = {
    "R0 s7": ROOT / "results/2026-10-10-batch/r0_outcome_seed7/metrics.jsonl",
    "R0 s8": ROOT / "results/2026-10-10-batch/r0_outcome_seed8/metrics.jsonl",
    "R1 s7": ROOT / "results/2026-10-10-grpo-run2/metrics.jsonl",
    "R1 s8": ROOT / "results/2026-10-10-batch/r1_full_seed8/metrics.jsonl",
    "R2 s7": ROOT / "results/2026-10-10-batch/r2_gated_seed7/metrics.jsonl",
    "R2 s8": ROOT / "results/2026-10-10-batch/r2_gated_seed8/metrics.jsonl",
    **{name: FIXES / "runs" / run / "metrics.jsonl" for name, run in FIXES_RUNS.items()},
}
RUN_ORDER = [
    "R0 s7",
    "R0 s8",
    "R0 s9",
    "R1 s7",
    "R1 s8",
    "R1 s9",
    "R2 s7",
    "R2 s8",
    "R2 s9",
    "R3 s7",
    "R3 s8",
    "R3 s9",
    "R4 s7",
    "R4 s8",
    "R4 s9",
    "R0-dyn s8",
]
METHODS = ["R0", "R1", "R2", "R3", "R4"]
METHOD_LABEL = {
    "R0": "R0 outcome-only",
    "R1": "R1 VEGR",
    "R2": "R2 bersyarat (gated)",
    "R3": "R3 cakupan ×2",
    "R4": "R4 tanpa norm. std",
    "R0-dyn": "R0-dyn (1 run)",
}

# Okabe-Ito (colorblind-safe); hatching and markers give a second, non-color cue.
STYLE = {
    "R0": {"color": "#D55E00", "hatch": "//", "marker": "s", "linestyle": "--"},
    "R1": {"color": "#0072B2", "hatch": "", "marker": "o", "linestyle": "-"},
    "R2": {"color": "#009E73", "hatch": "..", "marker": "^", "linestyle": ":"},
    "R3": {"color": "#CC79A7", "hatch": "xx", "marker": "v", "linestyle": "-."},
    "R4": {"color": "#E69F00", "hatch": "\\\\", "marker": "P", "linestyle": "--"},
    "R0-dyn": {"color": "#8C3B00", "hatch": "++", "marker": "X", "linestyle": "--"},
}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def skipped_steps(rows: list[dict]) -> tuple[int, int]:
    return sum(row["skipped_reason"] is not None for row in rows), len(rows)


def precision_by_block(rows: list[dict]) -> list[dict]:
    blocks: dict[int, list[float]] = {}
    for row in rows:
        block = (row["step"] - 1) // BLOCK
        values = [
            r["citation_precision"]
            for r in row.get("rollouts", [])
            if r.get("citation_precision") is not None
        ]
        blocks.setdefault(block, []).extend(values)
    return [
        {
            "steps": f"{b * BLOCK + 1}-{(b + 1) * BLOCK}",
            "mean_precision": sum(v) / len(v) if v else None,
            "n_valid_rollouts": len(v),
        }
        for b, v in sorted(blocks.items())
    ]


def system_diagram() -> None:
    """Figure 3.1: one training episode and the GRPO update."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.6,rounding_size=1.5",
                facecolor=color,
                edgecolor="#333333",
                linewidth=1,
            )
        )
        ax.text(
            x + w / 2, y + h - 2.2, title, ha="center", va="top", fontweight="bold", fontsize=8.5
        )
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7, linespacing=1.35)

    def arrow(start, end, label="", offset=(0, 1.4), style="-|>", rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle=style,
                mutation_scale=11,
                color="#333333",
                linewidth=1.1,
                connectionstyle=f"arc3,rad={rad}",
            )
        )
        if label:
            mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
            ax.text(mx, my, label, ha="center", va="bottom", fontsize=6.8, family="monospace")

    box(
        1,
        30,
        17,
        18,
        "Pertanyaan",
        "distractor setting\n10 paragraf\n(2 emas + 8 pengecoh)",
        "#F2F2F2",
    )
    box(
        27,
        30,
        21,
        18,
        "Agen (policy)",
        "Qwen2.5-7B-Instruct\n+ LoRA (r=16)\nsatu aksi per giliran",
        "#DCEBF7",
    )
    box(
        58,
        30,
        21,
        18,
        "SearchEnvironment",
        "BM25 pada 10\nparagraf, 3 dokumen teratas\nmaks. 3 pencarian",
        "#FBE6D9",
    )
    box(
        58,
        3,
        21,
        18,
        "Reward VEGR",
        "EM + F1 + cakupan\n+ presisi + recall\n− 0,1·srch − 0,1·dup\ntidak valid → −1",
        "#DDF1E8",
    )
    box(
        27,
        3,
        21,
        18,
        "Pembaruan GRPO",
        "G = 8 rollout\nadvantage dinormalisasi grup\nobjektif clipped, ε = 0,2\ngrup variansi nol dilewati",
        "#EDE7F6",
    )
    box(83, 30, 16, 18, "Label emas", "jawaban\nsupporting facts\n(judul, kalimat)", "#F2F2F2")

    arrow((18.6, 39), (26.4, 39))
    arrow((48.6, 44), (57.4, 44), "SEARCH[q]", offset=(0, 0.5))
    arrow((48.6, 38), (57.4, 38), "ANSWER[a]", offset=(0, 0.5))
    arrow((57.4, 32), (48.6, 32), "hasil", offset=(0, 0.5))
    arrow((68.5, 29.4), (68.5, 21.6), "episode\nselesai", offset=(-6.5, -2.5))
    arrow((91, 29.4), (79.6, 15), rad=-0.2)
    arrow((57.4, 12), (48.6, 12), "reward ×8")
    arrow((37.5, 21.6), (37.5, 29.4), "perbarui LoRA", offset=(-7.5, -1.5))
    fig.savefig(OUT / "fig3_1_system.pdf")
    fig.savefig(OUT / "fig3_1_system.png", dpi=300)
    plt.close(fig)


EPISODES = {
    "B0": ROOT / "results/2026-10-10-grpo-smoke/b0_episodes.jsonl",
    "R0 s7": ROOT / "results/2026-10-10-batch/r0_outcome_seed7/episodes.jsonl",
    "R0 s8": ROOT / "results/2026-10-10-batch/r0_outcome_seed8/episodes.jsonl",
    "R1 s7": ROOT / "results/2026-10-10-grpo-run2/grpo_run2_episodes.jsonl",
    "R1 s8": ROOT / "results/2026-10-10-batch/r1_full_seed8/episodes.jsonl",
    "R2 s7": ROOT / "results/2026-10-10-batch/r2_gated_seed7/episodes.jsonl",
    "R2 s8": ROOT / "results/2026-10-10-batch/r2_gated_seed8/episodes.jsonl",
    **{name: FIXES / "results" / run / "episodes.jsonl" for name, run in FIXES_RUNS.items()},
}

FAILURE_KINDS = ("evidence", "protocol", "budget")


def episode_table(path: Path) -> dict[str, dict]:
    return {row["question_id"]: row for row in load(path)}


def em_all(row: dict) -> float:
    if row["status"] != "completed":
        return 0.0
    return float(row["metrics"]["answer_exact_match"])


def grounded_em(row: dict) -> float:
    """EM = 1, at least one citation, and every citation a gold fact (precision 1)."""
    if em_all(row) != 1.0 or not row["citations"]:
        return 0.0
    return float(row["metrics"]["citation_precision"] == 1.0)


def run_summary(rows: dict[str, dict]) -> dict:
    done = [r for r in rows.values() if r["status"] == "completed"]
    return {
        "em_all": sum(em_all(r) for r in rows.values()) / len(rows),
        "grounded_em": sum(grounded_em(r) for r in rows.values()) / len(rows),
        "citation_precision": sum(r["metrics"]["citation_precision"] for r in done) / len(done),
        "completed": len(done),
        "failures": {
            k: sum(r.get("failure_kind") == k for r in rows.values()) for k in FAILURE_KINDS
        },
    }


def per_question(runs: list[dict[str, dict]], score) -> dict[str, float]:
    """Score per question, averaged over the given runs (seeds)."""
    ids = set.intersection(*(set(r) for r in runs))
    return {q: sum(score(r[q]) for r in runs) / len(runs) for q in ids}


def paired_bootstrap(
    a: dict[str, float], b: dict[str, float], resamples: int = 10_000, seed: int = 0
) -> dict:
    import numpy as np

    ids = sorted(set(a) & set(b))
    diff = np.array([a[q] - b[q] for q in ids])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(ids), size=(resamples, len(ids)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "delta": float(diff.mean()) * 100,
        "low": float(low) * 100,
        "high": float(high) * 100,
        "n": len(ids),
        "resamples": resamples,
    }


def seed_bootstrap(
    a_runs: list[dict[str, dict]],
    b_runs: list[dict[str, dict]],
    score,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Seed-aware paired bootstrap, the same procedure as scripts/peer_review_reanalysis.py:
    each resample draws seeds with replacement per method and questions with replacement."""
    import random
    from itertools import product

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
    return {
        "delta": point,
        "low": out[int(0.025 * resamples)],
        "high": out[int(0.975 * resamples)],
        "n": len(ids),
        "resamples": resamples,
    }


def method_of(name: str) -> str:
    return name.split()[0]


def ch4_figures() -> dict:
    episodes = {name: episode_table(path) for name, path in EPISODES.items()}
    summaries = {name: run_summary(rows) for name, rows in episodes.items()}
    points = {
        name: {
            "em_all": s["em_all"],
            "grounded_em": s["grounded_em"],
            "citation_precision": s["citation_precision"],
            "completed": s["completed"],
        }
        for name, s in summaries.items()
    }
    style_b0 = {"color": "#666666", "marker": "D", "hatch": "xx", "linestyle": "-"}

    def style(name):
        return style_b0 if name == "B0" else STYLE[method_of(name)]

    # Figure 4.3: EM vs. citation precision, every run.
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
    for name, p in points.items():
        st = style(name)
        ax.scatter(
            p["em_all"] * 100,
            p["citation_precision"],
            s=46,
            marker=st["marker"],
            facecolor=st["color"],
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
    labels_43 = {
        "B0": (8, -3),
        "R0 s7": (8, -3),
        "R0 s8": (-6, 8),
        "R0 s9": (0, 8),
        "R0-dyn s8": (0, 8),
    }
    for name, offset in labels_43.items():
        p = points[name]
        ax.annotate(
            name,
            (p["em_all"] * 100, p["citation_precision"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            ha="center" if offset[0] <= 0 else "left",
        )
    ax.annotate(
        "R1–R4, semua seed",
        (43, 0.79),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=7.5,
    )
    keys = ["B0", *METHODS, "R0-dyn"]
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=style(k if k == "B0" else k + " x")["marker"],
            color=style(k if k == "B0" else k + " x")["color"],
            markersize=6,
            label="B0 (tanpa pelatihan)" if k == "B0" else METHOD_LABEL[k],
        )
        for k in keys
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower left")
    ax.set_xlim(15, 55)
    ax.set_ylim(-0.06, 1.0)
    ax.set_xlabel("EM pada holdout (%), episode gagal dihitung 0")
    ax.set_ylabel("Presisi sitasi\n(episode yang selesai)")
    ax.set_title("Akurasi serupa, grounding berbeda", loc="left")
    fig.savefig(OUT / "fig4_3_em_vs_precision.pdf")
    fig.savefig(OUT / "fig4_3_em_vs_precision.png", dpi=300)
    plt.close(fig)

    # Figure 4.4: EM and grounded EM per seed for each method (two panels, one y-scale).
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), layout="constrained", sharey=True)
    seed_points = {}
    for ax, key, title in zip(
        axes, ("em_all", "grounded_em"), ("EM", "Precision-grounded EM"), strict=False
    ):
        seed_points[key] = {}
        for i, method in enumerate(METHODS):
            names = [n for n in RUN_ORDER if method_of(n) == method]
            values = [points[n][key] * 100 for n in names]
            seed_points[key][method] = dict(zip(names, values, strict=False))
            st = STYLE[method]
            for n, v in zip(names, values, strict=False):
                x = i + {"s7": -0.15, "s8": 0.0, "s9": 0.15}[n.split()[1]]
                ax.scatter(
                    x,
                    v,
                    s=36,
                    marker=st["marker"],
                    facecolor=st["color"],
                    edgecolor="white",
                    linewidth=1,
                    zorder=3,
                )
            mean = sum(values) / len(values)
            ax.hlines(mean, i - 0.28, i + 0.28, color="#333333", linewidth=1.5)
            ax.annotate(
                f"{mean:.1f}",
                (i + 0.28, mean),
                xytext=(2, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.5,
            )
        dyn = points["R0-dyn s8"][key] * 100
        ax.scatter(
            5, dyn, s=40, marker=STYLE["R0-dyn"]["marker"], color=STYLE["R0-dyn"]["color"], zorder=3
        )
        ax.axhline(points["B0"][key] * 100, color="#666666", linestyle=":", linewidth=1)
        ax.text(5.45, points["B0"][key] * 100 + 0.8, "B0", color="#666666", fontsize=7, ha="right")
        ax.set_xticks(range(6), [*METHODS, "R0-dyn\n(s8)"], fontsize=7.5)
        ax.set_xlim(-0.6, 5.6)
        ax.set_title(title, loc="left")
    axes[0].set_ylim(-2, 55)
    axes[0].set_ylabel("Skor holdout (%), 300 pertanyaan")
    fig.suptitle(
        "Skor holdout per seed (titik = seed 7, 8, 9; garis = rata-rata)",
        x=0.02,
        ha="left",
        fontsize=9,
    )
    fig.savefig(OUT / "fig4_4_em_per_seed.pdf")
    fig.savefig(OUT / "fig4_4_em_per_seed.png", dpi=300)
    plt.close(fig)

    # Figure 4.5: seed-aware paired bootstrap (seeds and questions resampled), the intervals
    # used for inference in the text. Question-only intervals are kept in the JSON for reference.
    by_method = {m: [episodes[n] for n in RUN_ORDER if method_of(n) == m] for m in METHODS}
    by_method["B0"] = [episodes["B0"]]
    by_method["R0 s8"] = [episodes["R0 s8"]]
    by_method["R0-dyn s8"] = [episodes["R0-dyn s8"]]
    comparisons = [
        ("R0", "B0"),
        ("R1", "B0"),
        ("R2", "B0"),
        ("R3", "B0"),
        ("R4", "B0"),
        ("R1", "R0"),
        ("R2", "R1"),
        ("R3", "R1"),
        ("R4", "R1"),
        ("R0-dyn s8", "R0 s8"),
    ]
    boot, seed_boot = {}, {}
    for metric, score in (("em", em_all), ("grounded_em", grounded_em)):
        boot[metric] = {
            f"{a} − {b}": paired_bootstrap(
                per_question(by_method[a], score), per_question(by_method[b], score)
            )
            for a, b in comparisons
        }
        seed_boot[metric] = {
            f"{a} − {b}": seed_bootstrap(by_method[a], by_method[b], score)
            for a, b in comparisons
            if len(by_method[a]) > 1
        }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), layout="constrained", sharey=True)
    labels = [f"{a} − {b}" for a, b in comparisons]
    for ax, metric, title in zip(
        axes, ("em", "grounded_em"), ("ΔEM", "ΔPrecision-grounded EM"), strict=False
    ):
        for y, label in enumerate(reversed(labels)):
            single_run = label not in seed_boot[metric]
            r = boot[metric][label] if single_run else seed_boot[metric][label]
            st = STYLE[method_of(label)]
            ax.errorbar(
                r["delta"],
                y,
                xerr=[[r["delta"] - r["low"]], [r["high"] - r["delta"]]],
                fmt=st["marker"],
                color=st["color"],
                capsize=3,
                markersize=5,
                mfc="white" if single_run else st["color"],
            )
        ax.axvline(0, color="#333333", linewidth=1)
        ax.axhline(0.5, color="#BBBBBB", linewidth=0.8, linestyle="--")
        ax.axhline(4.5, color="#BBBBBB", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"{title} (poin), SK 95%")
    axes[0].set_yticks(range(len(labels)), list(reversed(labels)))
    axes[1].text(
        0.98,
        0.98,
        "terhadap B0 (RQ1)",
        transform=axes[1].transAxes,
        fontsize=7,
        va="top",
        ha="right",
        color="#555555",
    )
    axes[1].text(
        0.98,
        0.14,
        "antarmetode\n(RQ2/RQ3)",
        transform=axes[1].transAxes,
        fontsize=7,
        va="bottom",
        ha="right",
        color="#555555",
    )
    fig.suptitle(
        "Selisih berpasangan pada holdout (SK yang memperhitungkan seed;\n"
        "penanda kosong = satu run, SK hanya me-resample pertanyaan)",
        x=0.02,
        ha="left",
        fontsize=9,
    )
    fig.savefig(OUT / "fig4_5_bootstrap.pdf")
    fig.savefig(OUT / "fig4_5_bootstrap.png", dpi=300)
    plt.close(fig)

    # Figure 4.6: failure breakdown.
    fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    names = ["B0", *RUN_ORDER]
    kind_style = {
        "evidence": ("#CC79A7", ""),
        "protocol": ("#E69F00", "//"),
        "budget": ("#56B4E9", ".."),
    }
    kind_label = {"evidence": "bukti", "protocol": "protokol", "budget": "anggaran"}
    bottoms = [0] * len(names)
    for kind in FAILURE_KINDS:
        values = [summaries[n]["failures"][kind] for n in names]
        color, hatch = kind_style[kind]
        ax.bar(
            range(len(names)),
            values,
            bottom=bottoms,
            color=color,
            hatch=hatch,
            edgecolor="white",
            width=0.7,
            label=kind_label[kind],
        )
        bottoms = [b + v for b, v in zip(bottoms, values, strict=False)]
    for i, total in enumerate(bottoms):
        ax.text(i, total + 2, str(total), ha="center", fontsize=7)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Episode gagal (dari 300)")
    ax.set_title("Episode holdout yang gagal menurut jenisnya", loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(OUT / "fig4_6_failures.pdf")
    fig.savefig(OUT / "fig4_6_failures.png", dpi=300)
    plt.close(fig)

    # Figure 4.7: cited sentence index, R1 s8.
    cited = {}
    for row in episodes["R1 s8"].values():
        if row["status"] != "completed":
            continue
        for c in row["citations"]:
            index = int(c.rsplit("|", 1)[1])
            cited[index] = cited.get(index, 0) + 1
    fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
    xs = sorted(cited)
    ax.bar(xs, [cited[x] for x in xs], color=STYLE["R1"]["color"], edgecolor="white", width=0.7)
    for x in xs:
        ax.text(x, cited[x] + 3, str(cited[x]), ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_ylim(0, max(cited.values()) * 1.12)
    ax.set_xlabel("Indeks kalimat di dalam paragraf yang disitasi")
    ax.set_ylabel("Jumlah sitasi")
    ax.set_title("Kalimat yang disitasi R1 (seed 8)", loc="left")
    fig.savefig(OUT / "fig4_7_cited_index.pdf")
    fig.savefig(OUT / "fig4_7_cited_index.png", dpi=300)
    plt.close(fig)

    return {
        "points": points,
        "seed_points": seed_points,
        "bootstrap": {
            "note": "question-only: per-question scores averaged over seeds 7/8/9 per "
            "method; 10k resamples, RNG seed 0",
            **boot,
        },
        "seed_bootstrap": {
            "note": "seed-aware (plotted in Figure 4.5): seeds and questions "
            "resampled; 10k resamples, random.Random(0)",
            **seed_boot,
        },
        "failures": {n: summaries[n]["failures"] for n in names},
        "cited_sentence_index_r1_s8": cited,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    system_diagram()
    ch4 = ch4_figures()
    data = {name: load(path) for name, path in RUNS.items()}

    # Figure 4.1
    skipped = {name: skipped_steps(rows) for name, rows in data.items()}
    fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
    names = RUN_ORDER
    for i, name in enumerate(names):
        count, total = skipped[name]
        style = STYLE[method_of(name)]
        ax.bar(i, count, color=style["color"], hatch=style["hatch"], edgecolor="white", width=0.7)
        ax.text(
            i,
            count + 3,
            f"{count}" if total == 256 else f"{count}\n/{total}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(0, 300)
    ax.set_ylabel("Langkah dilewati (dari 256;\nR0-dyn: dari 512)")
    ax.set_title("Langkah dengan grup variansi nol yang dilewati selama pelatihan", loc="left")
    fig.savefig(OUT / "fig4_1_skipped_steps.pdf")
    fig.savefig(OUT / "fig4_1_skipped_steps.png", dpi=300)
    plt.close(fig)

    # Figure 4.2
    curves = {name: precision_by_block(data[name][:256]) for name in ("R0 s8", "R1 s8")}
    fig, ax = plt.subplots(figsize=(5.5, 3.0), layout="constrained")
    for name, blocks in curves.items():
        style = STYLE[method_of(name)]
        xs = [i for i, b in enumerate(blocks) if b["mean_precision"] is not None]
        ys = [blocks[i]["mean_precision"] for i in xs]
        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
        )
        ax.annotate(
            name,
            (xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=style["color"],
        )
    labels = [b["steps"] for b in curves["R0 s8"]]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_xlim(-0.3, len(labels) + 0.2)
    ax.set_ylim(-0.02, 1)
    ax.set_xlabel("Langkah pelatihan (blok 32 langkah)")
    ax.set_ylabel("Rata-rata presisi sitasi\n(rollout pelatihan yang valid)")
    ax.set_title("Presisi sitasi selama pelatihan, seed 8", loc="left")
    fig.savefig(OUT / "fig4_2_citation_precision.pdf")
    fig.savefig(OUT / "fig4_2_citation_precision.png", dpi=300)
    plt.close(fig)

    provenance = {
        "sources": {name: str(path.relative_to(ROOT)) for name, path in RUNS.items()},
        "fig4_1_skipped_steps": {
            name: {"skipped": c, "total": t} for name, (c, t) in skipped.items()
        },
        "fig4_2_citation_precision": curves,
        "episode_sources": {name: str(path.relative_to(ROOT)) for name, path in EPISODES.items()},
        "fig4_3_to_4_7": ch4,
    }
    for metric in ("em", "grounded_em"):
        print(
            metric,
            json.dumps(
                {
                    k: [round(v[x], 1) for x in ("delta", "low", "high")]
                    for k, v in ch4["bootstrap"][metric].items()
                },
                ensure_ascii=False,
            ),
        )
    print(
        json.dumps(
            {
                n: {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()}
                for n, p in ch4["points"].items()
            }
        )
    )
    print(json.dumps(ch4["failures"]), json.dumps(ch4["cited_sentence_index_r1_s8"]))
    (OUT / "figures_data.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance["fig4_1_skipped_steps"]))
    for name, blocks in curves.items():
        print(
            name,
            [
                (
                    b["steps"],
                    round(b["mean_precision"], 3) if b["mean_precision"] is not None else None,
                    b["n_valid_rollouts"],
                )
                for b in blocks
            ],
        )


if __name__ == "__main__":
    main()
