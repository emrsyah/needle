"""Re-render the thesis figures for the IEEE paper (two-column layout).

This wraps ``scripts/make_thesis_figures.py`` instead of changing it, so the
book figures in ``docs/thesis/figures/`` are untouched. It

* writes every figure to ``paper/figures/`` (PDF + PNG + ``figures_data.json``),
* shrinks single-column figures to the IEEE column width (default 3.5 in) with a
  7 pt base font, while the system diagram stays full text width (7 in) for a
  ``figure*`` float,
* renames the R4 legend label to "R4 no std norm." (R4 is not full Dr. GRPO).

Usage (from the repo root)::

    python -I paper/make_figures.py [--column-width 3.5] [--out paper/figures]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "make_thesis_figures.py"


def load_thesis_module():
    spec = importlib.util.spec_from_file_location("make_thesis_figures", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column-width", type=float, default=3.5, help="inches")
    parser.add_argument("--font-size", type=float, default=7.0)
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "figures")
    args = parser.parse_args()

    figs = load_thesis_module()
    plt = figs.plt
    plt.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3 (IEEE PDF eXpress)
    figs.OUT = args.out.resolve()
    figs.METHOD_LABEL["R4"] = "R4 no std norm."

    original_subplots = plt.subplots
    state = {"scale": True}

    def subplots(*a, **kw):
        size = kw.get("figsize")
        if state["scale"] and size is not None:
            width, height = size
            factor = args.column_width / width
            # Keep a little more height than a pure rescale so labels stay legible.
            kw["figsize"] = (args.column_width, max(height * factor * 1.35, 1.9))
            plt.rcParams["font.size"] = args.font_size
        else:
            plt.rcParams["font.size"] = 8
        fig, axes = original_subplots(*a, **kw)
        if state["scale"] and size == (5.5, 3.6):  # Fig. 4.3: separate two close labels
            axes.annotate = _spread_labels(axes.annotate)
        if not state["scale"]:  # system diagram: English labels for the paper
            axes.text = _english_labels(axes.text)
        return fig, axes

    # The thesis book draws the system diagram with Indonesian labels; the paper
    # keeps its English ones (search term named Srch, as in the paper text).
    english = {
        "Pertanyaan": "Question",
        "distractor setting\n10 paragraf\n(2 emas + 8 pengecoh)": (
            "distractor setting\n10 paragraphs\n(2 gold + 8 distractors)"
        ),
        "Agen (policy)": "Agent (policy)",
        "Qwen2.5-7B-Instruct\n+ LoRA (r=16)\nsatu aksi per giliran": (
            "Qwen2.5-7B-Instruct\n+ LoRA (r=16)\none action per turn"
        ),
        "BM25 pada 10\nparagraf, 3 dokumen teratas\nmaks. 3 pencarian": (
            "BM25 over the 10\nparagraphs, top-3 docs\nmax 3 searches"
        ),
        "Reward VEGR": "VEGR reward",
        "EM + F1 + cakupan\n+ presisi + recall\n− 0,1·srch − 0,1·dup\ntidak valid → −1": (
            "EM + F1 + coverage\n+ precision + recall\n− 0.1·srch − 0.1·dup\ninvalid → −1"
        ),
        "Pembaruan GRPO": "GRPO update",
        "G = 8 rollout\nadvantage dinormalisasi grup\nobjektif clipped, ε = 0,2\n"
        "grup variansi nol dilewati": (
            "G = 8 rollouts\ngroup-normalized advantage\nclipped objective, ε = 0.2\n"
            "zero-variance groups skipped"
        ),
        "Label emas": "Gold labels",
        "jawaban\nsupporting facts\n(judul, kalimat)": (
            "answer\nsupporting facts\n(title, sentence)"
        ),
        "hasil": "results",
        "episode\nselesai": "finished\nepisode",
        "perbarui LoRA": "update LoRA",
    }

    def _english_labels(text_fn):
        def wrapped(x, y, s, *a, **kw):
            return text_fn(x, y, english.get(s, s), *a, **kw)

        return wrapped

    def _spread_labels(annotate):
        nudges = {"R0 s8": ((-4, 8), "right"), "R0-dyn s8": ((-6, 8), "left")}

        def wrapped(text, *a, **kw):
            if text in nudges:
                kw["xytext"], kw["ha"] = nudges[text]
            return annotate(text, *a, **kw)

        return wrapped

    plt.subplots = subplots

    original_diagram = figs.system_diagram

    def full_width_diagram() -> None:
        state["scale"] = False
        try:
            original_diagram()
        finally:
            state["scale"] = True

    figs.system_diagram = full_width_diagram

    with contextlib.redirect_stdout(None):
        figs.main()
    print(f"wrote figures to {figs.OUT}")


if __name__ == "__main__":
    main()
