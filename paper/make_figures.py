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
        return fig, axes

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
