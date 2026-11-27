# Needle paper (IEEE conference, 6 pages)

`main.tex` is the thesis as a two-column IEEE conference paper (stock `IEEEtran`, conference mode). It is self-contained: the text is here, the numbers come from `docs/thesis/chapters-id/` (newest source, 2026-10-27) and `logs/2026-10-13-peer-review-reanalysis.md`.

## Build

```bash
bash paper/build.sh            # figures + PDF -> paper/main.pdf
bash paper/build.sh --no-figs  # PDF only
```

`build.sh` needs Python with matplotlib (for the figures) and one LaTeX engine: [tectonic](https://tectonic-typesetting.github.io/) (preferred, downloads packages on first run) or `latexmk`. On Overleaf, upload `main.tex`, `refs.bib`, `fewshot.tex` and `figures/*.pdf`, and compile with pdfLaTeX.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Paper text. `% EXTEND` comments mark what to add for an 8-page limit. |
| `refs.bib` | References, converted from `docs/thesis/chapters-id/daftar-pustaka.md`; arXiv IDs checked 2026-11-27. |
| `make_figures.py` | Re-renders the thesis figures at IEEE column width (3.5 in) into `figures/`. It wraps `scripts/make_thesis_figures.py`, so the book figures are unchanged. |
| `fewshot.tex` | Values for the few-shot baseline row (B0-FS). Placeholder until the run exists. |
| `fill_fewshot.py` | Writes `fewshot.tex` from a finished B0-FS run. |

## Few-shot baseline (B0-FS, issue #9)

1. On a RunPod L40S pod set up with `docs/runbooks/runpod.md` steps 0–3: `cd /workspace/needle && git pull && bash scripts/run_fewshot_b0.sh`. Expect ~0.5–1 GPU-hour; the script stops itself after 2.0 h.
2. Download `results/<date>-b0-fewshot/` into the repo, then **stop the pod**.
3. `python -P paper/fill_fewshot.py results/<date>-b0-fewshot && bash paper/build.sh --no-figs`.

The paper text switches from "pending" to the measured values automatically.
