# Needle paper (ICGHIT 2027, 6 pages)

`main.tex` is the thesis as a two-column conference paper for ICGHIT 2027 (track: AI & Big Neural Network). It uses the conference's LaTeX template, which is the IEEE conference template of 2024-06-27; its `IEEEtran.cls` (v1.8b) is pinned in this folder. The limit is 6 pages including references, and both PDFs are exactly 6 pages. The numbers come from `docs/thesis/chapters-id/` and `logs/2026-10-13-peer-review-reanalysis.md`.

## Build

```bash
bash paper/build.sh            # figures + main.pdf + main-anon.pdf
bash paper/build.sh --no-figs  # PDFs only
```

- `main.pdf` has the author block and the code link.
- `main-anon.pdf` is the blind-review version. `main-anon.tex` defines `\anonymous` and inputs `main.tex`, which then drops the author, affiliation and repository link.

`build.sh` needs Python with matplotlib (for the figures) and a LaTeX engine: [tectonic](https://tectonic-typesetting.github.io/) (preferred) or `latexmk`. The text is set in Times (Type 1) under both engines. The figures embed TrueType fonts, so no Type 3 fonts reach the PDF, which IEEE PDF eXpress expects. On Overleaf, upload `main.tex`, `main-anon.tex`, `IEEEtran.cls`, `refs.bib`, `fewshot.tex` and `figures/*.pdf`, then compile with pdfLaTeX.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Paper text. `% EXTEND` comments mark what to add if the page limit grows. |
| `main-anon.tex` | Blind-review wrapper around `main.tex`. |
| `refs.bib` | References, converted from `docs/thesis/chapters-id/daftar-pustaka.md`. arXiv IDs were checked on 2026-11-27. |
| `make_figures.py` | Re-renders the thesis figures at IEEE column width into `figures/`. It wraps `scripts/make_thesis_figures.py`, so the book figures are unchanged. |
| `make_dynamics.py` | Renders `figures/fig_dynamics.pdf`, the training diagnostics for all 16 runs (32-step blocks). It reads each run's `metrics.jsonl`. |
| `fewshot.tex` | Values for the few-shot baseline row (B0-FS). A placeholder until the run exists. |
| `fill_fewshot.py` | Writes `fewshot.tex` from a finished B0-FS run. |

## Few-shot baseline (B0-FS)

1. On a RunPod L40S pod set up with steps 0–3 of `docs/runbooks/runpod.md`, run `cd /workspace/needle && git pull && bash scripts/run_fewshot_b0.sh`. Expect about 0.5–1 GPU-hour; the script stops itself after 2.0 h.
2. Download `results/<date>-b0-fewshot/` into the repo, then **stop the pod**.
3. Run `python -P paper/fill_fewshot.py results/<date>-b0-fewshot && bash paper/build.sh --no-figs`.

The paper text then switches from "pending" to the measured values. Recheck the page count afterwards.
