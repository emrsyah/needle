#!/usr/bin/env bash
# Build paper/main.pdf. Usage: bash paper/build.sh [--no-figs]
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" != "--no-figs" ]]; then
  python -P make_figures.py
fi

if command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile main.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode main.tex
  latexmk -c
else
  echo "Install tectonic (https://tectonic-typesetting.github.io/) or latexmk." >&2
  exit 1
fi

python -P - <<'EOF' || true
try:
    import pypdf
    print("pages:", len(pypdf.PdfReader("main.pdf").pages))
except ImportError:
    pass
EOF
