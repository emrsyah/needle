#!/usr/bin/env bash
# Build paper/main.pdf (with authors) and paper/main-anon.pdf (blind review).
# Usage: bash paper/build.sh [--no-figs]
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" != "--no-figs" ]]; then
  python -P make_figures.py
  python -P make_dynamics.py
fi

for doc in main main-anon; do
  if command -v tectonic >/dev/null 2>&1; then
    tectonic -X compile "$doc.tex"
  elif command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode "$doc.tex"
    latexmk -c "$doc.tex"
  else
    echo "Install tectonic (https://tectonic-typesetting.github.io/) or latexmk." >&2
    exit 1
  fi
done

python -P - <<'EOF' || true
try:
    import pypdf
    for doc in ("main", "main-anon"):
        print(f"{doc}.pdf pages:", len(pypdf.PdfReader(f"{doc}.pdf").pages))
except ImportError:
    pass
EOF
