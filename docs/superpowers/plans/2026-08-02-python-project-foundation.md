# Needle Python Project Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible, tested Python project foundation for Needle without adding dataset, retrieval, model, or RL dependencies.

**Architecture:** Use a Python 3.11 `src`-layout package managed by uv. Keep runtime dependencies empty at this stage, place pytest and Ruff in the development dependency group, and verify the same checks locally and in GitHub Actions.

**Tech Stack:** Python 3.11, uv, Hatchling, pytest, Ruff, GitHub Actions

---

## File Map

- Create: `pyproject.toml` — project metadata, build backend, development dependencies, pytest configuration, and Ruff configuration.
- Create: `.python-version` — default Python version for uv.
- Create: `.gitignore` — Python, uv, test, editor, and experiment artifact exclusions.
- Create: `src/needle/__init__.py` — package boundary and initial public version.
- Create: `tests/test_package.py` — package import and version smoke test.
- Create: `.github/workflows/ci.yml` — repeatable lint, format, and test checks.
- Modify: `README.md` — local setup and verification commands.
- Generate: `uv.lock` — locked development environment produced by uv.

## Chunk 1: Reproducible Package Skeleton

### Task 1: Add project metadata and lock the environment

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Generate: `uv.lock`

- [ ] **Step 1: Confirm the repository starts clean**

Run: `git status --short`

Expected: no output.

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "needle-agent"
version = "0.1.0"
description = "An evidence-seeking agentic RAG research project."
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[tool.uv]
build-constraint-dependencies = ["hatchling>=1.27,<2"]

[tool.hatch.build.targets.wheel]
packages = ["src/needle"]

[dependency-groups]
dev = [
  "pytest>=9,<10",
  "ruff>=0.9,<1",
]

[tool.pytest.ini_options]
addopts = "-ra -q"
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 3: Create `.python-version`**

```text
3.11
```

- [ ] **Step 4: Lock and synchronize development tools without installing the not-yet-created package**

Run: `uv lock`

Expected: exit code 0 and `uv.lock` exists.

Run: `uv lock --check`

Expected: exit code 0 and no lockfile changes.

Run: `uv sync --locked --dev --no-install-project`

Expected: exit code 0 and `.venv` contains runnable `pytest` and `ruff` commands.

- [ ] **Step 5: Commit reproducible project metadata**

```bash
git add pyproject.toml .python-version uv.lock
git commit -m "build: initialize Python project"
```

### Task 2: Add the importable package using TDD

**Files:**
- Create: `tests/test_package.py`
- Create: `src/needle/__init__.py`

- [ ] **Step 1: Create the source and test directories**

Run in PowerShell: `New-Item -ItemType Directory -Force src/needle, tests`

Expected: exit code 0 and both directories exist.

- [ ] **Step 2: Write the failing package smoke test**

```python
import needle


def test_package_exposes_version() -> None:
    assert needle.__version__ == "0.1.0"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_package.py -v`

Expected: exit code 1, zero collected tests, and collection fails with `ModuleNotFoundError: No module named 'needle'`.

- [ ] **Step 4: Add the minimal package implementation**

```python
"""Needle: an evidence-seeking agentic RAG research project."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Install the package from the locked environment**

Run: `uv sync --locked --dev`

Expected: exit code 0 and the editable `needle-agent` project is installed.

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_package.py -v`

Expected: exit code 0 with `1 passed`.

- [ ] **Step 7: Commit the package and smoke test**

```bash
git add src/needle/__init__.py tests/test_package.py
git commit -m "test: add package smoke test"
```

## Chunk 2: Quality Gates and Developer Entry Point

### Task 3: Add repository hygiene and local quality checks

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/

# Environments and tools
.venv/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/

# Editors and operating systems
.idea/
.vscode/
.DS_Store
Thumbs.db

# Local secrets
.env
.env.*
!.env.example

# Research artifacts
data/
artifacts/
checkpoints/
outputs/
wandb/
```

- [ ] **Step 2: Run linting**

Run: `uv run ruff check .`

Expected: `All checks passed!`.

- [ ] **Step 3: Verify formatting**

Run: `uv run ruff format --check .`

Expected: exit code 0 and Ruff reports that both Python files are already formatted.

- [ ] **Step 4: Run the complete test suite**

Run: `uv run pytest`

Expected: exit code 0 with `1 passed`.

- [ ] **Step 5: Commit repository hygiene**

```bash
git add .gitignore
git commit -m "chore: add Python repository hygiene"
```

### Task 4: Add continuous integration

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v8
        with:
          enable-cache: true
      - name: Install Python
        run: uv python install 3.11
      - name: Sync dependencies
        run: uv sync --locked --dev
      - name: Lint
        run: uv run ruff check .
      - name: Check formatting
        run: uv run ruff format --check .
      - name: Test
        run: uv run pytest
```

- [ ] **Step 2: Re-run all local quality gates**

Run: `uv run ruff check .`

Expected: `All checks passed!`.

Run: `uv run ruff format --check .`

Expected: exit code 0 and Ruff reports that both Python files are already formatted.

Run: `uv run pytest`

Expected: exit code 0 with `1 passed`.

- [ ] **Step 3: Commit continuous integration**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: verify Python project"
```

- [ ] **Step 4: Push and validate the workflow on GitHub Actions**

Run: `gh --version`

Expected: exit code 0 and an installed GitHub CLI version is printed.

Run: `gh auth status`

Expected: exit code 0 and authentication to `github.com` is active for an account allowed to read Actions runs in `emrsyah/needle`.

Capture the exact commit before pushing:

```powershell
$commitSha = git rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or -not $commitSha) { throw "Could not resolve HEAD." }
git push origin main
if ($LASTEXITCODE -ne 0) { throw "Push failed." }
```

Expected: exit code 0 and `origin/main` advances to the CI commit.

Find the workflow run for that exact commit, allowing GitHub time to enqueue it:

```powershell
$runId = $null
foreach ($attempt in 1..20) {
    $runId = gh run list --commit $commitSha --workflow ci.yml --limit 1 --json databaseId --jq '.[0].databaseId'
    if ($LASTEXITCODE -ne 0) { throw "Could not query workflow runs." }
    if ($runId) { break }
    Start-Sleep -Seconds 3
}
if (-not $runId) { throw "No CI run appeared for commit $commitSha." }
gh run watch $runId --exit-status
```

Expected: exit code 0 and the `quality` job concludes with `success`.

### Task 5: Document the developer workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the status-only README with setup instructions**

````markdown
# Needle

Needle is a research project for training an evidence-seeking agentic RAG system. It studies how reinforcement learning can help an agent search for relevant evidence, use that evidence in its answer, and avoid unnecessary retrieval steps.

## Status

Early research prototype. The current milestone is a deterministic HotpotQA vertical slice; model inference and reinforcement learning are not implemented yet.

## Development

Requirements:

- [uv](https://docs.astral.sh/uv/)

Set up the environment:

```bash
uv sync --dev
```

Run the quality gates:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
````

- [ ] **Step 2: Verify commands from the README**

Run all three documented quality commands.

Expected: all commands exit with code 0, Ruff reports no lint or format violations, and pytest reports `1 passed`.

- [ ] **Step 3: Stage and inspect the exact README diff**

Run: `git add README.md`

Expected: exit code 0 and only `README.md` is newly staged.

Run: `git diff --cached --check`

Expected: no whitespace errors.

Run: `git diff --cached -- README.md`

Expected: output contains only the intended status and development documentation changes.

- [ ] **Step 4: Commit developer documentation**

```bash
git commit -m "docs: add development setup"
```

- [ ] **Step 5: Confirm final repository state**

Run: `git status --short --branch`

Expected: clean `main` branch, ahead of `origin/main` by the final README commit until the final push.
