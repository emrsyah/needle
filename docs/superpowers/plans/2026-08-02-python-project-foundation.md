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

Expected: exit code 0 and Ruff reports all checked files are already formatted.

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

Expected: exit code 0 and Ruff reports all checked files are already formatted.

Run: `uv run pytest`

Expected: exit code 0 with `1 passed`.

- [ ] **Step 3: Commit continuous integration**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: verify Python project"
```

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

Expected: clean `chore/python-foundation` branch, ahead of `origin/main` by the completed foundation commits until integration.

### Task 6: Integrate the reviewed branch and validate CI remotely

- [ ] **Step 1: Verify the reviewed feature branch is clean and all local quality gates pass**

Run: `git status --short --branch`

Expected: clean `chore/python-foundation` branch with no staged or unstaged changes.

Run: `uv run ruff check .`

Expected: `All checks passed!`.

Run: `uv run ruff format --check .`

Expected: exit code 0 and Ruff reports all checked files are already formatted.

Run: `uv run pytest`

Expected: exit code 0 with `1 passed`.

- [ ] **Step 2: Record the exact reviewed feature commit**

```powershell
$featureSha = git rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or -not $featureSha) { throw "Could not resolve feature HEAD." }
```

Expected: `$featureSha` contains the full commit SHA of the reviewed `chore/python-foundation` branch.

- [ ] **Step 3: Switch to `main` and require its worktree to be clean**

```powershell
git switch main
if ($LASTEXITCODE -ne 0) { throw "Could not switch to main." }
$mainStatus = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Could not inspect main worktree." }
if ($mainStatus) { throw "Main worktree is not clean." }
```

Expected: the current branch is `main` and its worktree has no staged, unstaged, or untracked changes.

- [ ] **Step 4: Fast-forward `main` to the reviewed feature commit and verify the exact result**

```powershell
git merge --ff-only chore/python-foundation
if ($LASTEXITCODE -ne 0) { throw "Fast-forward merge failed." }
$mergedSha = git rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or -not $mergedSha) { throw "Could not resolve merged HEAD." }
if ($mergedSha -ne $featureSha) { throw "Merged HEAD $mergedSha does not equal reviewed feature HEAD $featureSha." }
```

Expected: `main` advances by fast-forward only and its HEAD exactly equals `$featureSha`.

- [ ] **Step 5: Preflight GitHub CLI availability and authentication**

Run: `gh --version`

Expected: exit code 0 and an installed GitHub CLI version is printed.

Run: `gh auth status`

Expected: exit code 0 and authentication to `github.com` is active for an account allowed to read Actions runs in `emrsyah/needle`.

- [ ] **Step 6: Push the reviewed commit to `origin/main`**

```powershell
git push origin main
if ($LASTEXITCODE -ne 0) { throw "Push failed." }
```

Expected: exit code 0 and `origin/main` advances to `$featureSha`.

- [ ] **Step 7: Locate the `ci.yml` run for the exact commit with bounded retry**

```powershell
$runId = $null
foreach ($attempt in 1..20) {
    $runJson = gh run list --commit $featureSha --event push --branch main --workflow ci.yml --limit 20 --json databaseId,headSha,event,headBranch
    if ($LASTEXITCODE -ne 0) { throw "Could not query workflow runs." }
    try {
        $runs = @($runJson | ConvertFrom-Json)
    } catch {
        throw "Could not parse workflow run response."
    }
    $matchingRuns = @(
        $runs | Where-Object {
            $_.headSha -eq $featureSha -and
            $_.event -eq "push" -and
            $_.headBranch -eq "main"
        }
    )
    if ($matchingRuns.Count -gt 0) {
        $run = $matchingRuns | Select-Object -First 1
        if (-not $run.databaseId) { throw "Matching workflow run has no database ID." }
        if ($run.headSha -ne $featureSha -or $run.event -ne "push" -or $run.headBranch -ne "main") {
            throw "Selected workflow run does not match the reviewed main push."
        }
        $runId = $run.databaseId
        break
    }
    Start-Sleep -Seconds 3
}
if (-not $runId) { throw "No main push CI run appeared for commit $featureSha." }
```

Expected: `$runId` contains the database ID of the `ci.yml` push run whose `headSha` is exactly `$featureSha` and whose `headBranch` is `main`.

- [ ] **Step 8: Watch the exact workflow run to completion**

```powershell
gh run watch $runId --exit-status
if ($LASTEXITCODE -ne 0) { throw "CI run $runId failed." }
```

Expected: exit code 0 and the `quality` job concludes with `success`.

- [ ] **Step 9: Verify remote synchronization and the final clean state**

```powershell
$qualityConclusion = gh run view $runId --json jobs --jq '.jobs[] | select(.name == "quality") | .conclusion'
if ($LASTEXITCODE -ne 0) { throw "Could not inspect CI job conclusion." }
if ($qualityConclusion -ne "success") { throw "Quality job did not succeed." }
$localSha = git rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or -not $localSha) { throw "Could not resolve local HEAD." }
$remoteSha = git rev-parse origin/main
if ($LASTEXITCODE -ne 0 -or -not $remoteSha) { throw "Could not resolve origin/main." }
if ($localSha -ne $featureSha -or $remoteSha -ne $featureSha) { throw "Local main, origin/main, and reviewed feature HEAD are not synchronized." }
$finalStatus = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Could not inspect final worktree." }
if ($finalStatus) { throw "Final worktree is not clean." }
```

Expected: the `quality` job succeeded, local `main` and `origin/main` both equal `$featureSha`, and the worktree is clean.
