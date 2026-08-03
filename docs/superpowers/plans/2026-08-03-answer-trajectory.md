# Answer Trajectory Implementation Plan

> **For agentic workers:** Implement this plan in one focused feature branch. Keep the two commits described below and finish with a clean worktree.

**Goal:** Add a terminal answer action with validated corpus citations and an immutable completed episode trajectory.

**Architecture:** Extend the existing `SearchEnvironment` state machine without introducing reward logic or an LLM action parser. Store immutable answer and trajectory snapshots while retaining the existing search-only history API.

**Tech Stack:** Python, pytest, Ruff

---

## Chunk 1: Answer and trajectory lifecycle

### Task 1: Commit the approved design

**Files:**
- Create: `docs/plans/2026-08-03-answer-trajectory-design.md`
- Create: `docs/superpowers/plans/2026-08-03-answer-trajectory.md`

- [ ] Copy this plan and its design document into the repository.
- [ ] Commit as `docs: define answer trajectory`.

### Task 2: Implement and verify the feature

**Files:**
- Modify: `src/needle/environment/search.py`
- Modify: `src/needle/environment/__init__.py`
- Create: `tests/environment/test_answer.py`

- [ ] Add failing tests for lifecycle, citation validation, mutation safety, immutable snapshots, and reset behavior.
- [ ] Run `pytest tests/environment/test_answer.py -q` and confirm the new tests fail for the missing API.
- [ ] Add frozen, slotted `AnswerObservation` and `EpisodeTrajectory` records.
- [ ] Add `terminated`, `answer_observation`, `trajectory`, and `answer(...)` to `SearchEnvironment` according to the design.
- [ ] Preserve the search-only meaning of `history`; reject search or answer after termination.
- [ ] Export the two new public records from `needle.environment`.
- [ ] Run `pytest tests/environment -q` and the full `pytest -q` suite.
- [ ] Run the repository's Ruff check and format-check commands.
- [ ] Inspect the final diff for scope creep and commit as `feat: add answer trajectory`.
