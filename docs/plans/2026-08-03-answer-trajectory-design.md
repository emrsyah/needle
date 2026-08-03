# Answer Trajectory Design

## Goal

Extend `SearchEnvironment` with a terminal answer action, corpus citations, and an immutable completed trajectory. Reward calculation and LLM action parsing remain out of scope.

## Chosen approach

Keep the existing environment API and add two frozen, slotted records:

- `AnswerObservation`: stripped answer text, a tuple of `EvidenceRef` citations, and the number of searches used.
- `EpisodeTrajectory`: the reset question observation, every search observation, and the final answer observation.

This is smaller and easier to evaluate than introducing a generic action protocol now. Citations are validated against the corpus but are not required to have appeared in retrieved results. That distinction is intentional: a later reward component can measure whether cited evidence was actually discovered and used.

## Environment contract

`answer(answer: str, citations: Sequence[EvidenceRef] = ()) -> AnswerObservation` terminates an active episode. Answering is allowed before any search and after the search budget is exhausted. Once terminated, another `search` or `answer` raises `SearchEnvironmentError`. Calling `reset` starts a fresh episode and clears the prior answer, trajectory, and search history.

The environment exposes:

- `terminated: bool`
- `answer_observation: AnswerObservation | None`
- `trajectory: EpisodeTrajectory | None`, which is `None` until a valid answer terminates the episode

The existing `history` property remains search-only for compatibility.

## Validation and mutation safety

The answer must be a string whose stripped value is non-empty. Citations are converted to a tuple; strings and values containing non-`EvidenceRef` members are rejected. Duplicate citations are rejected. Every citation must exactly match a corpus document title and use an in-range sentence index.

Any invalid answer or citation raises `SearchEnvironmentError` without consuming budget, changing search history, terminating the episode, or creating a trajectory.

## Tests

Cover answer-before-reset, zero-search answers, valid retrieved and unretrieved citations, empty or malformed answers, invalid and duplicate citations, post-termination rejection, answer-after-budget-exhaustion, immutable trajectory snapshots, and reset behavior. Run focused environment tests, the full suite, and Ruff checks.
