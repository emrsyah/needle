# Search Environment Design

## Goal

Wrap the deterministic BM25 retriever in a small stateful search session that later model and RL adapters can drive.

## Approach

Three shapes were considered: a stateless search function, a small stateful session, and a full Gym/OpenEnv environment. The stateful session is chosen because it records trajectory and enforces a search budget without introducing framework dependencies or premature answer/reward semantics.

## API

```python
environment = SearchEnvironment(example, top_k=2, max_searches=4)
question = environment.reset()
observation = environment.search("Ursula Le Guin birthplace")
```

`QuestionObservation` contains the question ID, question text, and search budget. `SearchObservation` contains a one-based turn number, normalized stored query, immutable BM25 results, and remaining search count.

## State and Validation

- `reset()` clears history and starts a new session.
- Searching before reset raises `SearchEnvironmentError`.
- Successful searches consume one budget unit and append an immutable observation.
- Invalid queries do not consume budget or alter history.
- Searching after budget exhaustion raises `SearchEnvironmentError`.
- Repeated queries remain allowed and are recorded; duplicate penalties belong in the future reward layer.
- `top_k` and `max_searches` must be positive integers and reject booleans.
- Public history is exposed as an immutable tuple.

## Scope

This milestone has no answer action, citations, termination reward, duplicate penalty, OpenEnv adapter, model inference, or RL logic.
