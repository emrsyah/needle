"""One-action-at-a-time model episode execution."""

from needle.actions import ActionParseError, AnswerAction, SearchAction, parse_action
from needle.environment import EpisodeTrajectory, SearchEnvironment, SearchEnvironmentError

from .client import CompletionResult, OpenRouterClient, OpenRouterError


class EpisodeRunnerError(RuntimeError):
    """Raised when a model episode cannot produce a valid completed trajectory."""


def build_prompt(environment: SearchEnvironment) -> str:
    """Create the deterministic prompt from the question and search history."""
    lines = [
        "You are a careful evidence-seeking QA agent.",
        "Return exactly one action: SEARCH[query] or ANSWER[answer] CITATIONS[title|index; ...].",
        (
            "Example: ANSWER[United States] CITATIONS[Ursula K. Le Guin|1; "
            "The Left Hand of Darkness|1]."
        ),
        (
            "Use one title|index citation per entry; separate entries with semicolons, "
            "never comma-separated indices."
        ),
        "Do not add a period, explanation, or any text after the final closing bracket.",
        "Use only the supplied search results for citations.",
        f"Question: {environment.example.question}",
        "Search results so far:",
    ]
    if not environment.history:
        lines.append("(none)")
    for observation in environment.history:
        lines.append(f"Search {observation.turn}: {observation.query}")
        for result in observation.results:
            document = result.document
            lines.append(f"[{document.title}]")
            lines.extend(
                f"{index}: {sentence}" for index, sentence in enumerate(document.sentences)
            )
    return "\n".join(lines)


class EpisodeRunner:
    """Drive a SearchEnvironment with one completion for each model action."""

    def __init__(self, client: OpenRouterClient, max_turns: int | None = None) -> None:
        self.client = client
        self.max_turns = max_turns

    def run(self, environment: SearchEnvironment) -> EpisodeTrajectory:
        environment.reset()
        max_turns = (
            self.max_turns if self.max_turns is not None else environment.remaining_searches + 1
        )
        if max_turns <= 0:
            raise EpisodeRunnerError("max_turns must be positive")
        for _ in range(max_turns):
            try:
                completion: CompletionResult = self.client.complete(
                    ({"role": "user", "content": build_prompt(environment)},)
                )
                action = parse_action(completion.text)
            except (OpenRouterError, ActionParseError, TypeError, ValueError) as error:
                raise EpisodeRunnerError(f"model action failed: {error}") from error
            try:
                if isinstance(action, SearchAction):
                    environment.search(action.query)
                elif isinstance(action, AnswerAction):
                    environment.answer(action.answer, action.citations)
                    trajectory = environment.trajectory
                    if trajectory is None:
                        raise AssertionError("answer should create a trajectory")
                    return trajectory
            except SearchEnvironmentError as error:
                raise EpisodeRunnerError(f"invalid environment action: {error}") from error
        raise EpisodeRunnerError("model exhausted episode turns without an answer")


__all__ = ["EpisodeRunner", "EpisodeRunnerError", "build_prompt"]
