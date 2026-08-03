"""One-action-at-a-time model episode execution."""

from needle.actions import (
    ActionParseError,
    AnswerAction,
    SearchAction,
    canonicalize_citation_title,
    parse_action,
)
from needle.data import EvidenceRef
from needle.environment import EpisodeTrajectory, SearchEnvironment, SearchEnvironmentError

from .client import CompletionResult, OpenRouterClient, OpenRouterError


class EpisodeRunnerError(RuntimeError):
    """Raised when a model episode cannot produce a valid completed trajectory."""


def build_prompt(environment: SearchEnvironment, feedback: str | None = None) -> str:
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
        (
            "Copy citation titles exactly as shown in the search results, including "
            "punctuation and accents."
        ),
        "Never invent a document title; use CITATIONS[] when no citation can be validated.",
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
    if feedback is not None:
        retrieved_titles = tuple(
            dict.fromkeys(
                result.document.title
                for observation in environment.history
                for result in observation.results
            )
        )
        lines.extend(
            (
                "Validator feedback: the previous action was rejected.",
                f"{feedback}",
                "Valid citation titles from retrieved results (copy exactly):",
                *(f"- {title}" for title in retrieved_titles),
                "Use CITATIONS[] if no valid title and sentence index can be selected.",
                "Return one corrected action only.",
            )
        )
    return "\n".join(lines)


class EpisodeRunner:
    """Drive a SearchEnvironment with one completion for each model action."""

    def __init__(
        self,
        client: OpenRouterClient,
        max_turns: int | None = None,
        max_retries: int = 0,
    ) -> None:
        self.client = client
        if max_turns is not None and (
            isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0
        ):
            raise ValueError("max_turns must be a positive integer or None")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self.max_turns = max_turns
        self.max_retries = max_retries

    def run(self, environment: SearchEnvironment) -> EpisodeTrajectory:
        environment.reset()
        max_turns = (
            self.max_turns if self.max_turns is not None else environment.remaining_searches + 1
        )
        if max_turns <= 0:
            raise EpisodeRunnerError("max_turns must be positive")
        for _ in range(max_turns):
            feedback: str | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    completion: CompletionResult = self.client.complete(
                        ({"role": "user", "content": build_prompt(environment, feedback)},)
                    )
                    action = parse_action(completion.text)
                    if isinstance(action, SearchAction):
                        environment.search(action.query)
                    elif isinstance(action, AnswerAction):
                        corpus_titles = tuple(
                            document.title for document in environment.example.documents
                        )
                        canonical_citations = tuple(
                            EvidenceRef(
                                canonicalize_citation_title(citation.document_title, corpus_titles)
                                or citation.document_title,
                                citation.sentence_index,
                            )
                            for citation in action.citations
                        )
                        environment.answer(action.answer, canonical_citations)
                        trajectory = environment.trajectory
                        if trajectory is None:
                            raise AssertionError("answer should create a trajectory")
                        return trajectory
                    break
                except OpenRouterError as error:
                    raise EpisodeRunnerError(f"model action failed: {error}") from error
                except (ActionParseError, SearchEnvironmentError, TypeError, ValueError) as error:
                    if attempt == self.max_retries:
                        message = (
                            "model action failed"
                            if isinstance(error, (ActionParseError, TypeError, ValueError))
                            else "invalid environment action"
                        )
                        raise EpisodeRunnerError(f"{message}: {error}") from error
                    feedback = str(error)
                    if isinstance(error, ActionParseError):
                        feedback = (
                            f"{feedback}\n"
                            "Parser correction: emit exactly one line matching SEARCH[query] "
                            "or ANSWER[answer] CITATIONS[title|index; ...]. Do not use "
                            "Markdown, explanations, or a final period."
                        )
        raise EpisodeRunnerError("model exhausted episode turns without an answer")


__all__ = ["EpisodeRunner", "EpisodeRunnerError", "build_prompt"]
