"""Policy-independent interactive rollout collection."""

from __future__ import annotations

from typing import Protocol

from needle.actions import (
    ActionParseError,
    AnswerAction,
    SearchAction,
    canonicalize_citation_title,
    parse_action,
)
from needle.data import EvidenceRef, QuestionExample
from needle.environment import SearchEnvironment, SearchEnvironmentError
from needle.inference import build_prompt
from needle.rewards import RewardConfig, evaluate

from .rollouts import (
    PROTOCOL_REWARD,
    ActionStep,
    FailureKind,
    PolicyResponse,
    PolicyTransportError,
    RolloutRecord,
    RolloutStatus,
)


class Policy(Protocol):
    """Minimal policy contract; local adapters own tokenization and log-probabilities."""

    def act(self, prompt_text: str, observation_text: str) -> PolicyResponse:
        """Generate exactly one raw action for the current prompt/observation."""


def _observation_text(environment: SearchEnvironment) -> str:
    if not environment.history:
        return "(none)"
    observation = environment.history[-1]
    lines = [f"Search {observation.turn}: {observation.query}"]
    for result in observation.results:
        lines.append(f"[{result.document.title}]")
        lines.extend(
            f"{index}: {sentence}" for index, sentence in enumerate(result.document.sentences)
        )
    return "\n".join(lines)


def _canonical_citations(example: QuestionExample, action: AnswerAction) -> tuple[EvidenceRef, ...]:
    corpus_titles = tuple(document.title for document in example.documents)
    return tuple(
        EvidenceRef(
            canonicalize_citation_title(citation.document_title, corpus_titles)
            or citation.document_title,
            citation.sentence_index,
        )
        for citation in action.citations
    )


class RolloutCollector:
    """Collect one fresh environment rollout with no action retries."""

    def __init__(
        self,
        policy: Policy,
        *,
        top_k: int = 3,
        max_searches: int = 3,
        max_turns: int | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(max_searches, int) or isinstance(max_searches, bool) or max_searches <= 0:
            raise ValueError("max_searches must be a positive integer")
        if max_turns is not None and (
            not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns <= 0
        ):
            raise ValueError("max_turns must be a positive integer or None")
        if reward_config is not None and not isinstance(reward_config, RewardConfig):
            raise TypeError("reward_config must be RewardConfig or None")
        self.policy = policy
        self.top_k = top_k
        self.max_searches = max_searches
        self.max_turns = max_turns
        self.reward_config = reward_config

    def _failure(
        self,
        example: QuestionExample,
        status: RolloutStatus,
        steps: list[ActionStep],
        kind: FailureKind,
        message: str,
    ) -> RolloutRecord:
        return RolloutRecord(
            question_id=example.question_id,
            status=status,
            steps=tuple(steps),
            trajectory=None,
            failure_kind=kind,
            failure_message=message,
            training_reward=None if status is RolloutStatus.TRANSPORT_ERROR else PROTOCOL_REWARD,
        )

    def collect(self, example: QuestionExample) -> RolloutRecord:
        """Run one fresh environment and return a record that can be scored offline."""
        if not isinstance(example, QuestionExample):
            raise TypeError("example must be a QuestionExample")
        environment = SearchEnvironment(
            example,
            top_k=self.top_k,
            max_searches=self.max_searches,
        )
        environment.reset()
        steps: list[ActionStep] = []
        max_turns = self.max_turns or self.max_searches + 1

        for turn in range(1, max_turns + 1):
            prompt = build_prompt(environment)
            observation = _observation_text(environment)
            response: PolicyResponse | None = None
            try:
                response = self.policy.act(prompt, observation)
                if not isinstance(response, PolicyResponse):
                    raise TypeError("policy must return PolicyResponse")
                if response.tokens is not None and response.tokens.prompt_text != prompt:
                    raise ValueError("token trace prompt_text must equal the collector prompt")
            except PolicyTransportError as error:
                steps.append(
                    ActionStep(
                        turn=turn,
                        prompt_text=prompt,
                        observation_text=observation,
                        raw_action=None,
                        validator_error=str(error),
                    )
                )
                return self._failure(
                    example,
                    RolloutStatus.TRANSPORT_ERROR,
                    steps,
                    FailureKind.TRANSPORT,
                    str(error),
                )
            except (TypeError, ValueError) as error:
                steps.append(
                    ActionStep(
                        turn=turn,
                        prompt_text=prompt,
                        observation_text=observation,
                        raw_action=response.action_text if response is not None else None,
                        model_metadata=response.model_metadata if response is not None else None,
                        tokens=response.tokens if response is not None else None,
                        validator_error=str(error),
                    )
                )
                return self._failure(
                    example,
                    RolloutStatus.INVALID,
                    steps,
                    FailureKind.PROTOCOL,
                    str(error),
                )

            step_index = len(steps)
            steps.append(
                ActionStep(
                    turn=turn,
                    prompt_text=prompt,
                    observation_text=observation,
                    raw_action=response.action_text,
                    model_metadata=response.model_metadata,
                    tokens=response.tokens,
                )
            )
            try:
                action = parse_action(response.action_text)
            except (ActionParseError, TypeError, ValueError) as error:
                steps[step_index] = ActionStep(
                    turn=turn,
                    prompt_text=prompt,
                    observation_text=observation,
                    raw_action=response.action_text,
                    model_metadata=response.model_metadata,
                    tokens=response.tokens,
                    validator_error=str(error),
                )
                return self._failure(
                    example,
                    RolloutStatus.INVALID,
                    steps,
                    FailureKind.PROTOCOL,
                    str(error),
                )

            if isinstance(action, SearchAction):
                try:
                    environment.search(action.query)
                except SearchEnvironmentError as error:
                    message = str(error)
                    steps[step_index] = ActionStep(
                        turn=turn,
                        prompt_text=prompt,
                        observation_text=observation,
                        raw_action=response.action_text,
                        model_metadata=response.model_metadata,
                        tokens=response.tokens,
                        validator_error=message,
                    )
                    kind = (
                        FailureKind.BUDGET
                        if "exhausted" in message.lower()
                        else FailureKind.RETRIEVAL
                    )
                    return self._failure(
                        example,
                        RolloutStatus.EXHAUSTED
                        if kind is FailureKind.BUDGET
                        else RolloutStatus.INVALID,
                        steps,
                        kind,
                        message,
                    )
                continue

            try:
                citations = _canonical_citations(example, action)
                environment.answer(action.answer, citations)
            except SearchEnvironmentError as error:
                message = str(error)
                kind = FailureKind.EVIDENCE if "citation" in message.lower() else FailureKind.ANSWER
                steps[step_index] = ActionStep(
                    turn=turn,
                    prompt_text=prompt,
                    observation_text=observation,
                    raw_action=response.action_text,
                    model_metadata=response.model_metadata,
                    tokens=response.tokens,
                    validator_error=message,
                )
                return self._failure(
                    example,
                    RolloutStatus.INVALID,
                    steps,
                    kind,
                    message,
                )

            trajectory = environment.trajectory
            if trajectory is None:
                raise AssertionError("answer should create a trajectory")
            reward = evaluate(example, trajectory, self.reward_config)
            return RolloutRecord(
                question_id=example.question_id,
                status=RolloutStatus.VALID,
                steps=tuple(steps),
                trajectory=trajectory,
                failure_kind=None,
                failure_message=None,
                training_reward=reward.total,
                reward_breakdown=reward,
            )

        return self._failure(
            example,
            RolloutStatus.EXHAUSTED,
            steps,
            FailureKind.BUDGET,
            "model exhausted episode turns without an answer",
        )


__all__ = ["Policy", "RolloutCollector"]
