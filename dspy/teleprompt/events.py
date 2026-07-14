from dataclasses import dataclass
from typing import Any, Literal

from dspy.utils.callback import OptimizerEvent

__all__ = [
    "CandidateEvaluated",
    "CandidateProposed",
    "CandidateSelected",
    "OptimizerEvent",
    "OptimizerScore",
    "RandomSearchCandidateProposed",
]


@dataclass(frozen=True, slots=True)
class OptimizerScore:
    """A score whose unit and aggregation are explicit."""

    value: Any
    unit: str
    aggregation: str


@dataclass(frozen=True, slots=True)
class CandidateProposed(OptimizerEvent):
    """A logical optimizer candidate that is about to be constructed.

    Candidate IDs and indexes are scoped to one optimizer invocation.
    """

    candidate_id: str
    candidate_index: int


@dataclass(frozen=True, slots=True)
class RandomSearchCandidateProposed(CandidateProposed):
    """RandomSearch-native details for a proposed candidate."""

    seed: int
    kind: Literal["zero_shot", "labeled_few_shot", "unshuffled_bootstrap", "shuffled_bootstrap"]


@dataclass(frozen=True, slots=True)
class CandidateEvaluated(OptimizerEvent):
    """The scores produced by one successful candidate evaluation."""

    candidate_id: str
    candidate_index: int
    evaluation_id: str
    metric_key: str
    score: OptimizerScore
    example_scores: tuple[OptimizerScore, ...]


@dataclass(frozen=True, slots=True)
class CandidateSelected(OptimizerEvent):
    """The candidate returned by an optimizer run."""

    candidate_id: str
    candidate_index: int
    score: OptimizerScore
