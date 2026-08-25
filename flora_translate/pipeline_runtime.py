"""Runtime controls for the shared production FlowPilot pipeline.

These controls change execution mechanics, not chemistry.  The GUI uses the
defaults; benchmark callers can freeze retrieval exclusions, council budget,
and recorder settings without maintaining a second implementation of the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from flora_translate.engine.council_v4.execution_config import CouncilExecutionConfig


@dataclass(frozen=True)
class PipelineRuntimeOptions:
    exclude_record_ids: frozenset[str] = field(default_factory=frozenset)
    retrieval_mode: str = "semantic"
    candidate_budget: int = 12
    objective_override: str | None = None
    hard_constraints: tuple[str, ...] = ()
    benchmark_recorder: Any = None
    benchmark_strict_scoring: bool = False
    benchmark_scoring_batch_size: int | None = None
    benchmark_strong_revision_mode: bool = False
    benchmark_branching_revision_mode: bool = False
    benchmark_max_descendants_per_candidate: int = 2
    benchmark_max_total_revised_candidates: int | None = None
    council_execution: CouncilExecutionConfig | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be at least 1")
        if self.retrieval_mode not in {"semantic", "lexical"}:
            raise ValueError("retrieval_mode must be 'semantic' or 'lexical'")
        if self.benchmark_max_descendants_per_candidate < 1:
            raise ValueError(
                "benchmark_max_descendants_per_candidate must be at least 1"
            )
        if self.council_execution is not None:
            object.__setattr__(
                self,
                "council_execution",
                CouncilExecutionConfig.coerce(self.council_execution),
            )

    @classmethod
    def coerce(
        cls,
        value: "PipelineRuntimeOptions | dict[str, Any] | None",
    ) -> "PipelineRuntimeOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        data = dict(value)
        data["exclude_record_ids"] = frozenset(
            str(item) for item in data.get("exclude_record_ids") or () if item
        )
        data["hard_constraints"] = tuple(
            str(item) for item in data.get("hard_constraints") or () if item
        )
        return cls(**data)

    def provenance(self) -> dict[str, Any]:
        """Serializable execution metadata stored with every result."""

        return {
            "pipeline_entry_point": "flora_translate.main.translate",
            "exclude_record_ids": sorted(self.exclude_record_ids),
            "retrieval_mode": self.retrieval_mode,
            "candidate_budget": self.candidate_budget,
            "objective_override": self.objective_override,
            "hard_constraints": list(self.hard_constraints),
            "benchmark_strict_scoring": self.benchmark_strict_scoring,
            "benchmark_scoring_batch_size": self.benchmark_scoring_batch_size,
            "benchmark_strong_revision_mode": self.benchmark_strong_revision_mode,
            "benchmark_branching_revision_mode": (
                self.benchmark_branching_revision_mode
            ),
            "benchmark_max_descendants_per_candidate": (
                self.benchmark_max_descendants_per_candidate
            ),
            "benchmark_max_total_revised_candidates": (
                self.benchmark_max_total_revised_candidates
            ),
            "council_execution": (
                self.council_execution.provenance()
                if isinstance(self.council_execution, CouncilExecutionConfig)
                else None
            ),
        }


def merged_hard_constraints(
    runtime_constraints: Iterable[str],
    intake_constraints: Any,
) -> list[str] | Any:
    """Prefer explicit benchmark constraints, preserving GUI intake behavior."""

    values = [str(item) for item in runtime_constraints if str(item).strip()]
    return values if values else intake_constraints
