from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import json

from healthagent.components.utility_evolver import UtilityEvolver, UtilityEvolverRun
from healthagent.memory import BaseMemoryStore
from healthagent.schemas import PostPackage
from healthagent.utils.debug_sink import DebugSink

from .episode_runner import EpisodeRun, EpisodeRunner


@dataclass(slots=True)
class SelfEvolvingRun:
    episode_run: EpisodeRun
    evolver_run: UtilityEvolverRun


class SelfEvolvingRunner:
    """
    Decoupled self-evolving orchestration.

    Stage 1: run_episode(...)
        - runs the expensive episode pipeline once
        - returns EpisodeRun

    Stage 2: evolve_episode(...)
        - takes an existing EpisodeRun
        - runs mode judge + routed evolver
        - writes new memory to the shared memory store

    This allows the self-evolving stage to be retried without rerunning the
    episode pipeline.
    """

    def __init__(
        self,
        *,
        episode_runner: EpisodeRunner,
        utility_evolver: UtilityEvolver,
        memory_store: BaseMemoryStore,
        stream_print: bool = False,
        stream_printer: Optional[Callable[[str], None]] = None,
        debug_sink: DebugSink | None = None,
        sample_key: str | None = None,
    ) -> None:
        self.episode_runner = episode_runner
        self.utility_evolver = utility_evolver
        self.memory_store = memory_store
        self.stream_print = stream_print
        self.stream_printer = stream_printer or print

        self.debug_sink = debug_sink
        self.sample_key = sample_key

    def _format_emit_body(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _emit(self, title: str, payload: Any) -> None:
        body = self._format_emit_body(payload)

        if self.debug_sink is not None:
            self.debug_sink.emit(
                title=title,
                payload=payload,   # Original structured data, suitable for saving in JSONL format
                rendered=body,     # Formatted text, convenient for saving as txt
                sample_key=self.sample_key,
                stage="episode",
            )

        if not self.stream_print:
            return

        self.stream_printer(f"\n===== {title} =====\n{body}\n")

    # -------------------------
    # Stage 1: episode only
    # -------------------------
    def run_episode(
        self,
        post: PostPackage,
        *,
        created_at_ms: Optional[int] = None,
    ) -> EpisodeRun:
        return self.episode_runner.run(
            post=post,
            created_at_ms=created_at_ms,
        )

    # -------------------------
    # Stage 2: evolve only
    # -------------------------
    def evolve_episode(
        self,
        *,
        post: PostPackage,
        episode_run: EpisodeRun,
        write_memory: bool = True,
    ) -> UtilityEvolverRun:
        self._emit("EVOLVER INPUT MODE", "running utility evolver from existing episode")

        evolver_run = self.utility_evolver.evolve_run(
            post=post,
            episode_run=episode_run,
        )

        self._emit("MODE JUDGE OUTPUT", evolver_run.mode_judge_run.output.model_dump())
        self._emit("MODE", evolver_run.mode)
        self._emit("ROUTED EVOLVER OUTPUT", evolver_run.routed_output.model_dump())

        if write_memory:
            self.write_memory(evolver_run)

        return evolver_run

    def write_memory(
        self,
        evolver_run: UtilityEvolverRun,
    ) -> None:
        if evolver_run.planner_records:
            self.memory_store.add_records(evolver_run.planner_records)
            self._emit(
                "WRITTEN PLANNER MEMORY",
                [record.model_dump() for record in evolver_run.planner_records],
            )

        if evolver_run.actor_records:
            self.memory_store.add_records(evolver_run.actor_records)
            self._emit(
                "WRITTEN ACTOR MEMORY",
                [record.model_dump() for record in evolver_run.actor_records],
            )

    # -------------------------
    # Convenience wrapper
    # -------------------------
    def run(
        self,
        post: PostPackage,
        *,
        created_at_ms: Optional[int] = None,
        write_memory: bool = True,
    ) -> SelfEvolvingRun:
        """
        Convenience method for the full pipeline:
        run episode first, then evolve from the resulting EpisodeRun.

        This still uses the decoupled internals, so callers can choose to run
        the two stages separately when needed.
        """
        episode_run = self.run_episode(
            post=post,
            created_at_ms=created_at_ms,
        )
        evolver_run = self.evolve_episode(
            post=post,
            episode_run=episode_run,
            write_memory=write_memory,
        )

        return SelfEvolvingRun(
            episode_run=episode_run,
            evolver_run=evolver_run,
        )