from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import traceback
import shutil
import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from datetime import date, datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from healthagent.components.actor import Actor
from healthagent.components.history_builder import HistoryBuilder
from healthagent.components.memory_query_builder import MemoryQueryBuilder
from healthagent.components.rubric_planner import RubricPlanner
from healthagent.components.utility_evolver import UtilityEvolver
from healthagent.components.utility_mode_judge import UtilityModeJudge
from healthagent.memory import InMemoryStore, MemoryRecord, SimpleActorMemory, SimplePlannerMemory
from healthagent.pipeline import EpisodeRunner, SelfEvolvingRunner
from healthagent.schemas import PostPackage
from healthagent.tools.crawl_engine import FirecrawlCrawlEngine
from healthagent.tools.search_engine import SerperSearchEngine
from healthagent.tools.summary_model import GoalConditionedSummaryModel
from healthagent.utils.debug_sink import DebugSink
from healthagent.tools.llm_client import VLLMChatClient


FIRECRAWL_API_KEY = ""
SERPER_API_KEY = ""
OPENROUTER_API_KEY = ""
VLLM_BASE_URL = "http://localhost:8002/v1"
VLLM_API_KEY = ""
JINA_API_KEY = ""

IDX_START = 11
IDX_END = 12

PAIR_SIZE = 2


# ============================================================
# 0) Config dataclasses
# ============================================================

@dataclass(slots=True)
class DebugPaths:
    dataset_path: str
    memory_updates_path: str
    final_results_path: str
    traces_path: str

    # per-sample emit/debug transcript outputs
    events_text_dir: str
    events_jsonl_dir: str


@dataclass(slots=True)
class SampleContext:
    idx: int
    raw_record: dict[str, Any]
    tweet_id: str
    note_id: str
    sample_key: str
    post: PostPackage


@dataclass(slots=True)
class RuntimeBundle:
    llm: Any
    memory_store: InMemoryStore
    episode_runner: EpisodeRunner
    self_evolving_runner: SelfEvolvingRunner
    paths: DebugPaths


@dataclass(slots=True)
class EpisodeArtifact:
    sample: SampleContext
    episode_run: Any  # EpisodeRun


@dataclass(slots=True)
class SelfEvolvingArtifact:
    sample: SampleContext
    episode_run: Any  # EpisodeRun
    evolver_run: Any  # UtilityEvolverRun


@dataclass(slots=True)
class WriteArtifact:
    sample_key: str
    memory_written: int
    result_written: bool
    trace_written: bool


@dataclass(slots=True)
class SuccessfulSampleRun:
    idx: int
    runtime: RuntimeBundle
    episode_artifact: EpisodeArtifact
    evolver_artifact: SelfEvolvingArtifact


# ============================================================
# 1) Path builder
# ============================================================

def build_paths_for_idx(idx: int) -> DebugPaths:
    """
    每个 idx 单独构造 DebugPaths。

    注意：
    - dataset_path / memory_updates_path / final_results_path / traces_path 通常是共享的
    - events_text_dir / events_jsonl_dir 按 idx 隔离，避免并行写 event 冲突
    """

    base_dir = "/home/jiaying/zihang/HealthAgent2/HealthAgent/debug_run3"

    return DebugPaths(
        dataset_path="/home/jiaying/zihang/HealthAgent2/HealthAgent/YOUR_DATASET.jsonl",

        memory_updates_path=f"{base_dir}/memory_updates.jsonl",
        final_results_path=f"{base_dir}/final_results.jsonl",
        traces_path=f"{base_dir}/traces.jsonl",

        events_text_dir=f"{base_dir}/debug_outputs/events_{idx}/text",
        events_jsonl_dir=f"{base_dir}/debug_outputs/events_{idx}/jsonl",
    )


# ============================================================
# 2) Small IO helpers
# ============================================================

def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    _ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(obj), ensure_ascii=False))
        f.write("\n")


def read_jsonl_record(path: str | Path, idx: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"idx={idx} is out of range for {path}")


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_caption_field(value: Any) -> dict[str, str]:
    if value is None:
        return {}

    if isinstance(value, str):
        value = _coerce_str(value).strip()
        return {"caption0": value} if value else {}

    if not isinstance(value, dict):
        raise TypeError("caption must be either a dict[str, str], a string, or null.")

    normalized: dict[str, str] = {}
    for k, v in value.items():
        key = _coerce_str(k).strip()
        val = _coerce_str(v).strip()
        if key and val:
            normalized[key] = val

    return normalized


def build_sample_event_paths(paths: DebugPaths, sample_key: str) -> tuple[str, str]:
    text_path = str(Path(paths.events_text_dir) / f"{sample_key}.events.txt")
    jsonl_path = str(Path(paths.events_jsonl_dir) / f"{sample_key}.events.jsonl")
    return text_path, jsonl_path


# ============================================================
# 3) Dataset -> SampleContext
# ============================================================

def build_sample_from_dataset(
    dataset_path: str | Path,
    idx: int,
) -> SampleContext:
    raw = read_jsonl_record(dataset_path, idx)

    tweet_id = _coerce_str(raw.get("tweetId"))
    note_id = _coerce_str(raw.get("noteId"))
    tweet_text = _coerce_str(raw.get("tweetText"))
    caption = _normalize_caption_field(raw.get("caption"))

    if not tweet_id:
        raise ValueError("Dataset record is missing 'tweetId'.")
    if not note_id:
        raise ValueError("Dataset record is missing 'noteId'.")
    if not tweet_text and not caption:
        raise ValueError("Dataset record must contain at least one of 'tweetText' or 'caption'.")

    post = PostPackage(
        post_id=tweet_id,
        tweet_text=tweet_text,
        caption=caption,
    )

    sample_key = f"{tweet_id}::{note_id}"

    return SampleContext(
        idx=idx,
        raw_record=raw,
        tweet_id=tweet_id,
        note_id=note_id,
        sample_key=sample_key,
        post=post,
    )


# ============================================================
# 4) Memory preload
# ============================================================

def load_memory_store_from_updates(memory_updates_path: str | Path) -> InMemoryStore:
    records: list[MemoryRecord] = []
    path = Path(memory_updates_path)

    if not path.exists():
        return InMemoryStore(records=[])

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # grouped audit record with planner_records / actor_records
            if "planner_records" in obj or "actor_records" in obj:
                for item in obj.get("planner_records", []):
                    records.append(MemoryRecord.model_validate(item))
                for item in obj.get("actor_records", []):
                    records.append(MemoryRecord.model_validate(item))
                continue

            # direct one-record-per-line fallback
            if {"scope", "trigger", "rule", "why"} <= set(obj.keys()):
                records.append(MemoryRecord.model_validate(obj))
                continue

    return InMemoryStore(records=records)


# ============================================================
# 5) Runtime assembly
# ============================================================

def build_runtime(
    *,
    llm: Any,
    paths: DebugPaths,
    sample_key: str,
    global_rubrics: list[str],
    stream_print: bool = True,
    max_steps: int = 15,
    max_searches: int = 5,
    max_visits: int = 12,
    min_searches_before_finalize: int = 1,
    min_visits_before_finalize: int = 4,
    max_text_chars: int = 270,
    max_platform_chars: int = 280,
    max_write_refinement_attempts: int = 5,
) -> RuntimeBundle:
    # 1) Load historical memory from the audit file
    memory_store = load_memory_store_from_updates(paths.memory_updates_path)

    # 2) Retrieval wrappers over the same in-memory store
    planner_memory_retriever = SimplePlannerMemory(
        memory_store,
        top_k=4,
    )
    actor_memory_retriever = SimpleActorMemory(
        memory_store,
        top_k=4,
    )

    # 3) Query builder uses the same llm
    memory_query_builder = MemoryQueryBuilder(
        chat_client=llm,
    )

    # 4) Main components use the same llm
    planner = RubricPlanner(
        chat_client=llm,
    )
    actor = Actor(
        chat_client=llm,
    )
    history_builder = HistoryBuilder()

    summary_model = GoalConditionedSummaryModel(
        chat_client=llm,
    )

    mode_judge = UtilityModeJudge(
        chat_client=llm,
    )
    utility_evolver = UtilityEvolver(
        mode_judge=mode_judge,
        chat_client=llm,
    )

    # 5) Tool backends
    search_engine = SerperSearchEngine(
        api_key=SERPER_API_KEY,
    )
    crawl_engine = FirecrawlCrawlEngine(
        api_key=FIRECRAWL_API_KEY,
        jina_api_key=JINA_API_KEY,
    )

    # 6) Per-sample debug sink for emit transcripts
    events_text_path, events_jsonl_path = build_sample_event_paths(paths, sample_key)

    debug_sink = DebugSink(
        jsonl_path=events_jsonl_path,
        text_path=events_text_path,
        print_to_console=stream_print,
    )

    # 7) Episode runner
    episode_runner = EpisodeRunner(
        planner=planner,
        actor=actor,
        history_builder=history_builder,
        search_engine=search_engine,
        crawl_engine=crawl_engine,
        summary_model=summary_model,
        global_rubrics=global_rubrics,
        max_steps=max_steps,
        max_searches=max_searches,
        max_visits=max_visits,
        min_searches_before_finalize=min_searches_before_finalize,
        min_visits_before_finalize=min_visits_before_finalize,
        max_text_chars=max_text_chars,
        max_platform_chars=max_platform_chars,
        max_write_refinement_attempts=max_write_refinement_attempts,
        planner_memory_retriever=planner_memory_retriever,
        actor_memory_retriever=actor_memory_retriever,
        memory_query_builder=memory_query_builder,
        stream_print=stream_print,
        debug_sink=debug_sink,
        sample_key=sample_key,
    )

    # 8) Decoupled self-evolving runner
    self_evolving_runner = SelfEvolvingRunner(
        episode_runner=episode_runner,
        utility_evolver=utility_evolver,
        memory_store=memory_store,
        stream_print=stream_print,
        debug_sink=debug_sink,
        sample_key=sample_key,
    )

    return RuntimeBundle(
        llm=llm,
        memory_store=memory_store,
        episode_runner=episode_runner,
        self_evolving_runner=self_evolving_runner,
        paths=paths,
    )


# ============================================================
# 6) Serialization helpers
# ============================================================

def serialize_episode_run(episode_run: Any) -> dict[str, Any]:
    return {
        "result": episode_run.result.model_dump(),
        "trace": episode_run.trace.model_dump(),
        "final_history": episode_run.final_history.model_dump(),
    }


def serialize_evolver_run(evolver_run: Any) -> dict[str, Any]:
    return {
        "mode": evolver_run.mode,
        "mode_judge": {
            "output": evolver_run.mode_judge_run.output.model_dump(),
            "prompt": evolver_run.mode_judge_run.prompt,
            "raw_output": evolver_run.mode_judge_run.raw_output,
        },
        "routed_output": evolver_run.routed_output.model_dump(),
        "planner_records": [
            record.model_dump(mode="json")
            for record in evolver_run.planner_records
        ],
        "actor_records": [
            record.model_dump(mode="json")
            for record in evolver_run.actor_records
        ],
        "prompt": evolver_run.prompt,
        "raw_output": evolver_run.raw_output,
    }


# ============================================================
# 7) Module 1: episode only
# ============================================================

def run_episode_module(
    runtime: RuntimeBundle,
    sample: SampleContext,
    *,
    created_at_ms: Optional[int] = None,
) -> EpisodeArtifact:
    episode_run = runtime.self_evolving_runner.run_episode(
        post=sample.post,
        created_at_ms=created_at_ms,
    )

    return EpisodeArtifact(
        sample=sample,
        episode_run=episode_run,
    )


# ============================================================
# 8) Module 2: self-evolving only
# ============================================================

def run_self_evolving_module(
    runtime: RuntimeBundle,
    episode_artifact: EpisodeArtifact,
) -> SelfEvolvingArtifact:
    evolver_run = runtime.self_evolving_runner.evolve_episode(
        post=episode_artifact.sample.post,
        episode_run=episode_artifact.episode_run,
        write_memory=False,
    )

    return SelfEvolvingArtifact(
        sample=episode_artifact.sample,
        episode_run=episode_artifact.episode_run,
        evolver_run=evolver_run,
    )


# ============================================================
# 9) Module 3: write outputs
# ============================================================

def write_outputs_module(
    runtime: RuntimeBundle,
    episode_artifact: EpisodeArtifact,
    evolver_artifact: Optional[SelfEvolvingArtifact] = None,
) -> WriteArtifact:
    sample = episode_artifact.sample
    episode_run = episode_artifact.episode_run

    # --------------------------------------------------------
    # A) memory updates
    # --------------------------------------------------------
    memory_written = 0

    planner_records: list[dict[str, Any]] = []
    actor_records: list[dict[str, Any]] = []
    evolver_payload: Optional[dict[str, Any]] = None

    if evolver_artifact is not None:
        evolver_run = evolver_artifact.evolver_run

        # update in-memory store for this runtime only
        if evolver_run.planner_records:
            runtime.memory_store.add_records(evolver_run.planner_records)
        if evolver_run.actor_records:
            runtime.memory_store.add_records(evolver_run.actor_records)

        planner_records = [
            record.model_dump(mode="json")
            for record in evolver_run.planner_records
        ]
        actor_records = [
            record.model_dump(mode="json")
            for record in evolver_run.actor_records
        ]

        memory_written = len(planner_records) + len(actor_records)

        memory_obj = {
            "sample_key": sample.sample_key,
            "idx": sample.idx,
            "tweetId": sample.tweet_id,
            "noteId": sample.note_id,
            "post_id": sample.post.post_id,
            "mode": evolver_run.mode,
            "mode_judge_output": evolver_run.mode_judge_run.output.model_dump(),
            "planner_records": planner_records,
            "actor_records": actor_records,
        }

        append_jsonl(runtime.paths.memory_updates_path, memory_obj)
        evolver_payload = serialize_evolver_run(evolver_run)

    # --------------------------------------------------------
    # B) final result
    # --------------------------------------------------------
    result_obj = {
        "sample_key": sample.sample_key,
        "idx": sample.idx,
        "tweetId": sample.tweet_id,
        "noteId": sample.note_id,
        "post_id": sample.post.post_id,
        "result": episode_run.result.model_dump(),
    }

    append_jsonl(runtime.paths.final_results_path, result_obj)

    # --------------------------------------------------------
    # C) full trace
    # --------------------------------------------------------
    trace_obj = {
        "sample_key": sample.sample_key,
        "idx": sample.idx,
        "tweetId": sample.tweet_id,
        "noteId": sample.note_id,
        "post_id": sample.post.post_id,
        "episode_run": serialize_episode_run(episode_run),
        "evolver_run": evolver_payload,
    }

    append_jsonl(runtime.paths.traces_path, trace_obj)

    return WriteArtifact(
        sample_key=sample.sample_key,
        memory_written=memory_written,
        result_written=True,
        trace_written=True,
    )


# ============================================================
# 10) Convenience loader
# ============================================================

def prepare_one_sample(
    *,
    dataset_path: str,
    idx: int,
) -> SampleContext:
    return build_sample_from_dataset(
        dataset_path=dataset_path,
        idx=idx,
    )


# ============================================================
# 11) Event output reset
# ============================================================

def reset_event_outputs(paths: DebugPaths) -> None:
    """
    清空当前 idx 的 event debug 输出目录。

    因为 events_text_dir / events_jsonl_dir 已经依赖 idx，
    所以并行时不同样本不会互删。
    """

    dir_paths = {
        paths.events_text_dir,
        paths.events_jsonl_dir,
    }

    for dir_path in dir_paths:
        p = Path(dir_path)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


# ============================================================
# 12) Retry helper
# ============================================================

def run_with_retry(func, max_retries, *args, sleep_seconds=0, **kwargs):
    idx = kwargs.get("idx", None)

    prefix = f"idx={idx} " if idx is not None else ""

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            print(f"{prefix}开始运行，第 {attempt + 1} 次")
            return func(*args, **kwargs)

        except Exception as e:
            last_error = e
            print(f"{prefix}第 {attempt + 1} 次运行失败：{repr(e)}")
            traceback.print_exc()

            if attempt >= max_retries:
                print(f"{prefix}已达到最大重试次数 {max_retries}，不再重试")
                raise last_error

            if sleep_seconds > 0:
                print(f"{prefix}{sleep_seconds} 秒后重试")
                time.sleep(sleep_seconds)


# ============================================================
# 13) Single sample run without writing outputs
# ============================================================

def run_one_sample_no_write(*, idx: int) -> SuccessfulSampleRun:
    """
    单条样本运行函数。

    重要：
    这里只运行：
    1. episode
    2. self-evolving

    不调用 write_outputs_module。
    因此不会立刻写 memory_updates / final_results / traces。
    """

    llm = VLLMChatClient(
        base_url=VLLM_BASE_URL,
        default_model="google/medgemma-27b-text-it",
    )

    paths = build_paths_for_idx(idx)

    # 每个 idx 的 event 路径独立，因此这里可以安全清空
    reset_event_outputs(paths)

    global_rubrics = [
        "Do not repeat substantially similar searches.",
    ]

    sample = prepare_one_sample(
        dataset_path=paths.dataset_path,
        idx=idx,
    )

    runtime = build_runtime(
        llm=llm,
        paths=paths,
        sample_key=sample.sample_key,
        global_rubrics=global_rubrics,
        stream_print=False,
    )

    # 1) episode_runner module
    episode_artifact = run_episode_module(
        runtime,
        sample,
    )

    # 2) self-evolving module
    evolver_artifact = run_self_evolving_module(
        runtime,
        episode_artifact,
    )

    return SuccessfulSampleRun(
        idx=idx,
        runtime=runtime,
        episode_artifact=episode_artifact,
        evolver_artifact=evolver_artifact,
    )


# ============================================================
# 14) Parallel pair runner
# ============================================================

def run_pair_parallel(
    indices: list[int],
    *,
    max_retries: int = 5,
    sleep_seconds: int = 3,
) -> tuple[list[SuccessfulSampleRun], list[int]]:
    """
    并行运行一组样本，默认是两个 idx。

    逻辑：
    - pair 内每个 idx 并行运行 episode + self-evolving
    - worker 内不写 memory / result / trace
    - pair 全部结束后，主线程统一写成功样本
    - 如果一个失败，只写成功的那个
    - 如果都失败，则不写任何东西
    """

    successful_runs: list[SuccessfulSampleRun] = []
    failed_indices: list[int] = []

    if not indices:
        return successful_runs, failed_indices

    with ThreadPoolExecutor(max_workers=len(indices)) as executor:
        future_to_idx = {}

        for idx in indices:
            future = executor.submit(
                run_with_retry,
                run_one_sample_no_write,
                max_retries,
                idx=idx,
                sleep_seconds=sleep_seconds,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]

            try:
                run = future.result()
                successful_runs.append(run)
            except Exception as e:
                print(f"idx={idx} 最终失败，跳过该 idx：{repr(e)}")
                failed_indices.append(idx)

    # 重要：写入顺序按 idx 排序，而不是按完成顺序
    successful_runs.sort(key=lambda x: x.idx)

    # pair 全部结束后，统一写成功样本
    for run in successful_runs:
        write_artifact = write_outputs_module(
            run.runtime,
            run.episode_artifact,
            run.evolver_artifact,
        )
        print(write_artifact)

    return successful_runs, failed_indices


# ============================================================
# 15) Batching helper
# ============================================================

def batched_indices(start: int, end: int, batch_size: int = 2):
    for i in range(start, end, batch_size):
        yield list(range(i, min(i + batch_size, end)))


# ============================================================
# 16) Main
# ============================================================

def main():
    failed_indices: list[int] = []

    index_batches = list(
        batched_indices(
            IDX_START,
            IDX_END,
            batch_size=PAIR_SIZE,
        )
    )

    pbar = tqdm(index_batches, desc="Running index pairs")

    for indices in pbar:
        pbar.set_postfix(indices=indices)

        successful_runs, pair_failed_indices = run_pair_parallel(
            indices,
            max_retries=5,
            sleep_seconds=3,
        )

        failed_indices.extend(pair_failed_indices)

        print(
            f"当前 pair={indices} 完成，"
            f"成功={ [run.idx for run in successful_runs] }，"
            f"失败={pair_failed_indices}"
        )

    print("失败的 idx：", failed_indices)


if __name__ == "__main__":
    main()