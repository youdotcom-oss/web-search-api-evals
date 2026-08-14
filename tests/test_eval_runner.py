"""Tests for eval_runner resume behaviour and per-sampler problem isolation.

These tests use a stub sampler and never hit the network, so they run without
API keys.
"""

import argparse
from pathlib import Path

import pandas as pd
import pytest

from evals.configs import datasets
from evals.eval_runner import get_remaining_problems, run_evals, get_sampler_filepath


def _dataset(tmp_path: Path, problems: list[str]) -> datasets.Dataset:
    df = pd.DataFrame({"problem": problems, "answer": [f"a-{p}" for p in problems]})
    csv_path = tmp_path / "fake.csv"
    df.to_csv(csv_path, index=False)
    dataset = datasets.Dataset(
        dataset_name="fake",
        csv_path=str(csv_path),
        grader=None,
        df=df,
    )
    return dataset


class _StubSampler:
    """Stands in for a real sampler; records what it was asked to run."""

    def __init__(self, name: str, evaluation_result: str = "is_correct"):
        self.sampler_name = name
        self.max_concurrency = 5
        self.evaluation_result = evaluation_result
        self.seen: list[str] = []

    async def __call__(self, query, dataset, ground_truth="", overwrite=False):
        self.seen.append(query)
        return {
            "query": query,
            "internal_response_time_ms": 1.0,
            "request_response_time_ms": 2.0,
            "evaluation_result": self.evaluation_result,
            "generated_answer": f"answer for {query}",
            "ground_truth": ground_truth,
        }


def _write_results(results_dir: Path, dataset, sampler, rows: list[dict]):
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        get_sampler_filepath(sampler, dataset, results_dir), index=False
    )


def test_completed_problems_are_skipped(tmp_path):
    dataset = _dataset(tmp_path, ["q1", "q2", "q3"])
    sampler = _StubSampler("stub")
    results_dir = tmp_path / "results"
    _write_results(
        results_dir,
        dataset,
        sampler,
        [
            {
                "query": "q1",
                "evaluation_result": "is_correct",
                "generated_answer": "a",
            }
        ],
    )

    remaining = get_remaining_problems(dataset, sampler, results_dir)

    assert remaining["problem"].tolist() == ["q2", "q3"]


def test_failed_problems_are_retried(tmp_path):
    """A FAILED row must not count as completed, or transient errors become permanent."""
    dataset = _dataset(tmp_path, ["q1", "q2"])
    sampler = _StubSampler("stub")
    results_dir = tmp_path / "results"
    _write_results(
        results_dir,
        dataset,
        sampler,
        [
            {
                "query": "q1",
                "evaluation_result": "FAILED",
                "generated_answer": "FAILED",
            },
            {
                "query": "q2",
                "evaluation_result": "is_correct",
                "generated_answer": "a",
            },
        ],
    )

    remaining = get_remaining_problems(dataset, sampler, results_dir)

    assert remaining["problem"].tolist() == ["q1"]


def test_explicit_problems_argument_is_not_the_shared_dataframe(tmp_path):
    """Filtering must read from the passed-in frame, leaving dataset.df untouched."""
    dataset = _dataset(tmp_path, ["q1", "q2", "q3"])
    sampler = _StubSampler("stub")
    results_dir = tmp_path / "results"
    _write_results(
        results_dir,
        dataset,
        sampler,
        [{"query": "q1", "evaluation_result": "is_correct", "generated_answer": "a"}],
    )
    snapshot = dataset.df

    remaining = get_remaining_problems(
        dataset, sampler, results_dir, problems=dataset.df
    )

    assert remaining["problem"].tolist() == ["q2", "q3"]
    assert dataset.df is snapshot
    assert dataset.df["problem"].tolist() == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_second_sampler_is_not_narrowed_by_the_first(tmp_path, monkeypatch):
    """Regression: a resumed run must not shrink the problem set for later samplers.

    sampler_a has already completed q1, so it only needs q2 and q3. sampler_b has
    no prior results and must still be given all three.
    """
    dataset = _dataset(tmp_path, ["q1", "q2", "q3"])
    sampler_a = _StubSampler("sampler_a")
    sampler_b = _StubSampler("sampler_b")
    results_dir = tmp_path / "results"

    _write_results(
        results_dir,
        dataset,
        sampler_a,
        [{"query": "q1", "evaluation_result": "is_correct", "generated_answer": "a"}],
    )

    monkeypatch.setattr("evals.eval_runner.evals_utils.get_dataset", lambda _: dataset)
    monkeypatch.setattr(
        "evals.eval_runner.evals_utils.get_sampler",
        lambda name: {"sampler_a": sampler_a, "sampler_b": sampler_b}[name],
    )

    args = argparse.Namespace(
        samplers=["sampler_a", "sampler_b"],
        datasets=["fake"],
        limit=None,
        seed=None,
        batch_size=10,
        max_concurrent_tasks=5,
        clean=False,
    )
    await run_evals(args, results_dir=results_dir)

    assert sorted(sampler_a.seen) == ["q2", "q3"]
    assert sorted(sampler_b.seen) == ["q1", "q2", "q3"]
