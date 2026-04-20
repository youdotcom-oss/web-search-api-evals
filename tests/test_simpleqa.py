"""Tests for SimpleQA evaluation runner"""

import argparse
import os
from pathlib import Path
import shutil

import dotenv
import pandas as pd
import pytest

from evals.eval_runner import (
    run_evals,
    get_sampler_filepath,
)
from evals import utils as evals_utils
from evals.eval_results_analyzer import write_metrics
from evals.configs import samplers


dotenv.load_dotenv()


def get_test_results_dir() -> Path:
    """Get the test results directory path."""
    return Path(os.getcwd(), "tests/results")


@pytest.fixture
def test_results_cleanup():
    """Cleanup test results before and after test"""
    results_folder_path = get_test_results_dir()

    # Clean before test
    if os.path.isdir(results_folder_path):
        shutil.rmtree(results_folder_path)

    yield

    # Clean after test
    if os.path.isdir(results_folder_path):
        shutil.rmtree(results_folder_path)


@pytest.mark.asyncio
async def test_simpleqa_runner(test_results_cleanup):
    """
    Test running simpleqa_runner for all samplers
    on the simple_qa dataset.

    This test verifies that:
    1. The runner can process queries using all samplers
    2. Results are written to the correct output file
    3. Results contain expected columns and data
    4. Metrics are calculated correctly
    """
    # Create test arguments
    num_problems = 10
    results_dir = get_test_results_dir()
    args = argparse.Namespace(
        samplers=[sampler.sampler_name for sampler in samplers.SAMPLERS],
        datasets=["simpleqa"],
        limit=num_problems,  # Test with small subset for speed
        batch_size=10,
        max_concurrent_tasks=10,
        num_results=5,
        clean=True,
    )

    # Run the evaluation
    await run_evals(args, results_dir=results_dir)
    for sampler in args.samplers:
        sampler = evals_utils.get_sampler(sampler)
        dataset = evals_utils.get_dataset(args.datasets[0])
        results_filepath = get_sampler_filepath(sampler, dataset, results_dir)
        assert os.path.isfile(
            results_filepath
        ), f"Results file not created at {results_filepath}"

        # Read and verify results
        df_results = pd.read_csv(results_filepath)

        # Check expected columns exist
        expected_columns = [
            "query",
            "internal_response_time_ms",
            "end_to_end_time_ms",
            "evaluation_result",
        ]
        for col in expected_columns:
            assert (
                col in df_results.columns
            ), f"Expected column '{col}' not found in results"

        # Verify we got results for the queries
        assert (
            len(df_results) == num_problems
        ), f"Expected {num_problems} results, got {len(df_results)}"

        # Verify all queries have non-null values
        assert df_results["query"].notna().all(), "Some queries are null"

        # Write and verify metrics
        write_metrics(results_dir)
        metrics_path = results_dir / "analyzed_results.csv"
        assert os.path.isfile(metrics_path), "Metrics file not created"

        df_metrics = pd.read_csv(metrics_path)
        assert len(df_metrics) == len(
            args.samplers
        ), f"Expected {len(args.samplers)} sampler in metrics"
        assert (
            df_metrics["provider"].drop_duplicates().tolist().sort()
            == args.samplers.sort()
        )
        assert "accuracy_score" in df_metrics.columns
        assert "avg_internal_latency" in df_metrics.columns
        assert "avg_end_to_end_latency" in df_metrics.columns
        assert "problem_count" in df_metrics.columns


@pytest.mark.asyncio
async def test_simpleqa_runner_resume_capability(test_results_cleanup):
    """
    Test that the runner can resume from partial results.

    This test verifies that if a run is interrupted, it can continue
    from where it left off without re-processing completed queries.
    """
    num_problems = 10
    results_dir = get_test_results_dir()
    # Create test arguments for first run (partial)
    args = argparse.Namespace(
        samplers=["you_search_with_livecrawl"],
        datasets=["simpleqa"],
        limit=num_problems,
        batch_size=5,
        max_concurrent_tasks=5,
        num_results=5,
        clean=True,
    )

    # First run
    await run_evals(args, results_dir=results_dir)
    for sampler in args.samplers:
        sampler = evals_utils.get_sampler(sampler)
        dataset = evals_utils.get_dataset(args.datasets[0])
        results_filepath = get_sampler_filepath(sampler, dataset, results_dir)
        df_first = pd.read_csv(results_filepath)
        first_run_count = len(df_first)
        assert (
            first_run_count == num_problems
        ), f"Expected {num_problems} results from first run, got {first_run_count}"

    # Second run with more queries (should add new results)
    args.limit = 5
    args.clean = False  # Don't clean, resume from existing
    await run_evals(args, results_dir=results_dir)
    for sampler in args.samplers:
        sampler = evals_utils.get_sampler(sampler)
        dataset = evals_utils.get_dataset(args.datasets[0])
        results_filepath = get_sampler_filepath(sampler, dataset, results_dir)
        df_second = pd.read_csv(results_filepath)
        second_run_count = len(df_second)
        assert (
            second_run_count == num_problems + args.limit
        ), f"Expected {num_problems} total results after second run, got {second_run_count}"
