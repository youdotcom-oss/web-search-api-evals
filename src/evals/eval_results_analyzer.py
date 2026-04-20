"""
Analyze and calculate metrics from evaluation results.

This module provides functionality to process raw evaluation results,
calculate performance metrics, and generate summary reports.
"""

import glob
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


def get_default_results_dir() -> Path:
    """Get the default results directory path."""
    return Path(os.getcwd(), "src/evals/results")


def get_results_files(results_dir: Optional[Path] = None) -> List[str]:
    """
    Get all raw results files from the results directory.

    Args:
        results_dir: Optional path to results directory. Defaults to src/evals/results

    Returns:
        List of file paths to raw results files
    """
    if results_dir is None:
        results_dir = get_default_results_dir()

    return glob.glob(f"{results_dir}/dataset_*.csv")


def write_metrics(results_dir: Optional[Path] = None):
    """
    Calculate metrics from raw results such as accuracy score, P50 latency, and average latency.

    Args:
        results_dir: Optional path to results directory. Defaults to src/evals/results
    """
    if results_dir is None:
        results_dir = get_default_results_dir()

    files = get_results_files(results_dir)
    metric_rows = []

    for sampler_results_file in files:
        dataset_name = sampler_results_file.split("dataset_")[1].split("_raw_results")[
            0
        ]
        sampler_name = sampler_results_file.split("raw_results_")[-1].split(".")[0]

        df_sampler_results = pd.read_csv(sampler_results_file)
        successful_df = df_sampler_results[
            (df_sampler_results["evaluation_result"] != "FAILED")
            & (df_sampler_results["generated_answer"] != "FAILED")
        ]

        p50_internal_latency = (
            pd.to_numeric(successful_df["internal_response_time_ms"], errors="coerce")
            .dropna()
            .median()
        )
        p50_request_response_latency = (
            pd.to_numeric(successful_df["request_response_time_ms"], errors="coerce")
            .dropna()
            .median()
        )
        correct = len(
            df_sampler_results[df_sampler_results["evaluation_result"] == "is_correct"]
        )
        count_answered = len(successful_df)

        if count_answered == 0:
            raise ValueError(f"No successful results found for sampler {sampler_name}")

        accuracy_score = round((correct / count_answered) * 100, 2)

        metric_rows.append(
            {
                "provider": sampler_name,
                "dataset": dataset_name,
                "accuracy_score": accuracy_score,
                "p50_internal_latency": round(float(p50_internal_latency), 2),
                "p50_request_response_latency": round(
                    float(p50_request_response_latency), 2
                ),
                "problem_count": count_answered,
            }
        )

    write_path = results_dir / "analyzed_results.csv"
    metric_df = pd.DataFrame(metric_rows).sort_values(
        ["dataset", "accuracy_score"], ascending=False
    )
    metric_df.to_csv(write_path, index=False)
    print(f"Results were written to {write_path}")
    print(metric_df)
