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


def _mean_numeric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return None
    return float(values.mean())


def write_metrics(results_dir: Optional[Path] = None):
    """
    Calculate metrics from raw results such as accuracy score, P50 latency, and average latency.

    For people_search (scorer-based), ``accuracy_score`` is left blank — primary quality
    metrics are ``mean_field_fill``, ``mean_persona_field_fill``, and ``mean_judge_*``,
    with ``has_people_rate`` as a separate retrieval signal (not “accuracy”).

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
        count_answered = len(successful_df)

        if count_answered == 0:
            raise ValueError(f"No successful results found for sampler {sampler_name}")

        is_people_search = dataset_name == "people_search" or (
            "field_fill" in successful_df.columns
        )

        row = {
            "provider": sampler_name,
            "dataset": dataset_name,
            "p50_internal_latency": round(float(p50_internal_latency), 2)
            if pd.notna(p50_internal_latency)
            else None,
            "p50_request_response_latency": round(
                float(p50_request_response_latency), 2
            )
            if pd.notna(p50_request_response_latency)
            else None,
            "problem_count": count_answered,
        }

        if is_people_search:
            # Do not reuse accuracy_score — that means gold-answer correctness elsewhere.
            row["accuracy_score"] = None
            if "has_people" in successful_df.columns:
                rate = _mean_numeric(successful_df["has_people"])
                if rate is not None:
                    row["has_people_rate"] = round(rate, 4)
            if "field_fill" in successful_df.columns:
                mean_ff = _mean_numeric(successful_df["field_fill"])
                if mean_ff is not None:
                    row["mean_field_fill"] = round(mean_ff, 4)
            if "persona_field_fill" in successful_df.columns:
                mean_pff = _mean_numeric(successful_df["persona_field_fill"])
                if mean_pff is not None:
                    row["mean_persona_field_fill"] = round(mean_pff, 4)
            if "judge_overall" in successful_df.columns:
                mean_jo = _mean_numeric(successful_df["judge_overall"])
                if mean_jo is not None:
                    row["mean_judge_overall"] = round(mean_jo, 4)
            if "judge_persona" in successful_df.columns:
                mean_jp = _mean_numeric(successful_df["judge_persona"])
                if mean_jp is not None:
                    row["mean_judge_persona"] = round(mean_jp, 4)
            # Sort key within people_search: prefer field fill, then judges
            row["_sort_score"] = row.get("mean_field_fill") or row.get(
                "mean_judge_overall"
            ) or row.get("has_people_rate") or 0.0
        else:
            correct = len(
                df_sampler_results[
                    df_sampler_results["evaluation_result"] == "is_correct"
                ]
            )
            accuracy_score = round((correct / count_answered) * 100, 2)
            row["accuracy_score"] = accuracy_score
            row["_sort_score"] = accuracy_score

        metric_rows.append(row)

    write_path = results_dir / "analyzed_results.csv"
    metric_df = pd.DataFrame(metric_rows).sort_values(
        ["dataset", "_sort_score"], ascending=[True, False]
    )
    metric_df = metric_df.drop(columns=["_sort_score"])
    metric_df.to_csv(write_path, index=False)
    print(f"Results were written to {write_path}")
