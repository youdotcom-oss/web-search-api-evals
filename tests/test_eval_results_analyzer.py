"""Tests for metrics aggregation, including how failed rows are reported."""

import pandas as pd

from evals.eval_results_analyzer import write_metrics


def _write_raw(results_dir, dataset, sampler, rows):
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"dataset_{dataset}_raw_results_{sampler}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(query, evaluation_result, internal=10.0, request=20.0):
    answer = "FAILED" if evaluation_result == "FAILED" else f"answer for {query}"
    return {
        "query": query,
        "internal_response_time_ms": "FAILED"
        if evaluation_result == "FAILED"
        else internal,
        "request_response_time_ms": "FAILED"
        if evaluation_result == "FAILED"
        else request,
        "evaluation_result": evaluation_result,
        "generated_answer": answer,
        "ground_truth": "gt",
    }


def test_failed_rows_are_reported_not_hidden(tmp_path):
    """Accuracy excludes failures, so the count must be visible alongside it."""
    _write_raw(
        tmp_path,
        "fake",
        "sampler_a",
        [
            _row("q1", "is_correct"),
            _row("q2", "is_incorrect"),
            _row("q3", "FAILED"),
            _row("q4", "FAILED"),
        ],
    )

    write_metrics(tmp_path)
    metrics = pd.read_csv(tmp_path / "analyzed_results.csv")

    row = metrics.iloc[0]
    assert row["problem_count"] == 2
    assert row["failed_count"] == 2
    # 1 correct out of the 2 that were actually answered
    assert row["accuracy_score"] == 50.0


def test_one_fully_failed_sampler_does_not_destroy_the_summary(tmp_path):
    """Regression: a single dead sampler used to raise and lose every other result."""
    _write_raw(tmp_path, "fake", "healthy", [_row("q1", "is_correct")])
    _write_raw(tmp_path, "fake", "dead", [_row("q1", "FAILED"), _row("q2", "FAILED")])

    write_metrics(tmp_path)
    metrics = pd.read_csv(tmp_path / "analyzed_results.csv")

    assert metrics["provider"].tolist() == ["healthy"]
    assert metrics.iloc[0]["accuracy_score"] == 100.0


def test_no_summary_written_when_everything_failed(tmp_path):
    _write_raw(tmp_path, "fake", "dead", [_row("q1", "FAILED")])

    write_metrics(tmp_path)

    assert not (tmp_path / "analyzed_results.csv").exists()
