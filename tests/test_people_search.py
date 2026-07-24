"""Unit tests for people-search field-fill scorers (no API calls)."""

import json

import pandas as pd
import pytest

from evals.eval_results_analyzer import write_metrics
from evals.processing.evaluate_answer import AnswerGrader
from evals.processing.people_search.field_fill import score_people_output
from evals.processing.people_search.llm_judges import _parse_label
from evals.processing.people_search.schema import normalize_people_payload


def test_score_people_output_empty():
    scores = score_people_output({"people": [], "person_count": 0}, {"persona_slug": "recruiter"})
    assert scores["has_people"] == 0.0
    assert scores["field_fill"] == 0.0
    assert scores["persona_field_fill"] == 0.0


def test_score_people_output_partial_fill():
    output = {
        "person_count": 1,
        "people": [
            {
                "displayname": "Ada Lovelace",
                "current_title": "Analyst",
                "current_company": "Analytical Engines",
                "location": "London",
                "linkedin_url": "https://example.com/ada",
            }
        ],
    }
    scores = score_people_output(output, {"persona_slug": "recruiter"})
    assert scores["has_people"] == 1.0
    assert scores["person_count"] == 1
    assert 0 < scores["field_fill"] < 1
    assert scores["persona_field_fill"] > 0


def test_normalize_people_payload_variants():
    canonical = normalize_people_payload(
        {"people": [{"displayname": "A"}], "person_count": 1}
    )
    assert canonical["person_count"] == 1
    assert len(canonical["people"]) == 1

    nested = normalize_people_payload(
        {"summary": {"people": [{"displayname": "B"}], "person_count": 1}}
    )
    assert nested["people"][0]["displayname"] == "B"

    results_key = normalize_people_payload({"results": [{"displayname": "C"}]})
    assert results_key["person_count"] == 1


def test_parse_judge_label():
    label, score = _parse_label("Looks good.\nLABEL: Useful\n")
    assert label == "Useful"
    assert score == 0.7
    label, score = _parse_label("LABEL: High Value")
    assert label == "High Value"
    assert score == 1.0


@pytest.mark.asyncio
async def test_evaluate_single_people_search_grader(monkeypatch):
    monkeypatch.setenv("PEOPLE_SEARCH_LLM_JUDGES", "0")
    grader = AnswerGrader()
    target = json.dumps(
        {
            "benchmark_id": "fp_001",
            "persona_slug": "recruiter",
            "query_type": "enrichment",
            "person_name": "Ada Lovelace",
            "company": "Analytical Engines",
        }
    )
    predicted = json.dumps(
        {
            "provider": "http_people_search",
            "person_count": 1,
            "people": [
                {
                    "displayname": "Ada Lovelace",
                    "current_title": "Analyst",
                    "current_company": "Analytical Engines",
                }
            ],
        }
    )
    result = await grader.evaluate_single_people_search(
        "Find Ada Lovelace", target, predicted
    )
    assert result["score_name"] == "has_people"
    assert result["has_people"] == 1.0
    assert "field_fill" in result
    assert "persona_field_fill" in result
    assert "judge_overall" not in result


def test_people_search_dataset_builds_metadata_from_columns():
    """Shipped CSV has empty answer; load time assembles metadata for the runner."""
    from evals import utils as evals_utils

    ds = evals_utils.get_dataset("people_search")
    assert ds.df["answer"].iloc[0]
    meta = json.loads(ds.df["answer"].iloc[0])
    assert meta["benchmark_id"] == "fp_001"
    assert meta["persona_slug"] == "recruiter"
    assert meta["query_type"] == "enrichment"
    # On-disk CSV answer column remains empty
    raw = pd.read_csv("data/people_search_full_dataset.csv")
    assert raw["answer"].isna().all() or (raw["answer"].fillna("") == "").all()


def test_people_search_analyzed_metrics_omit_accuracy(tmp_path):
    """people_search analyzed rows must not treat has_people as accuracy_score."""
    raw = tmp_path / "dataset_people_search_raw_results_http_people_search.csv"
    raw.write_text(
        "query,internal_response_time_ms,request_response_time_ms,"
        "evaluation_result,generated_answer,ground_truth,"
        "has_people,field_fill,persona_field_fill,judge_overall,judge_persona\n"
        'q1,10,20,has_people,"{}", "{}",1,0.5,0.6,0.7,0.7\n'
        'q2,10,20,no_people,"{}", "{}",0,0.0,0.0,0.0,0.0\n',
        encoding="utf-8",
    )
    write_metrics(tmp_path)
    df = pd.read_csv(tmp_path / "analyzed_results.csv")
    assert len(df) == 1
    assert pd.isna(df.loc[0, "accuracy_score"])
    assert float(df.loc[0, "mean_field_fill"]) == 0.25
    assert float(df.loc[0, "has_people_rate"]) == 0.5
    assert float(df.loc[0, "mean_judge_overall"]) == 0.35
