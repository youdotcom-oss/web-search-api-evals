"""Tests for parsing LLM judge responses into scores.

The grader model is stubbed, so these run without network or API keys.
"""

import json

import pytest

from evals.processing import evaluate_answer
from evals.processing.evaluate_answer import AnswerGrader


@pytest.fixture
def grader_returning(monkeypatch):
    """Stub the judge so a canned response can be pushed through the real parser."""

    def _install(response: str) -> AnswerGrader:
        async def fake_call_llm(model, system_prompt, user_prompt):
            return response

        monkeypatch.setattr(evaluate_answer.llm, "call_llm", fake_call_llm)
        return AnswerGrader(model="gpt-test")

    return _install


# --- SimpleQA -------------------------------------------------------------


@pytest.mark.asyncio
async def test_simpleqa_bare_letter(grader_returning):
    result = await grader_returning("A").evaluate_single_simpleqa("q", "t", "p")
    assert result["is_correct"] is True


@pytest.mark.asyncio
async def test_simpleqa_ignores_capital_inside_a_word(grader_returning):
    """Regression: an unanchored (A|B|C) matched the B in "Based", grading A as B."""
    result = await grader_returning(
        "Based on the gold target, the predicted answer matches.\nA"
    ).evaluate_single_simpleqa("q", "t", "p")

    assert result["grade"] == "A"
    assert result["is_correct"] is True


@pytest.mark.asyncio
async def test_simpleqa_reads_the_verdict_after_an_echoed_rubric(grader_returning):
    result = await grader_returning(
        "A: CORRECT\nB: INCORRECT\nC: NOT_ATTEMPTED\n\nB"
    ).evaluate_single_simpleqa("q", "t", "p")

    assert result["grade"] == "B"
    assert result["is_incorrect"] is True


@pytest.mark.asyncio
async def test_simpleqa_unparseable_defaults_to_not_attempted(grader_returning):
    result = await grader_returning("no verdict here").evaluate_single_simpleqa(
        "q", "t", "p"
    )
    assert result["grade"] == "C"
    assert result["is_not_attempted"] is True


# --- FRAMES ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_frames_reads_labelled_decision_not_the_explanation(grader_returning):
    """Regression: the first TRUE/FALSE was often a word inside the reasoning."""
    result = await grader_returning(
        'Explanation: It is FALSE to say these differ; the values agree.\n'
        'Decision: "TRUE"'
    ).evaluate_single_frames("q", "t", "p")

    assert result["grade"] == "TRUE"
    assert result["is_correct"] is True


@pytest.mark.asyncio
async def test_frames_plain_decision(grader_returning):
    result = await grader_returning("Decision: FALSE").evaluate_single_frames(
        "q", "t", "p"
    )
    assert result["is_incorrect"] is True


@pytest.mark.asyncio
async def test_frames_unparseable_does_not_raise(grader_returning):
    """Regression: a missing match indexed a dict with None and raised KeyError."""
    result = await grader_returning("I cannot determine this.").evaluate_single_frames(
        "q", "t", "p"
    )

    assert result["grade"] == "FALSE"
    assert result["is_correct"] is False


# --- BrowseComp -----------------------------------------------------------


@pytest.mark.asyncio
async def test_browsecomp_is_case_and_space_tolerant(grader_returning):
    result = await grader_returning(
        "extracted_final_answer: Paris\nCorrect:   yes\nconfidence: 90%"
    ).evaluate_single_browsecomp("q", "t", "p")

    assert result["is_correct"] is True


@pytest.mark.asyncio
async def test_browsecomp_unparseable_defaults_to_no(grader_returning):
    result = await grader_returning("unclear").evaluate_single_browsecomp("q", "t", "p")
    assert result["is_correct"] is False


# --- FinSearchComp --------------------------------------------------------


@pytest.mark.asyncio
async def test_fin_search_reads_the_json_score(grader_returning):
    result = await grader_returning(
        'Scoring Basis: values match.\n- JSON:\n```\n{"answer_score": 1}\n```'
    ).evaluate_single_fin_search("q", "t", "p")

    assert result["is_correct"] is True


@pytest.mark.asyncio
async def test_fin_search_reads_the_last_score_when_an_example_is_echoed(
    grader_returning,
):
    result = await grader_returning(
        'Example format: {"answer_score": 1}\n'
        'Scoring Basis: the value is wrong.\n{"answer_score": 0}'
    ).evaluate_single_fin_search("q", "t", "p")

    assert result["is_correct"] is False


@pytest.mark.asyncio
async def test_fin_search_unparseable_defaults_to_zero(grader_returning):
    result = await grader_returning("no json here").evaluate_single_fin_search(
        "q", "t", "p"
    )
    assert result["is_correct"] is False


# --- shared ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepsearchqa_target_is_still_decoded(grader_returning):
    """The dataset packs answer and answer_type into JSON; parsing must survive."""
    target = json.dumps({"answer": "1843", "answer_type": "number"})
    grader = grader_returning('{"correct": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}')

    result = await grader.evaluate_single_deepsearchqa("q", target, "1843")

    assert "is_correct" in result
