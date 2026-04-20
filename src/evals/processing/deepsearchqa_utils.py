"""
Utility functions for processing DeepSearchQA dataset
"""

import json
import logging
from typing import Any


def _parse_json_response(ori_json_response: str) -> Any:
    """Parse a JSON object from the grader response, handling ```json fences."""
    try:
        json_str = ori_json_response.strip()
        start_idx = json_str.find("```json")
        if start_idx != -1:
            json_str = json_str[start_idx + len("```json") :].strip()
            end_idx = json_str.rfind("```")
            if end_idx != -1:
                json_str = json_str[:end_idx].strip()
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.info("json.JSONDecodeError: %s for response: %s", e, ori_json_response)
        return None


def _get_answer_correctness_details(json_response: Any) -> dict[str, bool] | None:
    try:
        details = json_response["Answer Correctness"]["Correctness Details"]
        if isinstance(details, dict):
            if all(isinstance(k, str) for k in details) and all(
                isinstance(v, bool) for v in details.values()
            ):
                return details
        logging.warning("Invalid format for Correctness Details: %s", details)
        return None
    except (KeyError, TypeError) as e:
        logging.info("Error extracting Correctness Details: %s", e)
        return None


def _get_excessive_answers(json_response: Any) -> list[str] | None:
    try:
        excessive = json_response["Answer Correctness"]["Excessive Answers"]
        if isinstance(excessive, list) and all(isinstance(i, str) for i in excessive):
            return excessive
        logging.warning("Invalid format for Excessive Answers: %s", excessive)
        return None
    except (KeyError, TypeError) as e:
        logging.info("Error extracting Excessive Answers (treating as empty): %s", e)
        return []


def _compute_deepsearchqa_scores(grader_response: str) -> dict[str, Any]:
    """Parse the grader JSON and compute correct/f1/precision/recall scores."""
    error: dict[str, Any] = {
        "scores": {
            "correct": 0.0,
            "correct_with_excessive": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        },
        "metadata": {},
    }

    parsed = _parse_json_response(grader_response)
    if parsed is None:
        error["metadata"] = {
            "error": "Failed to parse JSON",
            "raw_response": grader_response,
        }
        return error
    if parsed == "NULL":
        error["metadata"] = {"error": "Grader returned NULL (empty inputs)"}
        return error

    correctness_details = _get_answer_correctness_details(parsed)
    if correctness_details is None:
        error["metadata"] = {
            "error": "Failed to extract correctness details",
            "raw_response": grader_response,
        }
        return error

    excessive_answers = _get_excessive_answers(parsed)
    if excessive_answers is None:
        error["metadata"] = {
            "error": "Failed to extract excessive answers",
            "raw_response": grader_response,
        }
        return error

    explanation = ""
    try:
        explanation = parsed["Answer Correctness"].get("Explanation", "")
    except (KeyError, TypeError):
        pass

    num_correct = sum(1 for v in correctness_details.values() if v)
    num_expected = len(correctness_details)
    num_excessive = len(excessive_answers)

    all_expected_correct = (num_correct == num_expected) if num_expected > 0 else True
    correct = 1.0 if (all_expected_correct and num_excessive == 0) else 0.0
    correct_with_excessive = 1.0 if all_expected_correct else 0.0

    tp = num_correct
    fp = num_excessive
    fn = num_expected - num_correct
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "scores": {
            "correct": correct,
            "correct_with_excessive": correct_with_excessive,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        },
        "metadata": {
            "explanation": explanation,
            "correctness_details": correctness_details,
            "excessive_answers": excessive_answers,
            "num_correct": num_correct,
            "num_expected": num_expected,
            "num_excessive": num_excessive,
        },
    }
