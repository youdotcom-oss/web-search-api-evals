"""
This class is used to evaluate the correctness of the response. No changes have been made to the grading prompt.

To view or edit the model used for grading, see evals.constants
"""

import json
import logging
import re
from typing import Dict, Any

from evals import constants
from evals.processing import llm, deepsearchqa_utils


def _last_match(pattern: str, text: str, flags: int = 0) -> str | None:
    """Return the last capture-group match in `text`, or None if there is none.

    Graders are prompted to end with their verdict, and several prompts ask for
    reasoning first, restate the rubric, or include worked examples. Reading the
    last match keeps an echoed rubric or a word inside the explanation from being
    mistaken for the decision.
    """
    matches = re.findall(pattern, text, flags)
    return matches[-1] if matches else None


def _fallback(sampler_context: str, grading_response: str, default: str) -> str:
    """Log and return `default` when a grader response can't be parsed.

    Unparseable responses previously scored as incorrect without a trace, which
    is indistinguishable from a genuine wrong answer.
    """
    logging.warning(
        f"Could not parse {sampler_context} grader response, defaulting to "
        f"'{default}'. Response was: {grading_response[:200]!r}"
    )
    return default


class AnswerGrader:
    def __init__(self, model: str = constants.GRADER_MODEL):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model

    async def evaluate_single_simpleqa(
        self, question: str, target: str, predicted_answer: str
    ) -> Dict[str, Any]:
        """Evaluate a single response asynchronously for SimpleQA dataset"""
        grader_prompt = constants.SIMPLEQA_ANSWER_GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )

        grading_response = await llm.call_llm(self.model, "", grader_prompt)

        # Standalone letter only: an unanchored (A|B|C) matches the capital in a
        # word like "Based", silently grading a correct answer as INCORRECT.
        grade_letter = _last_match(r"\b([ABC])\b", grading_response) or _fallback(
            "SimpleQA", grading_response, "C"
        )

        score_name = {"A": "is_correct", "B": "is_incorrect", "C": "is_not_attempted"}[
            grade_letter
        ]

        is_correct = grade_letter == "A"
        is_incorrect = grade_letter == "B"
        is_not_attempted = grade_letter == "C"

        return {
            "grade": grade_letter,
            "score_name": score_name,
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "is_not_attempted": is_not_attempted,
            "score": is_correct,
        }

    async def evaluate_single_frames(
        self, question: str, target: str, predicted_answer: str
    ) -> Dict[str, Any]:
        """Evaluate a single response asynchronously for frames dataset"""
        grader_prompt = constants.FRAMES_ANSWER_GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )

        grading_response = await llm.call_llm(self.model, "", grader_prompt)

        # The prompt asks for an explanation before the decision, so the first
        # TRUE/FALSE in the response is frequently a word from the reasoning.
        # Prefer the labelled decision, then fall back to the last standalone
        # occurrence. A missing match previously raised KeyError.
        grade_letter = (
            _last_match(r'Decision:\s*"?(TRUE|FALSE)', grading_response, re.IGNORECASE)
            or _last_match(r"\b(TRUE|FALSE)\b", grading_response, re.IGNORECASE)
            or _fallback("FRAMES", grading_response, "FALSE")
        ).upper()

        score_name = {"TRUE": "is_correct", "FALSE": "is_incorrect"}[grade_letter]

        is_correct = grade_letter == "TRUE"
        is_incorrect = grade_letter == "FALSE"

        return {
            "grade": grade_letter,
            "score_name": score_name,
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "score": is_correct,
        }

    async def evaluate_single_deepsearchqa(
        self, question: str, target: str, predicted_answer: str
    ) -> Dict[str, Any]:
        """Evaluate a single response asynchronously for DeepSearchQA dataset.

        `target` is a JSON string with keys "answer" and "answer_type",
        encoded at dataset load time in utils.get_dataset.
        """
        target_dict = json.loads(target)
        answer = target_dict["answer"]
        answer_type = target_dict["answer_type"]

        grader_prompt = constants.DEEPSEARCHQA_GRADER_TEMPLATE.format(
            input=question,
            output=predicted_answer,
            expected=answer,
            answer_type=answer_type,
        )

        grading_response = await llm.call_llm(self.model, "", grader_prompt)

        result = deepsearchqa_utils._compute_deepsearchqa_scores(grading_response)
        scores = result["scores"]
        is_correct = scores["correct"] == 1.0

        return {
            "grade": "correct" if is_correct else "incorrect",
            "score_name": "is_correct" if is_correct else "is_incorrect",
            "is_correct": is_correct,
            "is_incorrect": not is_correct,
            "score": is_correct,
            # Additional DeepSearchQA-specific scores preserved in metadata
            "correct_with_excessive": scores["correct_with_excessive"],
            "f1": scores["f1"],
            "precision": scores["precision"],
            "recall": scores["recall"],
        }

    async def evaluate_single_browsecomp(
        self, question: str, target: str, predicted_answer: str
    ) -> Dict[str, Any]:
        """Evaluate a single response asynchronously for BrowseComp dataset"""
        grader_prompt = constants.BROWSECOMP_GRADER_TEMPLATE.format(
            question=question,
            correct_answer=target,
            response=predicted_answer,
        )

        grading_response = await llm.call_llm(self.model, "", grader_prompt)

        grade = (
            _last_match(r"correct:\s*(yes|no)", grading_response, re.IGNORECASE)
            or _fallback("BrowseComp", grading_response, "no")
        ).lower()

        is_correct = grade == "yes"
        is_incorrect = grade == "no"

        return {
            "grade": grade,
            "score_name": "is_correct" if is_correct else "is_incorrect",
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "score": is_correct,
        }

    async def evaluate_single_fin_search(
        self, question: str, target: str, predicted_answer: str
    ) -> Dict[str, Any]:
        """Evaluate a single response asynchronously for the FinSearchComp dataset."""
        grader_prompt = constants.FIN_SEARCH_GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )

        grading_response = await llm.call_llm(self.model, "", grader_prompt)

        # The prompt embeds worked examples, so read the last score rather than
        # the first in case the grader echoes one back.
        raw_score = _last_match(r'"answer_score"\s*:\s*([01])', grading_response)
        score_value = int(raw_score if raw_score is not None else _fallback(
            "FinSearchComp", grading_response, "0"
        ))

        is_correct = score_value == 1
        is_incorrect = score_value == 0

        return {
            "grade": str(score_value),
            "score_name": "is_correct" if is_correct else "is_incorrect",
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "score": is_correct,
        }
