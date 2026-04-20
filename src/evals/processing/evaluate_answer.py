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

        match = re.search(r"(A|B|C)", grading_response)
        grade_letter = match.group(0) if match else "C"

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

        match = re.search(r"(TRUE|FALSE)", grading_response)
        grade_letter = match.group(0) if match else None

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

        match = re.search(r"correct: (yes|no)", grading_response)
        grade = match.group(1) if match else "no"

        is_correct = grade == "yes"
        is_incorrect = grade == "no"

        return {
            "grade": grade,
            "score_name": "is_correct" if is_correct else "is_incorrect",
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "score": is_correct,
        }
