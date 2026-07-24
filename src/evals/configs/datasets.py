from dataclasses import dataclass
from collections.abc import Callable

import pandas as pd

from evals import constants
from evals.processing.evaluate_answer import AnswerGrader


evaluator = AnswerGrader()
fin_search_evaluator = AnswerGrader(model=constants.FIN_SEARCH_GRADER_MODEL)


@dataclass
class Dataset:
    dataset_name: str
    csv_path: str
    grader: Callable
    df: pd.DataFrame | None
    # When False, empty answer/ground_truth is allowed (scorer-based datasets).
    requires_ground_truth: bool = True


DATASETS = [
    Dataset(
        dataset_name="deepsearchqa",
        csv_path="data/deepsearchqa_full_dataset.csv",
        grader=evaluator.evaluate_single_deepsearchqa,
        df=None,
    ),
    Dataset(
        dataset_name="browsecomp",
        csv_path="data/browsecomp_full_dataset.csv",
        grader=evaluator.evaluate_single_browsecomp,
        df=None,
    ),
    Dataset(
        dataset_name="frames",
        csv_path="data/frames_full_dataset.csv",
        grader=evaluator.evaluate_single_frames,
        df=None,
    ),
    Dataset(
        dataset_name="simpleqa",
        csv_path="data/simpleqa_full_dataset.csv",
        grader=evaluator.evaluate_single_simpleqa,
        df=None,
    ),
    Dataset(
        dataset_name="fin_search_comp_t2_global",
        csv_path="data/fin_search_comp_t2_global.csv",
        grader=fin_search_evaluator.evaluate_single_fin_search,
        df=None,
    ),
    Dataset(
        dataset_name="fin_search_comp_t3_global",
        csv_path="data/fin_search_comp_t3_global.csv",
        grader=fin_search_evaluator.evaluate_single_fin_search,
        df=None,
    ),
    Dataset(
        dataset_name="people_search",
        csv_path="data/people_search_full_dataset.csv",
        grader=evaluator.evaluate_single_people_search,
        df=None,
        # answer column is empty in the CSV (no gold answers); metadata lives in
        # dedicated columns and is assembled at load time in utils.get_dataset
        requires_ground_truth=False,
    ),
]
