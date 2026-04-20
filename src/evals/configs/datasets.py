from dataclasses import dataclass
from collections.abc import Callable

import pandas as pd

from evals.processing.evaluate_answer import AnswerGrader


evaluator = AnswerGrader()


@dataclass
class Dataset:
    dataset_name: str
    csv_path: str
    grader: Callable
    df: pd.DataFrame | None


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
]
