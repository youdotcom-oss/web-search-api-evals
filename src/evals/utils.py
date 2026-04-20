import base64
import hashlib
import json
import pandas as pd

from evals.configs import datasets, samplers


def get_dataset(dataset_name):
    dataset = next(
        (
            dataset
            for dataset in datasets.DATASETS
            if dataset.dataset_name == dataset_name
        ),
        None,
    )
    if dataset is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized, run python src/evals/eval_runner.py --help for available datasets"
        )
    dataset.df = pd.read_csv(dataset.csv_path)

    if dataset_name == "deepsearchqa":
        # Encode answer + answer_type together so the grader can access both
        dataset.df["answer"] = dataset.df.apply(
            lambda row: json.dumps(
                {"answer": row["answer"], "answer_type": row["answer_type"]}
            ),
            axis=1,
        )
    if dataset_name == "browsecomp":
        dataset.df["problem"] = dataset.df.apply(
            lambda row: _decrypt(row["problem"], row["canary"]), axis=1
        )
        dataset.df["answer"] = dataset.df.apply(
            lambda row: _decrypt(row["answer"], row["canary"]), axis=1
        )

    if dataset.df is None:
        raise ValueError(
            f"Failed to initialize df for {dataset_name} and csv_path {dataset.csv_path}"
        )
    return dataset


def get_sampler(sampler_name: str):
    sampler = next(
        (
            sampler
            for sampler in samplers.SAMPLERS
            if sampler.sampler_name == sampler_name
        ),
        None,
    )
    if sampler is None:
        raise ValueError(
            f"Sampler '{sampler_name}' not found. Available samplers: {[sampler.sampler_name for sampler in samplers.SAMPLERS]}"
        )
    return sampler


# Utilities used for browsecomp
def _derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from a password using SHA256."""
    key = hashlib.sha256(password.encode()).digest()
    return (key * (length // len(key))) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt a base64-encoded XOR-ciphered string."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()
