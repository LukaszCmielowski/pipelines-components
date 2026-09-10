"""Helpers for generating per-run AutoML experiment launcher notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from kfp_components.components.training.automl.shared.run_status import shared_automl_dir

EXPERIMENT_NOTEBOOK_FILENAME = "automl_experiment_notebook.ipynb"
EXPERIMENT_NOTEBOOK_RELATIVE_PATH = f"notebooks/{EXPERIMENT_NOTEBOOK_FILENAME}"


def replace_placeholder_in_notebook(notebook: dict, replacements: dict[str, str]) -> dict:
    """Replace placeholder tokens in code cell sources."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        new_source = []
        for line in cell.get("source", []):
            for placeholder, value in replacements.items():
                line = line.replace(placeholder, value)
            new_source.append(line)
        cell["source"] = new_source
    return notebook


def _template_name(kind: Literal["tabular", "timeseries"]) -> str:
    return f"{kind}_experiment_notebook.ipynb"


def tabular_experiment_notebook_replacements(
    *,
    train_data_secret_name: str,
    train_data_bucket_name: str,
    train_data_file_key: str,
    test_data_bucket_name: str,
    test_data_file_key: str,
    label_column: str,
    task_type: str,
    top_n: int,
    positive_class: str,
    eval_metric: str,
    preset: str,
) -> dict[str, str]:
    """Build placeholder replacements for the tabular experiment notebook template."""
    return {
        "<REPLACE_S3_SECRET>": train_data_secret_name,
        "<REPLACE_DATA_BUCKET>": train_data_bucket_name,
        "<REPLACE_DATA_FILE_KEY>": train_data_file_key,
        "<REPLACE_TEST_DATA_BUCKET>": test_data_bucket_name,
        "<REPLACE_TEST_DATA_FILE_KEY>": test_data_file_key,
        "<REPLACE_LABEL_COLUMN>": label_column,
        "<REPLACE_TASK_TYPE>": task_type,
        "<REPLACE_TOP_N>": str(top_n),
        "<REPLACE_POSITIVE_CLASS>": positive_class,
        "<REPLACE_EVAL_METRIC>": eval_metric,
        "<REPLACE_PRESET>": preset,
    }


def timeseries_experiment_notebook_replacements(
    *,
    train_data_secret_name: str,
    train_data_bucket_name: str,
    train_data_file_key: str,
    test_data_bucket_name: str,
    test_data_file_key: str,
    target: str,
    id_column: str,
    timestamp_column: str,
    known_covariates_names: list[str] | None,
    prediction_length: int,
    top_n: int,
    eval_metric: str,
    preset: str,
) -> dict[str, str]:
    """Build placeholder replacements for the timeseries experiment notebook template."""
    return {
        "<REPLACE_S3_SECRET>": train_data_secret_name,
        "<REPLACE_DATA_BUCKET>": train_data_bucket_name,
        "<REPLACE_DATA_FILE_KEY>": train_data_file_key,
        "<REPLACE_TEST_DATA_BUCKET>": test_data_bucket_name,
        "<REPLACE_TEST_DATA_FILE_KEY>": test_data_file_key,
        "<REPLACE_TARGET>": target,
        "<REPLACE_ID_COLUMN>": id_column,
        "<REPLACE_TIMESTAMP_COLUMN>": timestamp_column,
        "<REPLACE_KNOWN_COVARIATES_NAMES>": repr(known_covariates_names or []),
        "<REPLACE_PREDICTION_LENGTH>": str(prediction_length),
        "<REPLACE_TOP_N>": str(top_n),
        "<REPLACE_EVAL_METRIC>": eval_metric,
        "<REPLACE_PRESET>": preset,
    }


def write_experiment_notebook(
    *,
    output_dir: Path,
    kind: Literal["tabular", "timeseries"],
    replacements: dict[str, str],
) -> Path:
    """Write a run-level experiment launcher notebook under ``output_dir/notebooks/``."""
    template_path = shared_automl_dir() / "notebook_templates" / _template_name(kind)
    with template_path.open(encoding="utf-8") as f:
        notebook = json.load(f)

    notebook = replace_placeholder_in_notebook(notebook, replacements)
    notebook_path = output_dir / "notebooks"
    notebook_path.mkdir(parents=True, exist_ok=True)
    destination = notebook_path / EXPERIMENT_NOTEBOOK_FILENAME
    with destination.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
        f.write("\n")
    return destination
