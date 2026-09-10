"""Helpers for generating per-run AutoML experiment launcher notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from kfp_components.components.training.automl.shared.run_status import shared_automl_dir

EXPERIMENT_NOTEBOOK_FILENAME = "automl_experiment_notebook.ipynb"
EXPERIMENT_NOTEBOOK_RELATIVE_PATH = f"notebooks/{EXPERIMENT_NOTEBOOK_FILENAME}"


def _py_str(value: str) -> str:
    """Return a safe Python string literal for notebook code cells."""
    return json.dumps(value)


def _py_list(values: list[str] | None) -> str:
    """Return a safe Python list literal for notebook code cells."""
    return json.dumps(values or [])


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
        "<REPLACE_S3_SECRET>": _py_str(train_data_secret_name),
        "<REPLACE_DATA_BUCKET>": _py_str(train_data_bucket_name),
        "<REPLACE_DATA_FILE_KEY>": _py_str(train_data_file_key),
        "<REPLACE_TEST_DATA_BUCKET>": _py_str(test_data_bucket_name),
        "<REPLACE_TEST_DATA_FILE_KEY>": _py_str(test_data_file_key),
        "<REPLACE_LABEL_COLUMN>": _py_str(label_column),
        "<REPLACE_TASK_TYPE>": _py_str(task_type),
        "<REPLACE_TOP_N>": str(top_n),
        "<REPLACE_POSITIVE_CLASS>": _py_str(positive_class),
        "<REPLACE_EVAL_METRIC>": _py_str(eval_metric),
        "<REPLACE_PRESET>": _py_str(preset),
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
        "<REPLACE_S3_SECRET>": _py_str(train_data_secret_name),
        "<REPLACE_DATA_BUCKET>": _py_str(train_data_bucket_name),
        "<REPLACE_DATA_FILE_KEY>": _py_str(train_data_file_key),
        "<REPLACE_TEST_DATA_BUCKET>": _py_str(test_data_bucket_name),
        "<REPLACE_TEST_DATA_FILE_KEY>": _py_str(test_data_file_key),
        "<REPLACE_TARGET>": _py_str(target),
        "<REPLACE_ID_COLUMN>": _py_str(id_column),
        "<REPLACE_TIMESTAMP_COLUMN>": _py_str(timestamp_column),
        "<REPLACE_KNOWN_COVARIATES_NAMES>": _py_list(known_covariates_names),
        "<REPLACE_PREDICTION_LENGTH>": str(prediction_length),
        "<REPLACE_TOP_N>": str(top_n),
        "<REPLACE_EVAL_METRIC>": _py_str(eval_metric),
        "<REPLACE_PRESET>": _py_str(preset),
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
