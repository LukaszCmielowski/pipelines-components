"""Tests for experiment notebook generation helpers."""

# ruff: noqa: D102

import json
from pathlib import Path

from ..experiment_notebook_utils import (
    EXPERIMENT_NOTEBOOK_FILENAME,
    EXPERIMENT_NOTEBOOK_RELATIVE_PATH,
    replace_placeholder_in_notebook,
    tabular_experiment_notebook_replacements,
    timeseries_experiment_notebook_replacements,
    write_experiment_notebook,
)

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "notebook_templates"


class TestExperimentNotebookUtils:
    """Unit tests for experiment notebook helper functions."""

    def test_replace_placeholder_in_notebook_replaces_code_cells_only(self):
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["<REPLACE_TASK_TYPE>\n"]},
                {"cell_type": "code", "source": ['task_type = "<REPLACE_TASK_TYPE>"\n']},
            ]
        }
        updated = replace_placeholder_in_notebook(notebook, {"<REPLACE_TASK_TYPE>": "regression"})
        assert updated["cells"][0]["source"] == ["<REPLACE_TASK_TYPE>\n"]
        assert updated["cells"][1]["source"] == ['task_type = "regression"\n']

    def test_tabular_experiment_notebook_replacements_maps_values(self):
        replacements = tabular_experiment_notebook_replacements(
            train_data_secret_name="secret",
            train_data_bucket_name="bucket",
            train_data_file_key="datasets/train.csv",
            test_data_bucket_name="",
            test_data_file_key="",
            label_column="price",
            task_type="regression",
            top_n=3,
            positive_class="",
            eval_metric="r2",
            preset="speed",
        )
        assert replacements["<REPLACE_S3_SECRET>"] == "secret"
        assert replacements["<REPLACE_TASK_TYPE>"] == "regression"
        assert replacements["<REPLACE_TOP_N>"] == "3"

    def test_timeseries_experiment_notebook_replacements_serializes_covariates(self):
        replacements = timeseries_experiment_notebook_replacements(
            train_data_secret_name="secret",
            train_data_bucket_name="bucket",
            train_data_file_key="datasets/ts.csv",
            test_data_bucket_name="test-bucket",
            test_data_file_key="datasets/test.csv",
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            known_covariates_names=["promo"],
            prediction_length=24,
            top_n=2,
            eval_metric="mean_absolute_scaled_error",
            preset="balanced",
        )
        assert replacements["<REPLACE_KNOWN_COVARIATES_NAMES>"] == "['promo']"
        assert replacements["<REPLACE_PREDICTION_LENGTH>"] == "24"

    def test_write_experiment_notebook_tabular(self, tmp_path):
        destination = write_experiment_notebook(
            output_dir=tmp_path,
            kind="tabular",
            replacements=tabular_experiment_notebook_replacements(
                train_data_secret_name="my-secret",
                train_data_bucket_name="my-bucket",
                train_data_file_key="datasets/train.csv",
                test_data_bucket_name="",
                test_data_file_key="",
                label_column="target",
                task_type="binary",
                top_n=3,
                positive_class="yes",
                eval_metric="accuracy",
                preset="speed",
            ),
        )
        assert destination == tmp_path / "notebooks" / EXPERIMENT_NOTEBOOK_FILENAME
        assert destination.exists()
        notebook = json.loads(destination.read_text(encoding="utf-8"))
        source = "".join(
            line for cell in notebook["cells"] if cell.get("cell_type") == "code" for line in cell.get("source", [])
        )
        assert "<REPLACE_S3_SECRET>" not in source
        assert 'train_data_secret_name = "my-secret"' in source
        assert 'task_type = "binary"' in source
        assert "kfp_components" not in source
        assert "client.run_pipeline" in source

    def test_write_experiment_notebook_timeseries(self, tmp_path):
        destination = write_experiment_notebook(
            output_dir=tmp_path,
            kind="timeseries",
            replacements=timeseries_experiment_notebook_replacements(
                train_data_secret_name="secret",
                train_data_bucket_name="bucket",
                train_data_file_key="datasets/ts.csv",
                test_data_bucket_name="",
                test_data_file_key="",
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                known_covariates_names=[],
                prediction_length=12,
                top_n=3,
                eval_metric="mean_absolute_scaled_error",
                preset="speed",
            ),
        )
        assert destination.name == EXPERIMENT_NOTEBOOK_FILENAME
        notebook = json.loads(destination.read_text(encoding="utf-8"))
        source = "".join(
            line for cell in notebook["cells"] if cell.get("cell_type") == "code" for line in cell.get("source", [])
        )
        assert 'pipeline_name = "autogluon-timeseries-training-pipeline"' in source
        assert 'target = "sales"' in source
        assert EXPERIMENT_NOTEBOOK_RELATIVE_PATH.endswith(EXPERIMENT_NOTEBOOK_FILENAME)
