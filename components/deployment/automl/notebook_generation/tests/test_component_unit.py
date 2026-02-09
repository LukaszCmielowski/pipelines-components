"""Tests for the notebook_generation component."""

import json
import tempfile
from pathlib import Path

import pytest

from ..component import notebook_generation

# Regression template: sample_row injected at cell 20; classification at cell 21.
SAMPLE_ROW_CELL_REGRESSION = 20
SAMPLE_ROW_CELL_CLASSIFICATION = 21


def _run_component(tmpdir, problem_type="regression", **kwargs):
    """Run notebook_generation with a mock artifact at tmpdir; return path to notebook."""
    mock_artifact = type("Artifact", (), {"path": tmpdir})()
    defaults = {
        "problem_type": problem_type,
        "model_name": "TestModel",
        "notebook_artifact": mock_artifact,
        "pipeline_name": "pipeline-foo",
        "run_id": "run-1",
        "sample_row": json.dumps([{"x": 1}]),
        "label_column": "x",
    }
    defaults.update(kwargs)
    notebook_generation.python_func(**defaults)
    return Path(tmpdir) / "automl_predictor_notebook.ipynb"


def _load_notebook(tmpdir):
    """Load generated notebook JSON from tmpdir."""
    path = Path(tmpdir) / "automl_predictor_notebook.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


class TestNotebookGenerationUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(notebook_generation)
        assert hasattr(notebook_generation, "python_func")

    def test_notebook_generation_with_valid_parameters(self):
        """Test component generates notebook with injected pipeline name, run_id, model, and sample row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_row = {"feature_a": 1.0, "feature_b": 2.0, "target": 0.5}
            _run_component(
                tmpdir,
                problem_type="regression",
                model_name="WeightedEnsemble_L2",
                pipeline_name="my-autogluon-pipeline-abc123",
                run_id="run-xyz-456",
                sample_row=json.dumps([sample_row]),
                label_column="target",
            )

            notebook_path = Path(tmpdir) / "automl_predictor_notebook.ipynb"
            assert notebook_path.exists()
            notebook = _load_notebook(tmpdir)
            assert "cells" in notebook

            # Cell 6: pipeline_name and run_id (experiment run details)
            cell6_sources = notebook["cells"][6]["source"]
            assert "<RUN_ID>" not in "".join(cell6_sources)
            assert "run-xyz-456" in "".join(cell6_sources)
            assert "<PIPELINE_NAME>" not in "".join(cell6_sources)
            assert "my-autogluon-pipeline" in "".join(cell6_sources)  # suffix stripped
            assert "abc123" not in "".join(cell6_sources)

            # Cell 10: model_name
            cell10_sources = notebook["cells"][10]["source"]
            assert "<MODEL_NAME>" not in "".join(cell10_sources)
            assert "WeightedEnsemble_L2" in "".join(cell10_sources)

            # Regression: sample_row at cell 20 (label column removed)
            cell20_sources = notebook["cells"][SAMPLE_ROW_CELL_REGRESSION]["source"]
            sample_row_str = "".join(cell20_sources)
            assert "<SAMPLE_ROW>" not in sample_row_str
            assert "feature_a" in sample_row_str
            assert "feature_b" in sample_row_str
            assert "target" not in sample_row_str

    def test_notebook_generation_classification_binary(self):
        """Test component generates classification notebook for problem_type='binary'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_row = {"f1": 1, "f2": 2, "label": 0}
            _run_component(
                tmpdir,
                problem_type="binary",
                model_name="WeightedEnsemble_L2",
                pipeline_name="my-pipeline-xyz",
                run_id="run-123",
                sample_row=json.dumps([sample_row]),
                label_column="label",
            )

            notebook = _load_notebook(tmpdir)
            # Classification template: run_id and model_name replaced
            cell6_text = "".join(notebook["cells"][6]["source"])
            assert "run-123" in cell6_text
            cell10_text = "".join(notebook["cells"][10]["source"])
            assert "WeightedEnsemble_L2" in cell10_text
            # Sample row at cell 21 for classification
            cell21_sources = notebook["cells"][SAMPLE_ROW_CELL_CLASSIFICATION]["source"]
            cell21_text = "".join(cell21_sources)
            assert "<SAMPLE_ROW>" not in cell21_text
            assert "f1" in cell21_text
            assert "f2" in cell21_text
            assert '"label"' not in cell21_text
            # Classification-specific: confusion matrix, predict_proba
            full_text = "".join("".join(c["source"]) for c in notebook["cells"])
            assert "confusion_matrix" in full_text
            assert "predict_proba" in full_text

    def test_notebook_generation_classification_multiclass(self):
        """Test component generates classification notebook for problem_type='multiclass'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_component(
                tmpdir,
                problem_type="multiclass",
                model_name="BestModel",
                sample_row=json.dumps([{"a": 1, "b": 2, "y": 0}]),
                label_column="y",
            )

            notebook = _load_notebook(tmpdir)
            cell21_text = "".join(notebook["cells"][SAMPLE_ROW_CELL_CLASSIFICATION]["source"])
            assert "<SAMPLE_ROW>" not in cell21_text
            assert "a" in cell21_text
            assert "b" in cell21_text
            assert '"y"' not in cell21_text

    def test_notebook_generation_invalid_problem_type_raises(self):
        """Test that invalid problem_type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid problem type: classification"):
                _run_component(tmpdir, problem_type="classification")

            with pytest.raises(ValueError, match="Invalid problem type: unknown"):
                _run_component(tmpdir, problem_type="unknown")

    def test_notebook_generation_pipeline_name_stripping(self):
        """Test that pipeline name has run suffix stripped (last segment after final hyphen)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_component(
                tmpdir,
                problem_type="regression",
                pipeline_name="prefix-middle-suffix",
                run_id="r1",
            )

            notebook = _load_notebook(tmpdir)
            cell6_text = "".join(notebook["cells"][6]["source"])
            assert "prefix-middle" in cell6_text
            assert "suffix" not in cell6_text

    def test_notebook_generation_sample_row_excludes_label_column(self):
        """Test that <SAMPLE_ROW> is replaced with sample row dict minus the label column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_row = {"f1": 10, "f2": 20, "label": 1}
            _run_component(
                tmpdir,
                problem_type="regression",
                sample_row=json.dumps([sample_row]),
                label_column="label",
            )

            notebook = _load_notebook(tmpdir)
            cell20_text = "".join(notebook["cells"][SAMPLE_ROW_CELL_REGRESSION]["source"])
            assert "<SAMPLE_ROW>" not in cell20_text
            assert "f1" in cell20_text
            assert "f2" in cell20_text
            assert '"label"' not in cell20_text

    def test_notebook_generation_regression_uses_predict(self):
        """Test regression notebook code uses predict(score_df)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_component(tmpdir, problem_type="regression")
            notebook = _load_notebook(tmpdir)
            full_text = "".join("".join(c["source"]) for c in notebook["cells"])
            assert "predict(score_df)" in full_text

    def test_notebook_generation_classification_uses_predict_proba(self):
        """Test classification notebook uses predict_proba()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_component(tmpdir, problem_type="binary")
            notebook = _load_notebook(tmpdir)
            full_text = "".join("".join(c["source"]) for c in notebook["cells"])
            assert "predict_proba(score_df)" in full_text

    def test_notebook_generation_notebook_valid_structure(self):
        """Test generated notebook has valid nbformat structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_component(tmpdir, problem_type="regression")
            notebook = _load_notebook(tmpdir)
            assert notebook.get("nbformat") == 4
            assert "nbformat_minor" in notebook
            assert "cells" in notebook
            assert len(notebook["cells"]) > 0
            assert "metadata" in notebook
            assert "kernelspec" in notebook["metadata"]

    def test_notebook_generation_creates_output_directory(self):
        """Test that component creates parent directories for the notebook path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_subdir = Path(tmpdir) / "nested" / "output"
            mock_artifact = type("Artifact", (), {"path": str(output_subdir)})()
            notebook_generation.python_func(
                problem_type="regression",
                model_name="M",
                notebook_artifact=mock_artifact,
                pipeline_name="p",
                run_id="r",
                sample_row=json.dumps([{"x": 1}]),
                label_column="x",
            )

            assert (output_subdir / "automl_predictor_notebook.ipynb").exists()
