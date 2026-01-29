"""Tests for the autogluon_tabular_training_pipeline pipeline."""

import tempfile
from pathlib import Path

import pytest
from kfp import compiler

from ..pipeline import autogluon_tabular_training_pipeline


class TestAutogluonTabularTrainingPipelineUnitTests:
    """Unit tests for pipeline logic."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly imported."""
        assert callable(autogluon_tabular_training_pipeline)
        # Pipelines don't have python_func like components do
        # They are DAG definitions that need to be compiled

    def test_pipeline_compiles(self):
        """Test that the pipeline compiles successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            # Verify the file was created
            assert Path(tmp_path).exists()
        except Exception as e:
            pytest.fail(f"Pipeline compilation failed: {e}")
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_pipeline_with_default_parameters(self):
        """Test pipeline instantiation with default parameters."""
        # Test that the pipeline can be called with required parameters and defaults
        try:
            pipeline_instance = autogluon_tabular_training_pipeline(
                secret_name="train-data-secret",
                bucket_name="test-bucket",
                file_key="test-data.csv",
                target_column="target",
                problem_type="regression",
            )
            # Pipeline should return a pipeline definition, not raise an error
            assert pipeline_instance is not None
        except Exception as e:
            pytest.fail(f"Pipeline instantiation with defaults failed: {e}")

    def test_pipeline_with_custom_top_n(self):
        """Test pipeline with custom top_n parameter."""
        try:
            pipeline_instance = autogluon_tabular_training_pipeline(
                secret_name="train-data-secret",
                bucket_name="test-bucket",
                file_key="test-data.csv",
                target_column="target",
                problem_type="regression",
                top_n=5,
            )
            assert pipeline_instance is not None
        except Exception as e:
            pytest.fail(f"Pipeline instantiation with custom top_n failed: {e}")

    def test_pipeline_with_binary_classification(self):
        """Test pipeline with binary classification problem type."""
        try:
            pipeline_instance = autogluon_tabular_training_pipeline(
                secret_name="train-data-secret",
                bucket_name="test-bucket",
                file_key="test-data.csv",
                target_column="target",
                problem_type="binary",
                top_n=3,
            )
            assert pipeline_instance is not None
        except Exception as e:
            pytest.fail(f"Pipeline instantiation with binary classification failed: {e}")

    def test_pipeline_with_multiclass_classification(self):
        """Test pipeline with multiclass classification problem type."""
        try:
            pipeline_instance = autogluon_tabular_training_pipeline(
                secret_name="train-data-secret",
                bucket_name="test-bucket",
                file_key="test-data.csv",
                target_column="target",
                problem_type="multiclass",
                top_n=2,
            )
            assert pipeline_instance is not None
        except Exception as e:
            pytest.fail(f"Pipeline instantiation with multiclass classification failed: {e}")

    def test_pipeline_with_regression(self):
        """Test pipeline with regression problem type."""
        try:
            pipeline_instance = autogluon_tabular_training_pipeline(
                secret_name="train-data-secret",
                bucket_name="test-bucket",
                file_key="test-data.csv",
                target_column="target",
                problem_type="regression",
                top_n=4,
            )
            assert pipeline_instance is not None
        except Exception as e:
            pytest.fail(f"Pipeline instantiation with regression failed: {e}")

    def test_pipeline_compiles_with_all_parameters(self):
        """Test that the pipeline compiles with all parameters specified."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create pipeline instance with all parameters
            pipeline_instance = autogluon_tabular_training_pipeline(
                secret_name="train-data-secret",
                bucket_name="test-bucket",
                file_key="test-data.csv",
                target_column="target",
                problem_type="regression",
                top_n=3,
            )

            # Compile the pipeline instance
            compiler.Compiler().compile(
                pipeline_func=pipeline_instance,
                package_path=tmp_path,
            )
            assert Path(tmp_path).exists()
        except Exception as e:
            pytest.fail(f"Pipeline compilation with all parameters failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
