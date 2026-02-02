"""Tests for the autogluon_models_full_refit component."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from ..component import autogluon_models_full_refit


class TestAutogluonModelsFullRefitUnitTests:
    """Unit tests for component logic."""

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_full_refit_with_valid_model(self, mock_predictor_class, mock_read_csv):
        """Test full refit with a valid model name."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor_class.load.return_value = mock_predictor
        eval_results = {"r2": 0.9, "root_mean_squared_error": 0.5}
        mock_predictor_clone.evaluate.return_value = eval_results
        feature_importance_dict = {"feature1": 0.1, "feature2": 0.05}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: feature_importance_dict)
        mock_predictor.problem_type = "regression"

        # Mock DataFrame for dataset
        mock_dataset_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_read_csv.return_value = mock_dataset_df

        # Create mock artifacts; use temp dir so we can verify metrics files are written
        mock_predictor_artifact = mock.MagicMock()
        mock_predictor_artifact.path = "/tmp/predictor"

        mock_full_dataset = mock.MagicMock()
        mock_full_dataset.path = "/tmp/full_dataset.csv"

        model_output_dir = tempfile.mkdtemp()
        try:
            mock_model_artifact = mock.MagicMock()
            mock_model_artifact.path = model_output_dir
            mock_model_artifact.metadata = {}

            # Call the component function
            autogluon_models_full_refit.python_func(
                model_name="LightGBM_BAG_L1",
                full_dataset=mock_full_dataset,
                predictor_artifact=mock_predictor_artifact,
                model_artifact=mock_model_artifact,
            )

            # Verify read_csv was called with correct path
            mock_read_csv.assert_called_once_with("/tmp/full_dataset.csv")

            # Verify TabularPredictor.load was called with correct path
            mock_predictor_class.load.assert_called_once_with("/tmp/predictor")

            # Verify refit_full was called with correct parameters
            mock_predictor.refit_full.assert_called_once_with(train_data_extra=mock_dataset_df, model="LightGBM_BAG_L1")

            # Verify evaluate and feature_importance called with full dataset dataframe (on clone)
            mock_predictor_clone.evaluate.assert_called_once_with(mock_dataset_df)
            mock_predictor_clone.feature_importance.assert_called_once_with(mock_dataset_df)

            # Verify clone was called with correct parameters (path is Path; includes model_name_FULL)
            mock_predictor.clone.assert_called_once()
            call_kw = mock_predictor.clone.call_args[1]
            assert Path(call_kw["path"]) == Path(model_output_dir) / "LightGBM_BAG_L1_FULL"
            assert call_kw["return_clone"] is True
            assert call_kw["dirs_exist_ok"] is True

            # Verify delete_models was called with correct models to keep
            mock_predictor_clone.delete_models.assert_called_once_with(
                models_to_keep=["LightGBM_BAG_L1", "LightGBM_BAG_L1_FULL"]
            )

            # Verify set_model_best was called with correct model
            mock_predictor_clone.set_model_best.assert_called_once_with(model="LightGBM_BAG_L1_FULL", save_trainer=True)

            # Verify save_space was called
            mock_predictor_clone.save_space.assert_called_once()

            # Verify metadata was set correctly
            assert mock_model_artifact.metadata["model_name"] == "LightGBM_BAG_L1_FULL"

            # Verify metrics files were written (under model_name_FULL/metrics/)
            metrics_dir = Path(model_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics"
            assert metrics_dir.exists()
            metrics_path = metrics_dir / "metrics.json"
            assert metrics_path.exists()
            assert json.loads(metrics_path.read_text()) == eval_results
            fi_path = metrics_dir / "feature_importance.json"
            assert fi_path.exists()
            assert json.loads(fi_path.read_text()) == feature_importance_dict
            # Regression: no confusion matrix
            assert not (metrics_dir / "confusion_matrix.json").exists()
        finally:
            shutil.rmtree(model_output_dir, ignore_errors=True)

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_full_refit_handles_file_not_found_predictor(self, mock_predictor_class, mock_read_csv):
        """Test that FileNotFoundError is raised when predictor path doesn't exist."""
        # Setup mocks to raise FileNotFoundError
        mock_predictor_class.load.side_effect = FileNotFoundError("Predictor file not found")

        mock_full_dataset = mock.MagicMock()
        mock_full_dataset.path = "/tmp/full_dataset.csv"

        mock_predictor_artifact = mock.MagicMock()
        mock_predictor_artifact.path = "/nonexistent/predictor"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/refitted_model"
        mock_model_artifact.metadata = {}

        # Verify FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            autogluon_models_full_refit.python_func(
                model_name="LightGBM_BAG_L1",
                full_dataset=mock_full_dataset,
                predictor_artifact=mock_predictor_artifact,
                model_artifact=mock_model_artifact,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_full_refit_handles_refit_failure(self, mock_predictor_class, mock_read_csv):
        """Test that ValueError is raised when refit_full fails."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.refit_full.side_effect = ValueError("Model not found in predictor")
        mock_predictor_class.load.return_value = mock_predictor

        # Mock DataFrame for dataset
        mock_dataset_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_read_csv.return_value = mock_dataset_df

        mock_full_dataset = mock.MagicMock()
        mock_full_dataset.path = "/tmp/full_dataset.csv"

        mock_predictor_artifact = mock.MagicMock()
        mock_predictor_artifact.path = "/tmp/predictor"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/refitted_model"
        mock_model_artifact.metadata = {}

        # Verify ValueError is raised
        with pytest.raises(ValueError, match="Model not found in predictor"):
            autogluon_models_full_refit.python_func(
                model_name="NonexistentModel",
                full_dataset=mock_full_dataset,
                predictor_artifact=mock_predictor_artifact,
                model_artifact=mock_model_artifact,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_full_refit_verifies_all_operations_called(self, mock_predictor_class, mock_read_csv):
        """Test that all required operations are called in correct order."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor_class.load.return_value = mock_predictor
        mock_predictor_clone.evaluate.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"feature1": 0.1})
        mock_predictor.problem_type = "regression"

        mock_dataset_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_read_csv.return_value = mock_dataset_df

        mock_full_dataset = mock.MagicMock()
        mock_full_dataset.path = "/tmp/full_dataset.csv"
        mock_predictor_artifact = mock.MagicMock()
        mock_predictor_artifact.path = "/tmp/predictor"
        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/refitted_model"
        mock_model_artifact.metadata = {}

        autogluon_models_full_refit.python_func(
            model_name="LightGBM_BAG_L1",
            full_dataset=mock_full_dataset,
            predictor_artifact=mock_predictor_artifact,
            model_artifact=mock_model_artifact,
        )

        # Verify call order: load -> refit_full -> clone -> evaluate -> feature_importance (on clone) -> ...
        assert mock_predictor_class.load.called
        assert mock_predictor.refit_full.called
        mock_predictor_clone.evaluate.assert_called_once_with(mock_dataset_df)
        mock_predictor_clone.feature_importance.assert_called_once_with(mock_dataset_df)
        assert mock_predictor.clone.called
        assert mock_predictor_clone.delete_models.called
        assert mock_predictor_clone.set_model_best.called
        assert mock_predictor_clone.save_space.called
        assert mock_predictor.refit_full.call_count == 1
        assert mock_predictor.clone.call_count == 1

    @mock.patch("autogluon.core.metrics.confusion_matrix")
    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_full_refit_writes_confusion_matrix_for_classification(
        self, mock_predictor_class, mock_read_csv, mock_confusion_matrix
    ):
        """Test that confusion matrix is written when problem_type is binary or multiclass."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor_class.load.return_value = mock_predictor
        mock_predictor_clone.evaluate.return_value = {"accuracy": 0.95}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"feature1": 0.1})
        mock_predictor.problem_type = "binary"
        mock_predictor_clone.predict.return_value = pd.Series([0, 1, 0])
        mock_predictor.label = "target"

        mock_dataset_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        mock_read_csv.return_value = mock_dataset_df
        confusion_matrix_dict = {"0": {"0": 2, "1": 0}, "1": {"0": 1, "1": 0}}
        mock_confusion_matrix.return_value = mock.MagicMock(to_dict=lambda: confusion_matrix_dict)

        mock_full_dataset = mock.MagicMock()
        mock_full_dataset.path = "/tmp/full_dataset.csv"
        mock_predictor_artifact = mock.MagicMock()
        mock_predictor_artifact.path = "/tmp/predictor"
        model_output_dir = tempfile.mkdtemp()
        try:
            mock_model_artifact = mock.MagicMock()
            mock_model_artifact.path = model_output_dir
            mock_model_artifact.metadata = {}

            autogluon_models_full_refit.python_func(
                model_name="LightGBM_BAG_L1",
                full_dataset=mock_full_dataset,
                predictor_artifact=mock_predictor_artifact,
                model_artifact=mock_model_artifact,
            )

            metrics_dir = Path(model_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics"
            cm_path = metrics_dir / "confusion_matrix.json"
            assert cm_path.exists()
            assert json.loads(cm_path.read_text()) == confusion_matrix_dict
        finally:
            shutil.rmtree(model_output_dir, ignore_errors=True)

    def test_component_imports_correctly(self):
        """Test that the component can be imported and has required attributes."""
        assert callable(autogluon_models_full_refit)
        assert hasattr(autogluon_models_full_refit, "python_func")
        assert hasattr(autogluon_models_full_refit, "component_spec")
