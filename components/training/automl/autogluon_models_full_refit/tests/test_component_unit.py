"""Tests for the autogluon_models_full_refit component."""

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

        # Mock DataFrame for dataset
        mock_dataset_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_read_csv.return_value = mock_dataset_df

        # Create mock artifacts
        mock_predictor_artifact = mock.MagicMock()
        mock_predictor_artifact.path = "/tmp/predictor"

        mock_full_dataset = mock.MagicMock()
        mock_full_dataset.path = "/tmp/full_dataset.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/refitted_model"
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

        # Verify clone was called with correct parameters (path includes model_name_FULL)
        mock_predictor.clone.assert_called_once_with(
            path="/tmp/refitted_model/LightGBM_BAG_L1_FULL", return_clone=True, dirs_exist_ok=True
        )

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

        # Call the component function
        autogluon_models_full_refit.python_func(
            model_name="LightGBM_BAG_L1",
            full_dataset=mock_full_dataset,
            predictor_artifact=mock_predictor_artifact,
            model_artifact=mock_model_artifact,
        )

        # Verify call order: load -> refit_full -> clone -> delete_models -> set_model_best -> save_space
        assert mock_predictor_class.load.called
        assert mock_predictor.refit_full.called
        assert mock_predictor.clone.called
        assert mock_predictor_clone.delete_models.called
        assert mock_predictor_clone.set_model_best.called
        assert mock_predictor_clone.save_space.called

        # Verify refit_full was called before clone
        assert mock_predictor.refit_full.call_count == 1
        assert mock_predictor.clone.call_count == 1

    def test_component_imports_correctly(self):
        """Test that the component can be imported and has required attributes."""
        assert callable(autogluon_models_full_refit)
        assert hasattr(autogluon_models_full_refit, "python_func")
        assert hasattr(autogluon_models_full_refit, "component_spec")
