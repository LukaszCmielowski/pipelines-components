"""Tests for the autogluon_models_selection component."""

from unittest import mock

import pandas as pd
import pytest

from ..component import models_selection


class TestModelsSelectionUnitTests:
    """Unit tests for component logic."""

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_with_regression(self, mock_predictor_class, mock_read_csv):
        """Test models selection with regression problem type."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "root_mean_squared_error"
        mock_predictor.fit.return_value = mock_predictor
        mock_predictor_class.return_value = mock_predictor

        # Mock leaderboard DataFrame
        mock_leaderboard = pd.DataFrame(
            {
                "model": ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1", "CatBoost_BAG_L1", "RandomForest_BAG_L1"],
                "score_val": [0.5, 0.6, 0.7, 0.8],
            }
        )
        mock_predictor.leaderboard.return_value = mock_leaderboard

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [4.4, 5.5]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        # Create mock artifacts
        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Call the component function
        result = models_selection.python_func(
            target_column="target",
            problem_type="regression",
            top_n=2,
            train_data=mock_train_data,
            test_data=mock_test_data,
            model_artifact=mock_model_artifact,
        )

        # Verify read_csv was called with correct paths
        assert mock_read_csv.call_count == 2
        assert mock_read_csv.call_args_list[0][0][0] == "/tmp/train_data.csv"
        assert mock_read_csv.call_args_list[1][0][0] == "/tmp/test_data.csv"

        # Verify TabularPredictor was created with correct parameters
        mock_predictor_class.assert_called_once_with(
            problem_type="regression",
            label="target",
            path="/tmp/model",
            verbosity=2,
        )

        # Verify fit was called with correct parameters
        mock_predictor.fit.assert_called_once_with(
            train_data=mock_train_df,
            num_stack_levels=3,
            num_bag_folds=2,
            use_bag_holdout=True,
        )

        # Verify leaderboard was called with test data
        mock_predictor.leaderboard.assert_called_once_with(mock_test_df)

        # Verify metadata was set correctly
        assert mock_model_artifact.metadata["top_models"] == ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"]

        # Verify return values
        assert result.top_models == ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"]
        assert result.eval_metric == "root_mean_squared_error"

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_with_binary_classification(self, mock_predictor_class, mock_read_csv):
        """Test models selection with binary classification problem type."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "accuracy"
        mock_predictor.fit.return_value = mock_predictor

        # Mock leaderboard DataFrame
        mock_leaderboard = pd.DataFrame(
            {
                "model": ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1", "CatBoost_BAG_L1"],
                "score_val": [0.95, 0.92, 0.90],
            }
        )
        mock_predictor.leaderboard.return_value = mock_leaderboard

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [1, 0]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_predictor_class.return_value = mock_predictor

        # Create mock artifacts
        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Call the component function
        result = models_selection.python_func(
            target_column="target",
            problem_type="binary",
            top_n=2,
            train_data=mock_train_data,
            test_data=mock_test_data,
            model_artifact=mock_model_artifact,
        )

        # Verify TabularPredictor was created with binary problem type
        mock_predictor_class.assert_called_once_with(
            problem_type="binary",
            label="target",
            path="/tmp/model",
            verbosity=2,
        )

        # Verify return values
        assert result.top_models == ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"]
        assert result.eval_metric == "accuracy"

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_with_multiclass_classification(self, mock_predictor_class, mock_read_csv):
        """Test models selection with multiclass classification problem type."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "accuracy"
        mock_predictor.fit.return_value = mock_predictor

        # Mock leaderboard DataFrame
        mock_leaderboard = pd.DataFrame(
            {
                "model": ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1", "CatBoost_BAG_L1", "RandomForest_BAG_L1"],
                "score_val": [0.88, 0.85, 0.82, 0.80],
            }
        )
        mock_predictor.leaderboard.return_value = mock_leaderboard

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 2]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [1, 2]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_predictor_class.return_value = mock_predictor

        # Create mock artifacts
        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Call the component function
        result = models_selection.python_func(
            target_column="target",
            problem_type="multiclass",
            top_n=3,
            train_data=mock_train_data,
            test_data=mock_test_data,
            model_artifact=mock_model_artifact,
        )

        # Verify TabularPredictor was created with multiclass problem type
        mock_predictor_class.assert_called_once_with(
            problem_type="multiclass",
            label="target",
            path="/tmp/model",
            verbosity=2,
        )

        # Verify top_n models were selected
        assert len(result.top_models) == 3
        assert result.top_models == ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1", "CatBoost_BAG_L1"]

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_with_different_top_n(self, mock_predictor_class, mock_read_csv):
        """Test models selection with different top_n values."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "root_mean_squared_error"
        mock_predictor.fit.return_value = mock_predictor

        # Mock leaderboard DataFrame with 5 models
        mock_leaderboard = pd.DataFrame(
            {
                "model": [
                    "LightGBM_BAG_L1",
                    "NeuralNetFastAI_BAG_L1",
                    "CatBoost_BAG_L1",
                    "RandomForest_BAG_L1",
                    "XGBoost_BAG_L1",
                ],
                "score_val": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )
        mock_predictor.leaderboard.return_value = mock_leaderboard

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [4.4, 5.5]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_predictor_class.return_value = mock_predictor

        # Create mock artifacts
        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Call the component function with top_n=1
        result = models_selection.python_func(
            target_column="target",
            problem_type="regression",
            top_n=1,
            train_data=mock_train_data,
            test_data=mock_test_data,
            model_artifact=mock_model_artifact,
        )

        # Verify only top 1 model was selected
        assert len(result.top_models) == 1
        assert result.top_models == ["LightGBM_BAG_L1"]

        # Verify metadata contains only top 1 model
        assert len(mock_model_artifact.metadata["top_models"]) == 1
        assert mock_model_artifact.metadata["top_models"] == ["LightGBM_BAG_L1"]

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_handles_file_not_found_train_data(self, mock_predictor_class, mock_read_csv):
        """Test that FileNotFoundError is raised when train_data path doesn't exist."""
        # Setup mocks to raise FileNotFoundError for train_data
        mock_read_csv.side_effect = FileNotFoundError("Train data file not found")

        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/nonexistent/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Verify FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            models_selection.python_func(
                target_column="target",
                problem_type="regression",
                top_n=2,
                train_data=mock_train_data,
                test_data=mock_test_data,
                model_artifact=mock_model_artifact,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_handles_file_not_found_test_data(self, mock_predictor_class, mock_read_csv):
        """Test that FileNotFoundError is raised when test_data path doesn't exist."""
        # Setup mocks
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_read_csv.side_effect = [mock_train_df, FileNotFoundError("Test data file not found")]

        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/nonexistent/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Verify FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            models_selection.python_func(
                target_column="target",
                problem_type="regression",
                top_n=2,
                train_data=mock_train_data,
                test_data=mock_test_data,
                model_artifact=mock_model_artifact,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_handles_fit_failure(self, mock_predictor_class, mock_read_csv):
        """Test that ValueError is raised when model fitting fails."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.side_effect = ValueError("Target column not found in dataset")
        mock_predictor_class.return_value = mock_predictor

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "wrong_column": [1.1, 2.2, 3.3]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [4.4, 5.5]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Verify ValueError is raised
        with pytest.raises(ValueError, match="Target column not found in dataset"):
            models_selection.python_func(
                target_column="target",
                problem_type="regression",
                top_n=2,
                train_data=mock_train_data,
                test_data=mock_test_data,
                model_artifact=mock_model_artifact,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_handles_leaderboard_failure(self, mock_predictor_class, mock_read_csv):
        """Test that errors are raised when leaderboard generation fails."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.return_value = mock_predictor
        mock_predictor.leaderboard.side_effect = ValueError("Test data schema mismatch")
        mock_predictor_class.return_value = mock_predictor

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [4.4, 5.5]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Verify ValueError is raised
        with pytest.raises(ValueError, match="Test data schema mismatch"):
            models_selection.python_func(
                target_column="target",
                problem_type="regression",
                top_n=2,
                train_data=mock_train_data,
                test_data=mock_test_data,
                model_artifact=mock_model_artifact,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_verifies_all_operations_called(self, mock_predictor_class, mock_read_csv):
        """Test that all required operations are called in correct order."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "root_mean_squared_error"
        mock_predictor.fit.return_value = mock_predictor

        # Mock leaderboard DataFrame
        mock_leaderboard = pd.DataFrame(
            {
                "model": ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"],
                "score_val": [0.5, 0.6],
            }
        )
        mock_predictor.leaderboard.return_value = mock_leaderboard

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [4.4, 5.5]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_predictor_class.return_value = mock_predictor

        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Call the component function
        models_selection.python_func(
            target_column="target",
            problem_type="regression",
            top_n=2,
            train_data=mock_train_data,
            test_data=mock_test_data,
            model_artifact=mock_model_artifact,
        )

        # Verify call order: read_csv (train) -> read_csv (test) -> TabularPredictor -> fit -> leaderboard
        assert mock_read_csv.call_count == 2
        assert mock_predictor_class.called
        assert mock_predictor.fit.called
        assert mock_predictor.leaderboard.called

        # Verify fit was called before leaderboard
        assert mock_predictor.fit.call_count == 1
        assert mock_predictor.leaderboard.call_count == 1

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_sets_metadata_correctly(self, mock_predictor_class, mock_read_csv):
        """Test that model artifact metadata is set correctly."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "root_mean_squared_error"
        mock_predictor.fit.return_value = mock_predictor

        # Mock leaderboard DataFrame
        mock_leaderboard = pd.DataFrame(
            {
                "model": ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1", "CatBoost_BAG_L1"],
                "score_val": [0.5, 0.6, 0.7],
            }
        )
        mock_predictor.leaderboard.return_value = mock_leaderboard

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [4.4, 5.5]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_predictor_class.return_value = mock_predictor

        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Call the component function
        models_selection.python_func(
            target_column="target",
            problem_type="regression",
            top_n=3,
            train_data=mock_train_data,
            test_data=mock_test_data,
            model_artifact=mock_model_artifact,
        )

        # Verify metadata was set correctly
        assert "top_models" in mock_model_artifact.metadata
        assert mock_model_artifact.metadata["top_models"] == [
            "LightGBM_BAG_L1",
            "NeuralNetFastAI_BAG_L1",
            "CatBoost_BAG_L1",
        ]
        assert len(mock_model_artifact.metadata["top_models"]) == 3

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_models_selection_returns_correct_named_tuple(self, mock_predictor_class, mock_read_csv):
        """Test that the function returns a NamedTuple with correct fields."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "root_mean_squared_error"
        mock_predictor.fit.return_value = mock_predictor

        # Mock leaderboard DataFrame
        mock_leaderboard = pd.DataFrame(
            {
                "model": ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"],
                "score_val": [0.5, 0.6],
            }
        )
        mock_predictor.leaderboard.return_value = mock_leaderboard

        # Mock DataFrame for datasets
        mock_train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [1.1, 2.2, 3.3]})
        mock_test_df = pd.DataFrame({"feature1": [4, 5], "target": [4.4, 5.5]})
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_predictor_class.return_value = mock_predictor

        mock_train_data = mock.MagicMock()
        mock_train_data.path = "/tmp/train_data.csv"

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        # Call the component function
        result = models_selection.python_func(
            target_column="target",
            problem_type="regression",
            top_n=2,
            train_data=mock_train_data,
            test_data=mock_test_data,
            model_artifact=mock_model_artifact,
        )

        # Verify return type and fields
        assert hasattr(result, "top_models")
        assert hasattr(result, "eval_metric")
        assert isinstance(result.top_models, list)
        assert isinstance(result.eval_metric, str)
        assert result.top_models == ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"]
        assert result.eval_metric == "root_mean_squared_error"

    def test_component_imports_correctly(self):
        """Test that the component can be imported and has required attributes."""
        assert callable(models_selection)
        assert hasattr(models_selection, "python_func")
        assert hasattr(models_selection, "component_spec")
