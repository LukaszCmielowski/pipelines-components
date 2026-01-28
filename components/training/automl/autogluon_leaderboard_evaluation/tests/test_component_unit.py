"""Tests for the leaderboard_evaluation component."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from ..component import leaderboard_evaluation


class TestLeaderboardEvaluationUnitTests:
    """Unit tests for component logic."""

    @mock.patch("pandas.DataFrame")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_leaderboard_evaluation_with_single_model(self, mock_predictor_class, mock_dataframe_class):
        """Test leaderboard evaluation with a single model."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.evaluate.return_value = {
            "root_mean_squared_error": 0.5,
            "mean_absolute_error": 0.4,
            "r2": 0.9,
        }
        mock_predictor_class.load.return_value = mock_predictor

        # Mock DataFrame operations - chain sort_values().to_markdown()
        mock_df_sorted = mock.MagicMock()
        mock_df_sorted.to_markdown.return_value = "| model | rmse |\n|-------|------|\n| Model1 | 0.5 |"

        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        # Create mock artifacts
        mock_model = mock.MagicMock()
        mock_model.path = "/tmp/model1"
        mock_model.metadata = {"model_name": "Model1"}

        mock_dataset = mock.MagicMock()
        mock_dataset.path = "/tmp/test_data.csv"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as tmp_file:
            tmp_path = tmp_file.name

        try:
            mock_markdown = mock.MagicMock()
            mock_markdown.path = tmp_path

            # Call the component function
            leaderboard_evaluation.python_func(
                models=[mock_model],
                eval_metric="root_mean_squared_error",
                full_dataset=mock_dataset,
                markdown_artifact=mock_markdown,
            )

            # Verify TabularPredictor.load was called with correct path
            mock_predictor_class.load.assert_called_once_with("/tmp/model1")

            # Verify evaluate was called with dataset path
            mock_predictor.evaluate.assert_called_once_with("/tmp/test_data.csv")

            # Verify DataFrame was created with correct data
            mock_dataframe_class.assert_called_once()
            call_args = mock_dataframe_class.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0]["model"] == "Model1"
            assert call_args[0]["root_mean_squared_error"] == 0.5
            assert call_args[0]["mean_absolute_error"] == 0.4
            assert call_args[0]["r2"] == 0.9

            # Verify sorting was done by RMSE
            mock_df.sort_values.assert_called_once_with(by="root_mean_squared_error", ascending=False)

            # Verify markdown was generated
            mock_df_sorted.to_markdown.assert_called_once()

            # Verify file was written with correct content
            with open(tmp_path, "r") as f:
                content = f.read()
                assert content == "| model | rmse |\n|-------|------|\n| Model1 | 0.5 |"

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @mock.patch("pandas.DataFrame")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_leaderboard_evaluation_with_multiple_models(self, mock_predictor_class, mock_dataframe_class):
        """Test leaderboard evaluation with multiple models."""
        # Setup mocks
        mock_predictor1 = mock.MagicMock()
        mock_predictor1.evaluate.return_value = {
            "root_mean_squared_error": 0.8,
            "mean_absolute_error": 0.6,
        }

        mock_predictor2 = mock.MagicMock()
        mock_predictor2.evaluate.return_value = {
            "root_mean_squared_error": 0.3,
            "mean_absolute_error": 0.2,
        }

        mock_predictor3 = mock.MagicMock()
        mock_predictor3.evaluate.return_value = {
            "root_mean_squared_error": 0.5,
            "mean_absolute_error": 0.4,
        }

        mock_predictor_class.load.side_effect = [
            mock_predictor1,
            mock_predictor2,
            mock_predictor3,
        ]

        # Mock DataFrame operations - chain sort_values().to_markdown()
        mock_df_sorted = mock.MagicMock()
        mock_df_sorted.to_markdown.return_value = "| model | rmse |\n|-------|------|"

        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        # Create mock artifacts
        mock_model1 = mock.MagicMock()
        mock_model1.path = "/tmp/model1"
        mock_model1.metadata = {"model_name": "Model1"}

        mock_model2 = mock.MagicMock()
        mock_model2.path = "/tmp/model2"
        mock_model2.metadata = {"model_name": "Model2"}

        mock_model3 = mock.MagicMock()
        mock_model3.path = "/tmp/model3"
        mock_model3.metadata = {"model_name": "Model3"}

        mock_dataset = mock.MagicMock()
        mock_dataset.path = "/tmp/test_data.csv"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as tmp_file:
            tmp_path = tmp_file.name

        try:
            mock_markdown = mock.MagicMock()
            mock_markdown.path = tmp_path

            # Call the component function
            leaderboard_evaluation.python_func(
                models=[mock_model1, mock_model2, mock_model3],
                eval_metric="root_mean_squared_error",
                full_dataset=mock_dataset,
                markdown_artifact=mock_markdown,
            )

            # Verify all models were loaded
            assert mock_predictor_class.load.call_count == 3
            assert mock_predictor_class.load.call_args_list[0][0][0] == "/tmp/model1"
            assert mock_predictor_class.load.call_args_list[1][0][0] == "/tmp/model2"
            assert mock_predictor_class.load.call_args_list[2][0][0] == "/tmp/model3"

            # Verify all models were evaluated
            assert mock_predictor1.evaluate.call_count == 1
            assert mock_predictor2.evaluate.call_count == 1
            assert mock_predictor3.evaluate.call_count == 1

            # Verify DataFrame was created with all results
            mock_dataframe_class.assert_called_once()
            call_args = mock_dataframe_class.call_args[0][0]
            assert len(call_args) == 3
            assert call_args[0]["model"] == "Model1"
            assert call_args[0]["root_mean_squared_error"] == 0.8
            assert call_args[1]["model"] == "Model2"
            assert call_args[1]["root_mean_squared_error"] == 0.3
            assert call_args[2]["model"] == "Model3"
            assert call_args[2]["root_mean_squared_error"] == 0.5

            # Verify sorting was called
            mock_df.sort_values.assert_called_once_with(by="root_mean_squared_error", ascending=False)

            # Verify markdown was generated
            mock_df_sorted.to_markdown.assert_called_once()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_leaderboard_evaluation_handles_missing_model_name(self, mock_predictor_class):
        """Test that KeyError is raised when model metadata lacks model_name."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.evaluate.return_value = {"root_mean_squared_error": 0.5}
        mock_predictor_class.load.return_value = mock_predictor

        # Create mock artifacts without model_name in metadata
        mock_model = mock.MagicMock()
        mock_model.path = "/tmp/model1"
        mock_model.metadata = {}  # Missing model_name

        mock_dataset = mock.MagicMock()
        mock_dataset.path = "/tmp/test_data.csv"

        mock_markdown = mock.MagicMock()
        mock_markdown.path = "/tmp/output.md"

        # Verify KeyError is raised
        with pytest.raises(KeyError, match="model_name"):
            leaderboard_evaluation.python_func(
                models=[mock_model],
                eval_metric="root_mean_squared_error",
                full_dataset=mock_dataset,
                markdown_artifact=mock_markdown,
            )

    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_leaderboard_evaluation_handles_file_not_found(self, mock_predictor_class):
        """Test that FileNotFoundError is raised when model path doesn't exist."""
        # Setup mocks to raise FileNotFoundError
        mock_predictor_class.load.side_effect = FileNotFoundError("Model file not found")

        mock_model = mock.MagicMock()
        mock_model.path = "/nonexistent/model"
        mock_model.metadata = {"model_name": "Model1"}

        mock_dataset = mock.MagicMock()
        mock_dataset.path = "/tmp/test_data.csv"

        mock_markdown = mock.MagicMock()
        mock_markdown.path = "/tmp/output.md"

        # Verify FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            leaderboard_evaluation.python_func(
                models=[mock_model],
                eval_metric="root_mean_squared_error",
                full_dataset=mock_dataset,
                markdown_artifact=mock_markdown,
            )

    @mock.patch("pandas.DataFrame", create=True)
    @mock.patch("autogluon.tabular.TabularPredictor", create=True)
    def test_leaderboard_evaluation_sorts_by_rmse(self, mock_predictor_class, mock_dataframe_class):
        """Test that leaderboard is sorted by RMSE in descending order."""
        # Setup mocks with different RMSE values
        mock_predictor1 = mock.MagicMock()
        mock_predictor1.evaluate.return_value = {"root_mean_squared_error": 0.9}

        mock_predictor2 = mock.MagicMock()
        mock_predictor2.evaluate.return_value = {"root_mean_squared_error": 0.1}

        mock_predictor3 = mock.MagicMock()
        mock_predictor3.evaluate.return_value = {"root_mean_squared_error": 0.5}

        mock_predictor_class.load.side_effect = [
            mock_predictor1,
            mock_predictor2,
            mock_predictor3,
        ]

        # Mock DataFrame - chain sort_values().to_markdown()
        mock_df_sorted = mock.MagicMock()
        mock_df_sorted.to_markdown.return_value = "sorted markdown"

        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        # Create mock artifacts
        mock_models = []
        for i in range(3):
            mock_model = mock.MagicMock()
            mock_model.path = f"/tmp/model{i + 1}"
            mock_model.metadata = {"model_name": f"Model{i + 1}"}
            mock_models.append(mock_model)

        mock_dataset = mock.MagicMock()
        mock_dataset.path = "/tmp/test_data.csv"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as tmp_file:
            tmp_path = tmp_file.name

        try:
            mock_markdown = mock.MagicMock()
            mock_markdown.path = tmp_path

            leaderboard_evaluation.python_func(
                models=mock_models,
                eval_metric="root_mean_squared_error",
                full_dataset=mock_dataset,
                markdown_artifact=mock_markdown,
            )

            # Verify sorting was called with correct parameters
            mock_df.sort_values.assert_called_once_with(by="root_mean_squared_error", ascending=False)

            # Verify markdown was generated
            mock_df_sorted.to_markdown.assert_called_once()

            # Verify file was written
            with open(tmp_path, "r") as f:
                content = f.read()
                assert content == "sorted markdown"

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @mock.patch("pandas.DataFrame", create=True)
    @mock.patch("autogluon.tabular.TabularPredictor", create=True)
    def test_leaderboard_evaluation_writes_markdown_file(self, mock_predictor_class, mock_dataframe_class):
        """Test that markdown file is written correctly."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.evaluate.return_value = {
            "root_mean_squared_error": 0.5,
            "mean_absolute_error": 0.4,
        }
        mock_predictor_class.load.return_value = mock_predictor

        # Mock DataFrame - chain sort_values().to_markdown()
        expected_markdown = "| model | rmse | mae |\n|-------|------|-----|\n| Model1 | 0.5 | 0.4 |"
        mock_df_sorted = mock.MagicMock()
        mock_df_sorted.to_markdown.return_value = expected_markdown

        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        # Create mock artifacts
        mock_model = mock.MagicMock()
        mock_model.path = "/tmp/model1"
        mock_model.metadata = {"model_name": "Model1"}

        mock_dataset = mock.MagicMock()
        mock_dataset.path = "/tmp/test_data.csv"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as tmp_file:
            tmp_path = tmp_file.name

        try:
            mock_markdown = mock.MagicMock()
            mock_markdown.path = tmp_path

            leaderboard_evaluation.python_func(
                models=[mock_model],
                eval_metric="root_mean_squared_error",
                full_dataset=mock_dataset,
                markdown_artifact=mock_markdown,
            )

            # Verify DataFrame was created
            mock_dataframe_class.assert_called_once()
            call_args = mock_dataframe_class.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0]["model"] == "Model1"
            assert call_args[0]["root_mean_squared_error"] == 0.5
            assert call_args[0]["mean_absolute_error"] == 0.4

            # Verify sorting was called
            mock_df.sort_values.assert_called_once_with(by="root_mean_squared_error", ascending=False)

            # Verify markdown was generated
            mock_df_sorted.to_markdown.assert_called_once()

            # Verify file was written with correct content
            with open(tmp_path, "r") as f:
                content = f.read()
                assert content == expected_markdown

        finally:
            Path(tmp_path).unlink(missing_ok=True)
