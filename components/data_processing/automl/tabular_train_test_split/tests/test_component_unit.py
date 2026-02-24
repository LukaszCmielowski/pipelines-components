"""Tests for the tabular_train_test_split component."""

import tempfile
from pathlib import Path
from unittest import mock

from ..component import tabular_train_test_split


class TestTrainTestSplitUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(tabular_train_test_split)
        assert hasattr(tabular_train_test_split, "python_func")

    @mock.patch.dict(
        "sys.modules",
        {"pandas": mock.MagicMock(), "sklearn": mock.MagicMock(), "sklearn.model_selection": mock.MagicMock()},
    )
    @mock.patch("sklearn.model_selection.train_test_split")
    @mock.patch("pandas.read_csv")
    def test_component_with_default_parameters(self, mock_read_csv, mock_split):
        """Test component with valid input parameters and mocked pandas/sklearn."""
        mock_df = mock.MagicMock()
        mock_read_csv.return_value = mock_df

        mock_train = mock.MagicMock()
        mock_test = mock.MagicMock()
        mock_test.head.return_value.to_json.return_value = '[{"x":8,"y":0}]'
        mock_split.return_value = (mock_train, mock_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_dataset = mock.MagicMock()
            mock_dataset.path = str(Path(tmpdir) / "input.csv")

            mock_train_out = mock.MagicMock()
            mock_train_out.path = str(Path(tmpdir) / "train.csv")
            mock_train_out.uri = "train"

            mock_test_out = mock.MagicMock()
            mock_test_out.path = str(Path(tmpdir) / "test.csv")
            mock_test_out.uri = "test"

            result = tabular_train_test_split.python_func(
                dataset=mock_dataset,
                sampled_train_dataset=mock_train_out,
                sampled_test_dataset=mock_test_out,
                test_size=0.3,
            )

        assert hasattr(result, "sample_row")
        assert hasattr(result, "split_config")
        assert result.split_config["test_size"] == 0.3
        mock_read_csv.assert_called_once_with(mock_dataset.path)
        mock_split.assert_called_once()
