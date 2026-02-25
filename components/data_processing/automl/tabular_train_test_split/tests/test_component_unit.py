"""Tests for the tabular_train_test_split component."""

from unittest import mock

import pytest

from ..component import tabular_train_test_split


class TestTrainTestSplitUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(tabular_train_test_split)
        assert hasattr(tabular_train_test_split, "python_func")

    def test_component_with_default_parameters(self, tmp_path):
        """Test component with valid input parameters and default test_size."""
        pd = pytest.importorskip("pandas")

        # Create input CSV
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n")

        dataset = mock.MagicMock()
        dataset.path = str(input_csv)

        sampled_train_dataset = mock.MagicMock()
        sampled_train_dataset.path = str(tmp_path / "train.csv")
        sampled_train_dataset.uri = str(tmp_path / "train")

        sampled_test_dataset = mock.MagicMock()
        sampled_test_dataset.path = str(tmp_path / "test.csv")
        sampled_test_dataset.uri = str(tmp_path / "test")

        result = tabular_train_test_split.python_func(
            dataset=dataset,
            sampled_train_dataset=sampled_train_dataset,
            sampled_test_dataset=sampled_test_dataset,
        )

        assert hasattr(result, "sample_row")
        assert hasattr(result, "split_config")
        assert result.split_config["test_size"] == 0.3
        assert isinstance(result.sample_row, str)

        train_df = pd.read_csv(sampled_train_dataset.path)
        test_df = pd.read_csv(sampled_test_dataset.path)
        assert len(train_df) + len(test_df) == 4
        assert list(train_df.columns) == ["a", "b", "c"]
        assert list(test_df.columns) == ["a", "b", "c"]
