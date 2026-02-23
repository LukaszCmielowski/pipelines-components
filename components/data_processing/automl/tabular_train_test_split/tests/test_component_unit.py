"""Tests for the tabular_train_test_split component."""

import tempfile
from pathlib import Path
from unittest import mock

from ..component import tabular_train_test_split

# Mock pandas and sklearn so unit tests run without installing them
_MOCK_MODULES = {
    "pandas": mock.MagicMock(),
    "sklearn": mock.MagicMock(),
    "sklearn.model_selection": mock.MagicMock(),
}


class TestTrainTestSplitUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(tabular_train_test_split)
        assert hasattr(tabular_train_test_split, "python_func")

    @mock.patch.dict("sys.modules", _MOCK_MODULES)
    @mock.patch("sklearn.model_selection.train_test_split")
    @mock.patch("pandas.read_csv")
    def test_component_with_default_parameters(self, mock_read_csv, mock_train_test_split):
        """Test component with valid input parameters and mocked pandas/sklearn."""
        mock_read_csv.return_value = mock.MagicMock()
        mock_train = mock.MagicMock()
        mock_test = mock.MagicMock()
        mock_test.head.return_value.to_json.return_value = '[{"col1":1,"col2":10}]'
        mock_train_test_split.return_value = (mock_train, mock_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_text("col1,col2\n1,10\n2,20\n3,30\n4,40\n5,50\n")
            dataset = mock.MagicMock()
            dataset.path = str(input_path)
            sampled_train = mock.MagicMock()
            sampled_train.path = str(Path(tmpdir) / "train.csv")
            sampled_train.uri = str(Path(tmpdir) / "train")
            sampled_test = mock.MagicMock()
            sampled_test.path = str(Path(tmpdir) / "test.csv")
            sampled_test.uri = str(Path(tmpdir) / "test")

            result = tabular_train_test_split.python_func(
                dataset=dataset,
                sampled_train_dataset=sampled_train,
                sampled_test_dataset=sampled_test,
                test_size=0.3,
            )

        assert result is not None
        assert hasattr(result, "sample_row")
        assert hasattr(result, "split_config")
        assert result.split_config["test_size"] == 0.3
