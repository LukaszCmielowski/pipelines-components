"""Tests for the tabular_data_loader component."""

import tempfile
from pathlib import Path
from unittest import mock

from ..component import automl_data_loader

# Mock boto3 and pandas so unit tests run without installing them
_MOCK_MODULES = {"boto3": mock.MagicMock(), "pandas": mock.MagicMock()}


class TestAutomlDataLoaderUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(automl_data_loader)
        assert hasattr(automl_data_loader, "python_func")

    @mock.patch.dict("sys.modules", _MOCK_MODULES)
    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"})
    @mock.patch("pandas.read_csv")
    @mock.patch("boto3.client")
    def test_component_with_default_parameters(self, mock_boto_client, mock_read_csv):
        """Test component with valid input parameters and mocked S3/pandas."""
        mock_df = mock.MagicMock()
        mock_df.sample.return_value = mock_df  # component calls .sample() then .to_csv()
        mock_read_csv.return_value = mock_df
        mock_boto_client.return_value.download_file = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            full_dataset = mock.MagicMock()
            full_dataset.path = str(Path(tmpdir) / "out.csv")

            result = automl_data_loader.python_func(
                file_key="data/train.csv",
                bucket_name="my-bucket",
                full_dataset=full_dataset,
            )

        assert result is not None
        assert hasattr(result, "sample_config")
        assert result.sample_config["n_samples"] == 500
