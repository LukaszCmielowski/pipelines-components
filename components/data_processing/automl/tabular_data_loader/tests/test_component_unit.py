"""Tests for the tabular_data_loader component."""

import sys
import tempfile
from pathlib import Path
from unittest import mock

# So @mock.patch("boto3.client") can resolve without installing boto3
if "boto3" not in sys.modules:
    sys.modules["boto3"] = mock.MagicMock()
# Component also imports pandas; inject so it can run without installing pandas
if "pandas" not in sys.modules:
    sys.modules["pandas"] = mock.MagicMock()

from ..component import automl_data_loader


class TestAutomlDataLoaderUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(automl_data_loader)
        assert hasattr(automl_data_loader, "python_func")

    @mock.patch.dict(
        "os.environ",
        {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_S3_ENDPOINT": "https://s3.example.com",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    )
    @mock.patch("boto3.client")
    def test_component_with_default_parameters(self, mock_boto_client):
        """Test component with valid input parameters and mocked S3."""
        mock_s3 = mock.MagicMock()
        mock_boto_client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "full_dataset.csv"

            # Component downloads to full_dataset.path then reads; mock download_file to write a minimal CSV
            def write_csv(Bucket, Key, Filename):
                Path(Filename).write_text("a,b\n1,2\n3,4\n5,6\n")

            mock_s3.download_file.side_effect = write_csv

            mock_full_dataset = mock.MagicMock()
            mock_full_dataset.path = str(out_path)

            # Component does pd.read_csv(full_dataset.path) then .sample().to_csv(); configure pandas mock
            pd_mock = sys.modules["pandas"]
            mock_df = mock.MagicMock()
            mock_df.sample.return_value = mock_df
            pd_mock.read_csv.return_value = mock_df

            result = automl_data_loader.python_func(
                file_key="data/train.csv",
                bucket_name="my-bucket",
                full_dataset=mock_full_dataset,
            )

        assert result.sample_config["n_samples"] == 500
        mock_s3.download_file.assert_called_once_with("my-bucket", "data/train.csv", mock_full_dataset.path)
