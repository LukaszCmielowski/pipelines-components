"""Local runner tests for the tabular_data_loader component."""

import pytest

from ..component import automl_data_loader


class TestAutomlDataLoaderLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    @pytest.mark.skip(reason="Requires S3 credentials, boto3, and a real S3 object; use unit tests with mocks.")
    def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
        """Test component execution with LocalRunner."""
        result = automl_data_loader(file_key="test.csv", bucket_name="test-bucket")
        assert result is not None
