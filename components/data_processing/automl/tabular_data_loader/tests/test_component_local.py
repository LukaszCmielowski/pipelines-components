"""Local runner tests for the tabular_data_loader component."""

import pytest

from ..component import automl_data_loader


@pytest.mark.skip(reason="Component requires S3 credentials and bucket; run in environment with AWS/S3 configured.")
class TestAutomlDataLoaderLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
        """Test component execution with LocalRunner."""
        result = automl_data_loader(
            file_key="data/train.csv",
            bucket_name="test-bucket",
        )
        assert result is not None
