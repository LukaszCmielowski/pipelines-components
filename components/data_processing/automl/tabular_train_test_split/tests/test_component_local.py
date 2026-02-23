"""Local runner tests for the tabular_train_test_split component."""

import pytest

from ..component import tabular_train_test_split


class TestTrainTestSplitLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    @pytest.mark.skip(reason="LocalRunner expects Dataset artifact for 'dataset' input, not a path string.")
    def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
        """Test component execution with LocalRunner."""
        result = tabular_train_test_split(dataset="path/to/dataset.csv", test_size=0.3)
        assert result is not None
