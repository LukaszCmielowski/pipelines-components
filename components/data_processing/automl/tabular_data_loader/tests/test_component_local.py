"""Local runner tests for the tabular_data_loader component."""

from ..component import automl_data_loader


class TestAutomlDataLoaderLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
        """Test component execution with LocalRunner."""
        # TODO: Implement local runner tests for your component

        # Example test structure:
        result = automl_data_loader(input_param="test_value")

        # Add assertions about expected outputs if needed
        assert result is not None
