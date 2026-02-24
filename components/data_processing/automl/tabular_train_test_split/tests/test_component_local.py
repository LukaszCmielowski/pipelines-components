"""Local runner tests for the tabular_train_test_split component."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ..component import tabular_train_test_split


@pytest.mark.skip(reason="Requires pandas and sklearn for real execution; use unit tests with mocks.")
class TestTrainTestSplitLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
        """Test component execution by running python_func with real temp paths (local execution)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "input.csv"
            input_csv.write_text(
                "feature1,feature2,target\n1,10,0\n2,20,1\n3,30,0\n4,40,1\n5,50,0\n6,60,1\n7,70,0\n8,80,1\n9,90,0\n10,100,1\n"
            )

            dataset = SimpleNamespace(path=str(input_csv))
            train_out = SimpleNamespace(path=str(Path(tmpdir) / "train.csv"), uri="train")
            test_out = SimpleNamespace(path=str(Path(tmpdir) / "test.csv"), uri="test")

            result = tabular_train_test_split.python_func(
                dataset=dataset,
                sampled_train_dataset=train_out,
                sampled_test_dataset=test_out,
                test_size=0.3,
            )

        assert result is not None
        assert hasattr(result, "sample_row")
        assert hasattr(result, "split_config")
        assert result.split_config["test_size"] == 0.3
        assert Path(train_out.path).exists()
        assert Path(test_out.path).exists()
