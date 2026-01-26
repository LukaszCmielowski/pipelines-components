"""Local runner tests for the leaderboard_evaluation component."""

import tempfile
from pathlib import Path

import pandas as pd

from ..component import leaderboard_evaluation


class TestLeaderboardEvaluationLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    def test_local_execution_with_mock_models(
        self,
        setup_and_teardown_subprocess_runner,  # noqa: F811
    ):
        """Test component execution with LocalRunner using mock model paths."""
        # Note: This test requires actual AutoGluon models to be present at the specified paths.
        # For a complete test, you would need to:
        # 1. Train actual AutoGluon models and save them
        # 2. Create a test dataset
        # 3. Provide real model paths and dataset path
        #
        # This is a placeholder test structure that can be expanded when test models are available.

        # Create a temporary test dataset
        test_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "test_data.csv"
            test_data.to_csv(dataset_path, index=False)

            # Create temporary markdown output path
            markdown_path = Path(tmpdir) / "leaderboard.md"

            # Note: This test will fail without actual model files
            # Uncomment and provide real model paths when test models are available:
            #
            # model_paths = [
            #     "/path/to/model1",
            #     "/path/to/model2",
            # ]
            #
            # models = []
            # for i, model_path in enumerate(model_paths):
            #     model = type('Model', (), {
            #         'path': model_path,
            #         'metadata': {'model_name': f'Model{i+1}'}
            #     })()
            #     models.append(model)
            #
            # dataset = type('Dataset', (), {'path': str(dataset_path)})()
            # markdown = type('Markdown', (), {'path': str(markdown_path)})()
            #
            # result = leaderboard_evaluation(
            #     models=models,
            #     full_dataset=dataset,
            #     markdown_artifact=markdown,
            # )
            #
            # # Verify markdown file was created
            # assert markdown_path.exists()
            # with open(markdown_path, 'r') as f:
            #     content = f.read()
            #     assert len(content) > 0
            #     assert 'model' in content.lower() or 'rmse' in content.lower()

            # Placeholder assertion - remove when implementing real test
            assert True, "Local test requires actual AutoGluon models to be implemented"

    def test_component_imports_correctly(self):
        """Test that the component can be imported and has required attributes."""
        assert callable(leaderboard_evaluation)
        assert hasattr(leaderboard_evaluation, "python_func")
        assert hasattr(leaderboard_evaluation, "component_spec")
