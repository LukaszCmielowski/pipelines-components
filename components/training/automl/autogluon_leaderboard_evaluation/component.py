from typing import List

from kfp import dsl


@dsl.component(
    base_image="quay.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9:rhoai-3.2",
)
def leaderboard_evaluation(
    models: List[dsl.Model],
    eval_metric: str,
    html_artifact: dsl.Output[dsl.HTML],
):
    """Evaluate multiple AutoGluon models and generate a leaderboard.

    This component aggregates the evaluation results of a list of trained AutoGluon TabularPredictor models
    and generates a html-formatted leaderboard ranking the models by their performance metrics.

    Args:
        models: A list of Model artifacts containing trained AutoGluon
            TabularPredictor models to evaluate. Each model should have
            metadata containing a "model_name" field.
        eval_metric: The name of the evaluation metric to use for ranking
            models in the leaderboard. This should match one of the metrics
            returned by the TabularPredictor's evaluate method (e.g., "accuracy"
            for classification, "root_mean_squared_error" for regression).
            The leaderboard will be sorted by this metric in descending order.
        html_artifact: Output artifact where the html-formatted
            leaderboard will be written. The leaderboard contains model names
            and their evaluation metrics.

    Raises:
        FileNotFoundError: If any model path or dataset path cannot be found.
        ValueError: If a model cannot be loaded or evaluated successfully.
        KeyError: If model metadata does not contain the required "model_name" field.

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_leaderboard_evaluation import (
            leaderboard_evaluation
        )

        @dsl.pipeline(name="model-evaluation-pipeline")
        def evaluation_pipeline(trained_models, test_data):
            leaderboard = leaderboard_evaluation(
                models=trained_models,
                eval_metric="root_mean_squared_error",
            )
            return leaderboard
    """
    import json
    from pathlib import Path

    import pandas as pd

    results = []
    for model in models:
        eval_results = json.load(
            (Path(model.path) / model.metadata["model_name"] / "metrics" / "metrics.json").open("r")
        )
        results.append({"model": model.metadata["model_name"]} | eval_results)

    with open(html_artifact.path, "w") as f:
        f.write(pd.DataFrame(results).sort_values(by=eval_metric, ascending=False).to_html())


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
