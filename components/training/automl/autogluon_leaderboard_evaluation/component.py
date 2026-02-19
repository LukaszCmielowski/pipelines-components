from typing import List, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
)
def leaderboard_evaluation(
    models: List[dsl.Model],
    eval_metric: str,
    html_artifact: dsl.Output[dsl.HTML],
) -> NamedTuple("outputs", best_model=str):
    """Evaluate multiple AutoGluon models and generate a leaderboard.

    This component aggregates evaluation results from a list of Model artifacts
    (reading pre-computed metrics from JSON) and generates an HTML-formatted
    leaderboard ranking the models by their performance metrics. Each model
    artifact is expected to contain metrics at
    model.path / model.metadata["model_name"] / metrics / metrics.json.

    Args:
        models: A list of Model artifacts. Each should have metadata containing
            a "model_name" field and metrics file at
            model.path / model_name / metrics / metrics.json.
        eval_metric: The name of the evaluation metric to use for ranking.
            Must match a key in the metrics JSON (e.g., "accuracy" for
            classification, "root_mean_squared_error" for regression).
            The leaderboard is sorted by this metric in descending order.
        html_artifact: Output artifact where the HTML-formatted leaderboard
            will be written. The leaderboard contains model names and their
            evaluation metrics.

    Raises:
        FileNotFoundError: If any model metrics path cannot be found.
        KeyError: If model metadata does not contain "model_name" or the
            metrics JSON does not contain the eval_metric key.

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_leaderboard_evaluation import (
            leaderboard_evaluation
        )

        @dsl.pipeline(name="model-evaluation-pipeline")
        def evaluation_pipeline(trained_models):
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

    leaderboard_df = pd.DataFrame(results).sort_values(by=eval_metric, ascending=False)
    with open(html_artifact.path, "w") as f:
        f.write(leaderboard_df.to_html())

    best_model = leaderboard_df.iloc[0]["model"]
    return NamedTuple("outputs", best_model=str)(best_model=best_model)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
