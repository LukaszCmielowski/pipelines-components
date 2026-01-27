from typing import List

from kfp import dsl


@dsl.component(
    base_image="autogluon/autogluon:1.3.1-cpu-framework-ubuntu22.04-py3.11",
    # packages_to_install=["numpy", "pandas"],  # Add your dependencies here
)
def leaderboard_evaluation(
    models: List[dsl.Model],
    eval_metric: str,
    full_dataset: dsl.Input[dsl.Dataset],
    markdown_artifact: dsl.Output[dsl.Markdown],
):
    """Evaluate multiple AutoGluon models and generate a leaderboard.

    This component evaluates a list of trained AutoGluon TabularPredictor models
    on a full dataset and generates a markdown-formatted leaderboard ranking
    the models by their performance metrics. Each model is loaded, evaluated
    on the provided dataset, and the results are compiled into a sorted
    leaderboard table.

    The leaderboard is sorted by the specified evaluation metric in descending
    order, making it easy to identify the best-performing models. The output
    is written as a markdown table that can be used for reporting and
    model selection decisions.

    Args:
        models: A list of Model artifacts containing trained AutoGluon
            TabularPredictor models to evaluate. Each model should have
            metadata containing a "model_name" field.
        eval_metric: The name of the evaluation metric to use for ranking
            models in the leaderboard. This should match one of the metrics
            returned by the TabularPredictor's evaluate method (e.g., "accuracy"
            for classification, "root_mean_squared_error" for regression).
            The leaderboard will be sorted by this metric in descending order.
        full_dataset: A Dataset artifact containing the evaluation dataset
            on which all models will be evaluated. The dataset should be
            compatible with the models' training data format.
        markdown_artifact: Output artifact where the markdown-formatted
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
                full_dataset=test_data
            )
            return leaderboard
    """
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    results = []
    for model in models:
        predictor = TabularPredictor.load(model.path)
        eval_results = predictor.evaluate(full_dataset.path)
        results.append({"model": model.metadata["model_name"]} | eval_results)

    with open(markdown_artifact.path, "w") as f:
        f.write(pd.DataFrame(results).sort_values(by=eval_metric, ascending=False).to_markdown())


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
