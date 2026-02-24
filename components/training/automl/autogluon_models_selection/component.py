from typing import List, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
    packages_to_install=[
        "autogluon.tabular==1.5.0",
        "catboost==1.2.8",
        "fastai==2.8.5",
        "lightgbm==4.6.0",
        "torch==2.9.1",
        "xgboost==3.1.3",
    ],
)
def models_selection(
    label_column: str,
    task_type: str,
    top_n: int,
    train_data: dsl.Input[dsl.Dataset],
    test_data: dsl.Input[dsl.Dataset],
    workspace_path: str,
) -> NamedTuple("outputs", top_models=List[str], eval_metric=str, predictor_path=str, model_config=dict):  # noqa: E501
    """Build multiple AutoGluon models and select top performers.

    This component trains multiple machine learning models using AutoGluon's
    ensembling approach (stacking and bagging) on sampled training data, then
    evaluates them on test data to identify the top N performing models.

    The component uses AutoGluon's TabularPredictor which automatically trains
    various model types (neural networks, tree-based models, linear models, etc.)
    and combines them using stacking with multiple levels and bagging. After
    training, models are evaluated on the test dataset and ranked by performance.
    The top N models are selected and their names are returned for use in
    subsequent refitting stages. The predictor is saved under workspace_path.

    This component is part of a two-stage training pipeline where models are
    first built and evaluated on sampled data (for efficiency), then the best
    candidates are refitted on the full dataset for optimal performance.

    Args:
        label_column: Name of the target/label column in the training and test datasets; used as the prediction target.
        task_type: Machine learning task type: "binary", "multiclass" (classification) or "regression"; determines metrics and model types.
        top_n: Number of top-performing models to select from the leaderboard; must be a positive integer.
        train_data: Dataset artifact (CSV) with training data; must include label_column and all feature columns.
        test_data: Dataset artifact (CSV) with test data for evaluation and leaderboard; schema should match training data.
        workspace_path: Path to the workspace directory where the TabularPredictor is saved (workspace_path / autogluon_predictor); returned as predictor_path.

    Returns:
        A NamedTuple with: top_models (List[str]): top N model names from leaderboard; eval_metric (str): metric used by TabularPredictor (e.g. "accuracy", "r2"); predictor_path (str): path to saved predictor (workspace_path / autogluon_predictor); model_config (dict): preset, eval metric, and time limit.

    Raises:
        FileNotFoundError: If the train_data or test_data paths cannot be found.
        ValueError: If label_column is missing, task_type is invalid, top_n is not positive, or training fails.
        KeyError: If required columns are missing from the datasets.

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_models_selection import (
            models_selection
        )

        @dsl.pipeline(name="model-selection-pipeline")
        def selection_pipeline(train_data, test_data, workspace_path):
            "Select top 3 models from training."
            result = models_selection(
                label_column="price",
                task_type="regression",
                top_n=3,
                train_data=train_data,
                test_data=test_data,
                workspace_path=workspace_path,
            )
            # result.top_models, result.eval_metric, result.predictor_path
            return result
    """  # noqa: E501
    from pathlib import Path

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    # Set constants
    DEFAULT_PRESET = "medium_quality"
    DEFAULT_TIME_LIMIT = 60 * 60  # 60 * 60 = 3600 seconds = 1 hour

    # Read the data
    train_data_df = pd.read_csv(train_data.path)
    test_data_df = pd.read_csv(test_data.path)

    eval_metric = "r2" if task_type == "regression" else "accuracy"

    predictor_path = Path(workspace_path) / "autogluon_predictor"
    predictor = TabularPredictor(
        problem_type=task_type,
        label=label_column,
        eval_metric=eval_metric,
        path=predictor_path,
        verbosity=2,
    ).fit(
        train_data=train_data_df,
        num_stack_levels=1,  # TODO: discuss optimal value
        num_bag_folds=2,
        use_bag_holdout=True,
        holdout_frac=0.2,  # 0.2 = 20% of the data is used for validation
        time_limit=DEFAULT_TIME_LIMIT,
        presets=DEFAULT_PRESET,
    )

    leaderboard = predictor.leaderboard(test_data_df)
    top_n_models = leaderboard.head(top_n)["model"].values.tolist()

    outputs = NamedTuple("outputs", top_models=List[str], eval_metric=str, predictor_path=str, model_config=dict)
    return outputs(
        top_models=top_n_models,
        eval_metric=str(predictor.eval_metric),
        predictor_path=str(predictor_path),
        model_config={"preset": DEFAULT_PRESET, "eval_metric": eval_metric, "time_limit": DEFAULT_TIME_LIMIT},
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        models_selection,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
