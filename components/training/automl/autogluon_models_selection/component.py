from typing import List, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=["autogluon.tabular[all]==1.5.0"],
)
def models_selection(
    label_column: str,
    task_type: str,
    top_n: int,
    train_data: dsl.Input[dsl.Dataset],
    test_data: dsl.Input[dsl.Dataset],
    workspace_path: str,
) -> NamedTuple("outputs", top_models=List[str], eval_metric=str, predictor_path=str):
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
        label_column: The name of the target/label column in the training
            and test datasets. This column will be used as the prediction target.
        task_type: The type of machine learning task. Supported values
            include "binary", "multiclass" (classification) or "regression". This
            determines the evaluation metrics and model types AutoGluon will use.
        top_n: The number of top-performing models to select from the leaderboard.
            Only the top N models will be returned and promoted to the refit stage.
            Must be a positive integer.
        train_data: A Dataset artifact containing the training data
            in CSV format. This data is used to train the AutoGluon models.
            The dataset should include the label_column and all feature columns.
        test_data: A Dataset artifact containing the test data in
            CSV format. This data is used to evaluate model performance and
            generate the leaderboard. The dataset should match the schema of
            the training data.
        workspace_path: Path (string) to the workspace directory where the
            trained TabularPredictor will be saved (under workspace_path /
            autogluon_predictor). This path is also returned as predictor_path
            for use by downstream components.

    Returns:
        A NamedTuple with the following fields:
            - top_models (List[str]): A list of model names (strings) representing
              the top N performing models selected from the leaderboard, ranked
              by performance on the test dataset.
            - eval_metric (str): The evaluation metric name used by the TabularPredictor
              to assess model performance. This metric is automatically determined
              based on the task_type (e.g., "accuracy" for classification,
              "r2" for regression).
            - predictor_path (str): The path to the saved TabularPredictor
              (workspace_path / autogluon_predictor), for use by downstream
              components such as autogluon_models_full_refit.

    Raises:
        FileNotFoundError: If the train_data or test_data
            paths cannot be found.
        ValueError: If the label_column is not found in the datasets, the
            task_type is invalid, top_n is not positive, or model training fails.
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
    """
    from pathlib import Path

    import pandas as pd
    from autogluon.tabular import TabularPredictor

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
    )

    leaderboard = predictor.leaderboard(test_data_df)
    top_n_models = leaderboard.head(top_n)["model"].values.tolist()

    outputs = NamedTuple("outputs", top_models=List[str], eval_metric=str, predictor_path=str)
    return outputs(top_models=top_n_models, eval_metric=str(predictor.eval_metric), predictor_path=str(predictor_path))


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        models_selection,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
