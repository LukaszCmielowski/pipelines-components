from typing import List, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="autogluon/autogluon:1.3.1-cpu-framework-ubuntu22.04-py3.11",
)
def models_selection(
    target_column: str,
    problem_type: str,
    top_n: int,
    train_data_regression: dsl.Input[dsl.Dataset],
    test_data_regression: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Output[dsl.Model],
) -> NamedTuple("outputs", top_models=List[str]):
    """Build multiple AutoGluon models and select top performers.

    This component trains multiple machine learning models using AutoGluon's
    ensembling approach (stacking and bagging) on sampled training data, then
    evaluates them on test data to identify the top N performing models.

    The component uses AutoGluon's TabularPredictor which automatically trains
    various model types (neural networks, tree-based models, linear models, etc.)
    and combines them using stacking with multiple levels and bagging. After
    training, models are evaluated on the test dataset and ranked by performance.
    The top N models are selected and their names are returned for use in
    subsequent refitting stages.

    This component is part of a two-stage training pipeline where models are
    first built and evaluated on sampled data (for efficiency), then the best
    candidates are refitted on the full dataset for optimal performance.

    Args:
        target_column: The name of the target/label column in the training
            and test datasets. This column will be used as the prediction target.
        problem_type: The type of machine learning problem. Supported values
            include "classification", "regression", or "time_series". This
            determines the evaluation metrics and model types AutoGluon will use.
        top_n: The number of top-performing models to select from the leaderboard.
            Only the top N models will be returned and promoted to the refit stage.
            Must be a positive integer.
        train_data_regression: A Dataset artifact containing the training data
            in CSV format. This data is used to train the AutoGluon models.
            The dataset should include the target_column and all feature columns.
        test_data_regression: A Dataset artifact containing the test data in
            CSV format. This data is used to evaluate model performance and
            generate the leaderboard. The dataset should match the schema of
            the training data.
        model_artifact: Output Model artifact where the trained TabularPredictor
            will be saved. The artifact metadata will contain a "top_models" key
            with the list of selected model names.

    Returns:
        A NamedTuple with the following fields:
            - top_models (List[str]): A list of model names (strings) representing
              the top N performing models selected from the leaderboard, ranked
              by performance on the test dataset.

    Raises:
        FileNotFoundError: If the train_data_regression or test_data_regression
            paths cannot be found.
        ValueError: If the target_column is not found in the datasets, the
            problem_type is invalid, top_n is not positive, or model training fails.
        KeyError: If required columns are missing from the datasets.

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_models_selection import (
            models_selection
        )

        @dsl.pipeline(name="model-selection-pipeline")
        def selection_pipeline(train_data, test_data):
            "Select top 3 models from training."
            result = models_selection(
                target_column="price",
                problem_type="regression",
                top_n=3,
                train_data_regression=train_data,
                test_data_regression=test_data
            )
            # result.top_models contains list of top 3 model names
            return result
    """
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    train_data_regression_df = pd.read_csv(train_data_regression.path)
    test_data_regression_df = pd.read_csv(test_data_regression.path)

    predictor_regression = TabularPredictor(
        problem_type=problem_type,
        label=target_column,
        path=model_artifact.path,
        verbosity=2,
    ).fit(
        train_data=train_data_regression_df,
        num_stack_levels=3,
        num_bag_folds=2,
        use_bag_holdout=True,
    )

    leaderboard = predictor_regression.leaderboard(test_data_regression_df)
    top_n_models = leaderboard.head(top_n)["model"].values.tolist()
    model_artifact.metadata["top_models"] = top_n_models

    outputs = NamedTuple("outputs", top_models=str)
    return outputs(top_models=top_n_models)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        models_selection,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
