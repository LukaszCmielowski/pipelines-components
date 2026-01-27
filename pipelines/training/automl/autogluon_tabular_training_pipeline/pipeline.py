from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.automl.tabular_data_loader import automl_data_loader
from kfp_components.components.data_processing.automl.train_test_split import train_test_split
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import leaderboard_evaluation
from kfp_components.components.training.automl.autogluon_models_full_refit import autogluon_models_full_refit
from kfp_components.components.training.automl.autogluon_models_selection import models_selection


@dsl.pipeline(
    name="autogluon-training-pipeline",
    description=(
        "End-to-end AutoGluon tabular training pipeline implementing a two-stage approach: "
        "first builds and selects top-performing models on sampled data, then refits them "
        "on the full dataset. The pipeline loads data from S3, splits it into train/test sets, "
        "trains multiple AutoGluon models using ensembling (stacking and bagging), selects the "
        "top N performers, refits each on the complete training data in parallel, and finally "
        "evaluates all refitted models to generate a comprehensive leaderboard with performance metrics."
    ),
)
def autogluon_tabular_training_pipeline(
    bucket_name: str,
    file_key: str,
    target_column: str,
    problem_type: str,
    top_n: int = 3,
) -> dsl.Markdown:
    """AutoGluon Tabular Training Pipeline.

    This pipeline implements an efficient two-stage training approach for AutoGluon tabular models
    that balances computational cost with model quality. The pipeline automates the complete
    machine learning workflow from data loading to final model evaluation.

    **Pipeline Stages:**

    1. **Data Loading**: Loads tabular data from an S3-compatible object storage bucket
       using AWS credentials configured via Kubernetes secrets. The component produces
       both a tabular_data artifact (for splitting) and a full_dataset artifact
       (for model refitting).

    2. **Data Splitting**: Splits the loaded tabular data into training and test sets
       using a configurable test size (default: 20% test, 80% train). The split is
       performed on the tabular_data artifact to create separate train and test
       datasets for model training and evaluation.

    3. **Model Selection**: Trains multiple AutoGluon models on the training data using
       AutoGluon's ensembling approach (stacking with 3 levels and bagging with 2 folds).
       The component automatically trains various model types including neural networks,
       tree-based models (XGBoost, LightGBM, CatBoost), and linear models. All models are
       evaluated on the test set and ranked by performance. The top N models are selected
       for the refitting stage.

    4. **Model Refitting**: Refits each of the top N selected models on the full dataset
       (the complete original dataset from the data loader). This stage runs in parallel
       (with parallelism of 2) to efficiently retrain multiple models. Each refitted model
       is saved with a "_FULL" suffix and optimized for deployment by removing unnecessary
       models and files.

    5. **Leaderboard Evaluation**: Evaluates all refitted models on the full dataset and
       generates a markdown-formatted leaderboard ranking models by their performance
       metrics. The leaderboard provides comprehensive evaluation results for model
       comparison and selection.

    **Two-Stage Training Benefits:**

    - **Efficient Exploration**: Initial model training uses the split training data
      with efficient ensembling rather than expensive hyperparameter optimization
    - **Optimal Performance**: Final models are refitted on the complete original dataset
      for maximum performance
    - **Parallel Efficiency**: Top models are refitted in parallel to minimize total
      pipeline execution time
    - **Production-Ready**: Refitted models are AutoGluon Predictors optimized and ready
      for deployment

    **AutoGluon Ensembling Approach:**

    The pipeline leverages AutoGluon's unique ensembling strategy that combines multiple
    model types using stacking and bagging rather than traditional hyperparameter optimization.
    This approach is more efficient and typically produces better results for tabular data
    by automatically:
    - Training diverse model families (neural networks, tree-based, linear)
    - Combining predictions using multi-level stacking
    - Using bootstrap aggregation (bagging) for robustness
    - Selecting optimal ensemble configurations

    Args:
        bucket_name: The name of the S3-compatible bucket containing the tabular data file.
            The bucket should be accessible using the AWS credentials configured in the
            'kubeflow-aws-secrets' Kubernetes secret.
        file_key: The key (path) of the data file within the S3 bucket. The file should
            be in CSV format and contain both feature columns and the target column.
        target_column: The name of the target/label column in the dataset. This column
            will be used as the prediction target for model training. The column must
            exist in the loaded dataset.
        problem_type: The type of machine learning problem. Supported values:
            - "binary" or "multiclass": For classification tasks
            - "regression": For regression tasks (predicting continuous values)
            This parameter determines the evaluation metrics and model types AutoGluon
            will use during training.
        top_n: The number of top-performing models to select and refit (default: 3).
            Must be a positive integer. Only the top N models from the initial training
            stage will be promoted to the refitting stage. Higher values increase pipeline
            execution time but provide more model options for final selection.

    Returns:
        A Markdown artifact containing the leaderboard with evaluation metrics for all
        refitted models, ranked by performance. The leaderboard is sorted by root mean
        squared error (RMSE) in descending order and can be used for model comparison
        and selection decisions.

    Raises:
        FileNotFoundError: If the S3 file cannot be found or accessed.
        ValueError: If the target_column is not found in the dataset, problem_type is
            invalid, top_n is not positive, or data splitting fails.
        KeyError: If required AWS credentials are missing from Kubernetes secrets or
            if required component outputs are not available.

    Example:
        from kfp import dsl
        from pipelines.training.automl.autogluon_tabular_training_pipeline import (
            autogluon_tabular_training_pipeline
        )

        # Compile and run the pipeline
        pipeline = autogluon_tabular_training_pipeline(
            bucket_name="my-data-bucket",
            file_key="datasets/housing_prices.csv",
            target_column="price",
            problem_type="regression",
            top_n=3
        )
    """
    tabular_loader_task = automl_data_loader(bucket_name=bucket_name, file_key=file_key)

    use_secret_as_env(
        tabular_loader_task,
        secret_name="kubeflow-aws-secrets",
        secret_key_to_env={
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "endpoint_url": "AWS_ENDPOINT_URL",
            "aws_region_name": "AWS_REGION",
        },
    )

    train_test_split_task = train_test_split(dataset=tabular_loader_task.outputs["full_dataset"], test_size=0.2)

    # Stage 1: Model Selection
    # Train multiple models on sampled data and select top N performers
    selection_task = models_selection(
        target_column=target_column,
        problem_type=problem_type,
        top_n=top_n,
        train_data=train_test_split_task.outputs["sampled_train_dataset"],
        test_data=train_test_split_task.outputs["sampled_test_dataset"],
    )

    # Stage 2: Model Refitting
    # Refit each top model on the full training dataset

    with dsl.ParallelFor(items=selection_task.outputs["top_models"], parallelism=2) as model_name:
        refit_full_task = autogluon_models_full_refit(
            model_name=model_name,
            full_dataset=tabular_loader_task.outputs["full_dataset"],
            predictor_artifact=selection_task.outputs["model_artifact"],
        )

    # Generate leaderboard
    leaderboard_evaluation_task = leaderboard_evaluation(
        models=dsl.Collected(refit_full_task.outputs["model_artifact"]),
        eval_metric=selection_task.outputs["eval_metric"],
        full_dataset=tabular_loader_task.outputs["full_dataset"],
    )

    return leaderboard_evaluation_task.outputs["markdown_artifact"]


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_tabular_training_pipeline,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
