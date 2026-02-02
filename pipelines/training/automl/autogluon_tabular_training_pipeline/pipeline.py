from typing import Dict

from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.automl.tabular_data_loader import automl_data_loader
from kfp_components.components.data_processing.automl.train_test_split import train_test_split
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import leaderboard_evaluation
from kfp_components.components.training.automl.autogluon_models_full_refit import autogluon_models_full_refit
from kfp_components.components.training.automl.autogluon_models_selection import models_selection


@dsl.pipeline(
    name="autogluon-tabular-training-pipeline",
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
    train_data_secret_name: str, train_data_bucket_name: str, train_data_file_key: str, label_column: str, task_type: str, top_n: int = 3
):
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
        train_data_secret_name: The Kubernetes secret name with S3-compatible credentials for tabular data file access.
            The following keys are required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION.
        train_data_bucket_name: The name of the S3-compatible bucket containing the tabular data file.
            The bucket should be accessible using the AWS credentials configured in the
            'train_data_secret_name' Kubernetes secret.
        train_data_file_key: The key (path) of the data file within the S3 bucket. The file should
            be in CSV format and contain both feature columns and the target column.
        label_column: The name of the target/label column in the dataset. This column
            will be used as the prediction target for model training. The column must
            exist in the loaded dataset.
        task_type: The type of machine learning task. Supported values:
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
        refitted models, ranked by performance. The leaderboard uses the metric
        determined by task_type (e.g., accuracy for classification, r2 for regression)
        and can be used for model comparison and selection decisions.

    Raises:
        FileNotFoundError: If the S3 file cannot be found or accessed.
        ValueError: If the label_column is not found in the dataset, task_type is
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
            train_data_secret_name="my-s3-secret",
            train_data_bucket_name="my-data-bucket",
            train_data_file_key="datasets/housing_prices.csv",
            label_column="price",
            task_type="regression",
            top_n=3,
        )
    """
    tabular_loader_task = automl_data_loader(bucket_name=train_data_bucket_name, file_key=train_data_file_key)

    use_secret_as_env(
        tabular_loader_task,
        secret_name=train_data_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT_URL": "AWS_ENDPOINT_URL",
            "AWS_REGION": "AWS_REGION",
        },
    )

    train_test_split_task = train_test_split(dataset=tabular_loader_task.outputs["full_dataset"], test_size=0.2)

    # Stage 1: Model Selection
    # Train multiple models on sampled data and select top N performers
    selection_task = models_selection(
        label_column=label_column,
        task_type=task_type,
        train_data=train_test_split_task.outputs["sampled_train_dataset"],
        test_data=train_test_split_task.outputs["sampled_test_dataset"],
        top_n=top_n,
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


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_tabular_training_pipeline,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
